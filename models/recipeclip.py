"""
models/recipeclip.py — RecipeCLIP-10K dual-encoder model.

Architecture
============

Image branch
------------
  open_clip ViT-B/32 visual encoder  (frozen by default for E2/E3)
      ↓ 512-d
  Projection head:
      Linear(512 → hidden) → GELU → Dropout → Linear(hidden → embed_dim)
      → L2 Normalise
      ↓ embed_dim-d

Recipe branch  (E2/E3 — mean-pooled LMDB intrs)
------------------------------------------------------
  mean-pool of per-sentence MiniLM embeddings from LMDB  (→ 384-d by default)
  Projection head:
      Linear(text_dim → hidden) → GELU → Dropout → Linear(hidden → embed_dim)
      → L2 Normalise
      ↓ embed_dim-d

Both embeddings are L2-normalised before being returned so that dot product
equals cosine similarity, making them directly usable with SymmetricInfoNCE.

Quick usage
===========
    model = RecipeCLIP()
    img   = torch.randn(8, 3, 224, 224)   # pre-processed by open_clip.transform
    rec   = torch.randn(8, 384)            # mean-pooled MiniLM embeddings
    img_emb, rec_emb = model(img, rec)    # both (8, 256) L2-normalised

Freezing / unfreezing schedule
================================
    model.freeze_image_backbone()    # call before E2 training
    model.unfreeze_image_backbone()  # call when switching to E3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip_torch is required.  Install with:  pip install open_clip_torch"
    ) from e


# ---------------------------------------------------------------------------
# Projection head  (shared structure for both branches)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head.

    Linear(in_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → out_dim)

    The output is NOT normalised here; normalisation is applied by the parent
    encoder so that it can be toggled off for debugging.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    open_clip visual backbone + projection head.

    Args:
        clip_model_name  : open_clip model name, e.g. "ViT-B-32"
        pretrained       : open_clip pretrained tag, e.g. "laion2b_s34b_b79k"
        embed_dim        : final output dimension (after projection)
        proj_hidden_dim  : hidden size inside the projection head
        dropout          : dropout probability in the projection head
        freeze_backbone  : if True, CLIP visual encoder weights are frozen
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        embed_dim: int = 256,
        proj_hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load visual encoder from open_clip (we discard the text side).
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.backbone = model.visual          # visual encoder module
        self.clip_embed_dim = model.visual.output_dim  # 512 for ViT-B/32

        # Projection head: CLIP-dim → embed_dim
        self.proj = ProjectionHead(
            in_dim=self.clip_embed_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

        # Freeze / unfreeze backbone
        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze the CLIP visual backbone (call for E2)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Unfreeze the CLIP visual backbone (call when switching to E3+)."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        self.backbone.train()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor pre-processed by open_clip.transform, shape (B, 3, H, W)
        Returns:
            L2-normalised image embedding, shape (B, embed_dim)
        """
        # Backbone forward (may be in eval mode if frozen)
        with torch.set_grad_enabled(self.backbone.training):
            feat = self.backbone(x)          # (B, clip_embed_dim)

        emb = self.proj(feat)                # (B, embed_dim)
        return F.normalize(emb, dim=-1)      # (B, embed_dim), unit sphere


# ---------------------------------------------------------------------------
# Recipe encoder  (E2 / E3 — mean-pooled LMDB intrs)
# ---------------------------------------------------------------------------

class RecipeEncoder(nn.Module):
    """
    Recipe encoder for E2/E3.

    Input: mean-pooled per-sentence MiniLM embeddings already stored in LMDB
           as the `intrs` field.  The DataLoader pre-computes the mean, so
           this module receives a flat vector of shape (B, text_dim).

    Args:
        text_dim         : dimension of the incoming text embedding (384 for MiniLM)
        embed_dim        : final output dimension (same as ImageEncoder)
        proj_hidden_dim  : hidden size inside the projection head
        dropout          : dropout probability
    """

    def __init__(
        self,
        text_dim: int = 384,
        embed_dim: int = 256,
        proj_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = ProjectionHead(
            in_dim=text_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mean-pooled recipe embeddings, shape (B, text_dim)
        Returns:
            L2-normalised recipe embedding, shape (B, embed_dim)
        """
        emb = self.proj(x)               # (B, embed_dim)
        return F.normalize(emb, dim=-1)  # unit sphere


# ---------------------------------------------------------------------------
# Top-level RecipeCLIP model
# ---------------------------------------------------------------------------

class RecipeCLIP(nn.Module):
    """
    RecipeCLIP-10K dual-encoder model.

    Composes an ImageEncoder and a RecipeEncoder.  Both outputs are
    L2-normalised, making their dot product equal to cosine similarity.

    Args: (see ImageEncoder / RecipeEncoder for full docs)
        clip_model_name  : open_clip model identifier
        pretrained       : open_clip pretrained weight tag
        embed_dim        : shared output embedding dimension
        proj_hidden_dim  : hidden dim inside both projection heads
        dropout          : dropout rate in projection heads
        text_dim         : incoming recipe embedding dimension
        freeze_backbone  : whether to freeze the CLIP visual backbone
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        embed_dim: int = 256,
        proj_hidden_dim: int = 512,
        dropout: float = 0.1,
        text_dim: int = 384,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            clip_model_name=clip_model_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
            proj_hidden_dim=proj_hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        self.recipe_encoder = RecipeEncoder(
            text_dim=text_dim,
            embed_dim=embed_dim,
            proj_hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Convenience freeze / unfreeze wrappers
    # ------------------------------------------------------------------
    def freeze_image_backbone(self):
        self.image_encoder.freeze_backbone()

    def unfreeze_image_backbone(self):
        self.image_encoder.unfreeze_backbone()

    # ------------------------------------------------------------------
    # Parameter groups (used by the training script)
    # ------------------------------------------------------------------
    def get_param_groups(self, lr_head: float, lr_backbone: float):
        """
        Return parameter groups for AdamW:
          - projection heads (both sides) → lr_head
          - CLIP visual backbone          → lr_backbone  (may be 0 if frozen)
        """
        head_params = list(self.image_encoder.proj.parameters()) + \
                      list(self.recipe_encoder.proj.parameters())
        backbone_params = list(self.image_encoder.backbone.parameters())
        return [
            {"params": head_params,     "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,   # (B, 3, H, W) — open_clip pre-processed
        recipes: torch.Tensor,  # (B, text_dim) — mean-pooled MiniLM embs
    ):
        """
        Returns:
            image_emb  : (B, embed_dim) L2-normalised
            recipe_emb : (B, embed_dim) L2-normalised
        """
        image_emb  = self.image_encoder(images)
        recipe_emb = self.recipe_encoder(recipes)
        return image_emb, recipe_emb
