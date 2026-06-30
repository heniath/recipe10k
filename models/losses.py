"""
models/losses.py — Symmetric InfoNCE contrastive loss for RecipeCLIP-10K.

Usage:
    loss_fn = SymmetricInfoNCE(temperature=0.07, learnable=True)
    loss = loss_fn(image_emb, recipe_emb)   # both L2-normalised, shape (B, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """
    Symmetric contrastive (InfoNCE / NT-Xent) loss.

    Given a batch of B (image, recipe) matched pairs, all other pairs
    in the batch act as negatives.  The loss is computed in both directions
    (image→recipe and recipe→image) and averaged.

    Args:
        temperature (float): Initial temperature value τ.
        learnable   (bool) : If True, log(1/τ) is a learnable parameter
                             (same as OpenAI CLIP).  Gradient is clipped to
                             keep τ ∈ [0.01, 1.0].
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            # Store log(1/τ) as a parameter — identical to CLIP's implementation.
            # Initialise so that 1/exp(logit_scale) == temperature.
            init_val = torch.log(torch.tensor(1.0 / temperature))
            self.logit_scale = nn.Parameter(init_val)
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / temperature)))
        self.learnable = learnable

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image_emb: torch.Tensor,   # (B, D), already L2-normalised
        recipe_emb: torch.Tensor,  # (B, D), already L2-normalised
    ) -> torch.Tensor:
        """
        Returns the mean of image→recipe and recipe→image cross-entropy losses.

        Both input tensors must already be L2-normalised so that their dot
        product equals cosine similarity.
        """
        B = image_emb.size(0)

        # Clamp temperature to a stable range (same as CLIP: max 100).
        scale = self.logit_scale.exp().clamp(max=100.0)

        # Similarity matrix: (B, B)
        logits = scale * image_emb @ recipe_emb.T   # (B, B)

        # Ground-truth labels: diagonal is the positive pair.
        labels = torch.arange(B, device=image_emb.device)

        # image → recipe (row-wise softmax)
        loss_i2r = F.cross_entropy(logits, labels)
        # recipe → image (column-wise softmax = transpose row-wise)
        loss_r2i = F.cross_entropy(logits.T, labels)

        return (loss_i2r + loss_r2i) / 2.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def temperature(self) -> float:
        """Current effective temperature τ (read-only convenience)."""
        return (1.0 / self.logit_scale.exp()).item()

    def extra_repr(self) -> str:
        return f"learnable={self.learnable}, temperature={self.temperature:.4f}"
