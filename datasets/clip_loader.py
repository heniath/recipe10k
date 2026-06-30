"""
datasets/clip_loader.py — LMDB-backed PyTorch Dataset for RecipeCLIP-10K.

Design decisions
================
* Reads the **same** LMDB and keys files as the original im2recipe baseline.
  Nothing is re-built; no raw-JSON parsing for E2/E3.
* Every sample is a **matched** (image, recipe) pair — no 80 % mismatch logic.
  In-batch negatives from the contrastive loss provide all the negative signal.
* Recipe embedding = mean of the per-sentence MiniLM vectors stored in the
  `intrs` field of each LMDB record (shape: N_sentences × 384).
* Image transform = open_clip's own preprocessing pipeline (passed in at
  construction).
* For val/test: always pick the first loadable image (no shuffle).
* For train: shuffle candidate images so different images are seen each epoch.

LMDB record structure (produced by mk_dataset_subset.py):
    {
        "ingrs"  : np.array (maxlen,)       — ingredient W2V indices (unused here)
        "intrs"  : np.array (N, 384)         — per-sentence MiniLM embeddings
        "classes": int                       — recipe category index
        "imgs"   : list[dict]                — image metadata: [{"id": "...", ...}, ...]
    }
"""

from __future__ import annotations

import os
import pickle
import sys
import numpy as np

import lmdb
import torch
import torch.utils.data as data
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_img_path(img_entry: dict, img_root: str) -> str:
    """Reconstruct filesystem path from LMDB image-metadata dict."""
    parts = [img_entry["id"][i] for i in range(4)]
    sub   = os.path.join(*parts)
    return os.path.join(img_root, sub, img_entry["id"])


def _load_pil(path: str) -> Image.Image:
    """Open an image file and return an RGB PIL Image; raise on failure."""
    img = Image.open(path)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RecipeCLIPDataset(data.Dataset):
    """
    PyTorch Dataset for RecipeCLIP contrastive training / evaluation.

    Args:
        img_path    : root directory of recipe images (e.g. "data/images/")
        data_path   : directory containing LMDB and key files (e.g. "data/")
        partition   : "train", "val", or "test"
        transform   : torchvision / open_clip image transform pipeline.
                      If None, raw PIL Images are returned (not useful for
                      training but handy for debugging).
    """

    def __init__(
        self,
        img_path: str,
        data_path: str,
        partition: str,
        transform=None,
    ):
        if partition not in ("train", "val", "test"):
            raise ValueError(f"partition must be 'train', 'val', or 'test', got '{partition}'")

        self.partition = partition
        self.img_path  = img_path
        self.transform = transform

        # Open LMDB (read-only, same flags as original data_loader.py)
        lmdb_path = os.path.join(data_path, f"{partition}_lmdb")
        self.env = lmdb.open(
            lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Load recipe IDs
        keys_path = os.path.join(data_path, f"{partition}_keys.pkl")
        with open(keys_path, "rb") as f:
            self.ids: list[str] = pickle.load(f)

        # Maximum number of candidate images to try per recipe
        self.max_imgs = 5

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        """
        Returns:
            image      : transformed image tensor  (C, H, W) or PIL Image
            recipe_emb : mean-pooled MiniLM tensor (text_dim,) — float32
            rec_class  : int class label (for optional category regularisation)
            rec_id     : str recipe ID  (needed for eval ranking)
        """
        # We may retry on a different index if the current one has no valid image.
        while True:
            rec_id = self.ids[index]

            # ----- load LMDB record ----------------------------------------
            with self.env.begin(write=False) as txn:
                raw = txn.get(rec_id.encode("latin1"))
            sample = pickle.loads(raw, encoding="latin1")

            # ----- recipe embedding (mean-pool per-sentence MiniLM) ---------
            intrs = sample["intrs"]          # np.array (N, text_dim)
            recipe_emb = torch.from_numpy(
                np.mean(intrs, axis=0).astype(np.float32)
            )                                # (text_dim,)

            # ----- image ----------------------------------------------------
            candidates = sample["imgs"][: self.max_imgs]
            if self.partition == "train":
                # Shuffle so we see different images across epochs.
                perm = np.random.permutation(len(candidates))
                candidates = [candidates[i] for i in perm]

            img = None
            for cand in candidates:
                cand_path = _build_img_path(cand, self.img_path)
                try:
                    img = _load_pil(cand_path)
                    break
                except Exception:
                    continue

            if img is None:
                if self.partition == "train":
                    # Silently resample during training.
                    index = np.random.randint(0, len(self.ids))
                    continue
                else:
                    # Val / test: use a white placeholder as last resort.
                    print(
                        f"[RecipeCLIPDataset] Warning: no valid image for {rec_id},"
                        " using placeholder.",
                        file=sys.stderr,
                    )
                    img = Image.new("RGB", (224, 224), "white")

            break   # successful load

        # ----- apply transform ----------------------------------------------
        if self.transform is not None:
            img = self.transform(img)

        rec_class = int(sample["classes"]) - 1  # 0-indexed

        return img, recipe_emb, rec_class, rec_id
