#!/usr/bin/env python
"""
verify_coverage.py — Post-build image-coverage audit for Recipe1M+ LMDBs.

For every recipe stored in train/val/test LMDBs, checks that at least one
image dict in the stored sample resolves to a file that physically exists
under --img_path.

Exit codes:
  0 — all splits have 100% image coverage
  1 — one or more samples have zero valid images (rebuild needed)

Usage:
    python scripts/verify_coverage.py \
        --img_path /path/to/recipe1M+_images \
        --data_path data/

Expected output when OK:
    train :  7260 /  7260 (100.00%) ✓
    val   :   897 /   897 (100.00%) ✓
    test  :   912 /   912 (100.00%) ✓
    Overall image coverage: 100.00% ✓
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import lmdb


# ---------------------------------------------------------------------------
# Helpers (mirror mk_dataset_subset.py helpers for consistency)
# ---------------------------------------------------------------------------

def image_path_from_id(img_id: str, image_root) -> Path:
    """Build the 4-level filesystem path for a Recipe1M+ image."""
    return Path(image_root) / img_id[0] / img_id[1] / img_id[2] / img_id[3] / img_id


def count_valid_images(imgs: list, image_root) -> int:
    """Return the number of image dicts in `imgs` whose files physically exist."""
    count = 0
    for img in imgs:
        img_id = img.get("id")
        if img_id and image_path_from_id(img_id, image_root).exists():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Verify image coverage of built LMDBs")
    p.add_argument("--img_path", required=True,
                   help="Root directory of Recipe1M+ images (4-level folder format)")
    p.add_argument("--data_path", default="../data",
                   help="Directory containing train/val/test _lmdb and _keys.pkl files")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Splits to check (default: train val test)")
    return p.parse_args()


def check_split(split: str, data_path: str, img_path: str) -> tuple[int, int]:
    """Return (total_samples, samples_with_zero_valid_images) for a split."""
    lmdb_path = os.path.join(data_path, f"{split}_lmdb")
    keys_path = os.path.join(data_path, f"{split}_keys.pkl")

    if not os.path.isdir(lmdb_path):
        print(f"  [{split}] LMDB not found at {lmdb_path} — skipping.")
        return 0, 0
    if not os.path.isfile(keys_path):
        print(f"  [{split}] Keys file not found at {keys_path} — skipping.")
        return 0, 0

    with open(keys_path, "rb") as f:
        keys = pickle.load(f)

    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                    readahead=False, meminit=False)

    total   = 0
    missing = 0  # samples with 0 valid images
    broken_ids = []

    with env.begin(write=False) as txn:
        for rid in keys:
            raw = txn.get(rid.encode("latin1"))
            if raw is None:
                print(f"  [{split}] WARNING: key '{rid}' not found in LMDB!")
                missing += 1
                total   += 1
                broken_ids.append(rid)
                continue

            sample = pickle.loads(raw, encoding="latin1")
            imgs   = sample.get("imgs", [])
            n_valid = count_valid_images(imgs, img_path)

            total += 1
            if n_valid == 0:
                missing += 1
                broken_ids.append(rid)

    env.close()

    pct  = 100.0 * (total - missing) / total if total > 0 else 0.0
    ok   = "✓" if missing == 0 else "✗  ← REBUILD NEEDED"
    print(f"  {split:<6}: {total - missing:>6} / {total:>6} ({pct:.2f}%) {ok}")

    if broken_ids:
        preview = broken_ids[:5]
        print(f"         Samples with no valid image (first {len(preview)}): {preview}")

    return total, missing


def main():
    args = get_args()

    print("=" * 60)
    print("Image Coverage Verification")
    print("=" * 60)
    print(f"  img_path  : {args.img_path}")
    print(f"  data_path : {args.data_path}")
    print()

    grand_total   = 0
    grand_missing = 0

    for split in args.splits:
        total, missing = check_split(split, args.data_path, args.img_path)
        grand_total   += total
        grand_missing += missing

    print()
    overall_pct = 100.0 * (grand_total - grand_missing) / grand_total if grand_total > 0 else 0.0
    if grand_missing == 0:
        print(f"Overall image coverage: {overall_pct:.2f}% ✓")
        print("All LMDB samples have at least one physically-existing image.")
        sys.exit(0)
    else:
        print(f"Overall image coverage: {overall_pct:.2f}%  ✗")
        print(f"  {grand_missing} sample(s) have ZERO valid images.")
        print("  Re-run mk_dataset_subset.py with --img_path to rebuild the dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
