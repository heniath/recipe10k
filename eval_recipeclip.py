#!/usr/bin/env python
"""
eval_recipeclip.py — Offline evaluation for RecipeCLIP-10K.

Loads a trained checkpoint, embeds the entire test (or val) split, and
computes retrieval metrics for both directions:
    Image → Recipe  (i2r)
    Recipe → Image  (r2i)

Metrics reported
================
    Recall@1, Recall@5, Recall@10
    Median Rank  (MedR)
    Mean Rank    (MeanR)

Results are also saved to a pickle file for downstream analysis.

Usage
=====
    # Evaluate on test split
    python eval_recipeclip.py \\
        --config configs/recipeclip.yaml \\
        --checkpoint snapshots/recipeclip/E2_frozen_backbone/best_e020_medR12.5.pth

    # Evaluate on val split
    python eval_recipeclip.py \\
        --config configs/recipeclip.yaml \\
        --checkpoint snapshots/recipeclip/E2_frozen_backbone/last.pth \\
        --partition val

    # Save embeddings only (no metric computation)
    python eval_recipeclip.py ... --save_embeds_only

Output
======
    results/recipeclip/<experiment>/test_metrics.txt
    results/recipeclip/<experiment>/img_embeds.pkl
    results/recipeclip/<experiment>/rec_embeds.pkl
    results/recipeclip/<experiment>/rec_ids.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.utils.data
import yaml
from tqdm import tqdm

from models.recipeclip import RecipeCLIP
from datasets.clip_loader import RecipeCLIPDataset

try:
    import open_clip
except ImportError as e:
    raise ImportError("Install open_clip_torch: pip install open_clip_torch") from e


# ============================================================
# CLI
# ============================================================

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RecipeCLIP-10K")
    p.add_argument("--config",     required=True,  help="Path to YAML config")
    p.add_argument("--checkpoint", required=True,  help="Path to .pth checkpoint")
    p.add_argument("--partition",  default="test", choices=["val", "test"],
                   help="Which split to evaluate on")
    p.add_argument("--gpu_ids",    default="",     help="GPU IDs, e.g. '0,1'")
    p.add_argument("--no_cuda",    action="store_true")
    p.add_argument("--results_dir", default="results/recipeclip/",
                   help="Directory to write metric text + embedding pickles")
    p.add_argument("--save_embeds_only", action="store_true",
                   help="Save embeddings without printing metrics table")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override eval batch size from config")
    p.add_argument("--medr_N",     type=int, default=None,
                   help="Override medr_N from config")
    p.add_argument("--n_trials",   type=int, default=None,
                   help="Override n_trials from config")
    return p


# ============================================================
# Ranking
# ============================================================

def compute_ranks(
    im_vecs: np.ndarray,
    rec_vecs: np.ndarray,
    N: int = 1000,
    n_trials: int = 10,
    direction: str = "i2r",
    seed: int = 42,
) -> dict:
    """
    Compute retrieval metrics over n_trials random N-subset draws.

    Returns dict with keys: MedR, MeanR, R@1, R@5, R@10
    """
    random.seed(seed)
    total = len(im_vecs)
    N = min(N, total)

    glob_ranks   = []
    glob_recall  = {1: 0.0, 5: 0.0, 10: 0.0}
    all_ranks    = []  # individual ranks across all trials (for MeanR)

    for _ in range(n_trials):
        ids = random.sample(range(total), N)
        im_sub  = im_vecs[ids]
        rec_sub = rec_vecs[ids]

        if direction == "i2r":
            sims = im_sub @ rec_sub.T   # (N, N)
        else:
            sims = rec_sub @ im_sub.T   # (N, N)

        trial_ranks = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        for ii in range(N):
            sim     = sims[ii]
            sorting = np.argsort(sim)[::-1].tolist()
            pos     = sorting.index(ii) + 1      # 1-indexed rank

            if pos <= 1:  recall[1]  += 1
            if pos <= 5:  recall[5]  += 1
            if pos <= 10: recall[10] += 1
            trial_ranks.append(pos)

        for k in recall:
            recall[k] /= N
            glob_recall[k] += recall[k]

        glob_ranks.append(np.median(trial_ranks))
        all_ranks.extend(trial_ranks)

    for k in glob_recall:
        glob_recall[k] /= n_trials

    return {
        "MedR":  float(np.median(glob_ranks)),
        "MeanR": float(np.mean(all_ranks)),
        "R@1":   glob_recall[1],
        "R@5":   glob_recall[5],
        "R@10":  glob_recall[10],
    }


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        im_vecs  : (total, D) image embeddings
        rec_vecs : (total, D) recipe embeddings
        names    : (total,)   recipe ID strings
    """
    model.eval()
    all_img  = []
    all_rec  = []
    all_ids  = []

    for imgs, recs, _, rec_ids in tqdm(loader, desc="  Extracting embeddings"):
        imgs = imgs.to(device, non_blocking=True)
        recs = recs.to(device, non_blocking=True)
        ie, re = model(imgs, recs)
        all_img.append(ie.cpu().numpy())
        all_rec.append(re.cpu().numpy())
        all_ids.extend(rec_ids)

    im_vecs  = np.concatenate(all_img,  axis=0)   # (N, D)
    rec_vecs = np.concatenate(all_rec,  axis=0)   # (N, D)
    names    = np.array(all_ids)

    # Sort by ID for deterministic trials
    sort_idx  = np.argsort(names)
    return im_vecs[sort_idx], rec_vecs[sort_idx], names[sort_idx]


# ============================================================
# Main
# ============================================================

def main():
    args = get_parser().parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ecfg = cfg.get("eval", {})
    medr_N   = args.medr_N   or ecfg.get("medr_N",   1000)
    n_trials = args.n_trials or ecfg.get("n_trials",  10)
    bs       = args.batch_size or ecfg.get("batch_size", 256)
    workers  = cfg["data"].get("workers", 4)

    # --- device ---
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on: {device}")

    # --- model ---
    mcfg = cfg["model"]
    model = RecipeCLIP(
        clip_model_name  = mcfg.get("clip_model_name",  "ViT-B-32"),
        pretrained       = mcfg.get("pretrained",       "laion2b_s34b_b79k"),
        embed_dim        = mcfg.get("embed_dim",        256),
        proj_hidden_dim  = mcfg.get("proj_hidden_dim",  512),
        dropout          = mcfg.get("dropout",          0.1),
        text_dim         = mcfg.get("text_dim",         384),
        freeze_backbone  = False,    # irrelevant at eval time
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle DataParallel checkpoints transparently
    state = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)

    # If multiple GPUs available, wrap for faster inference
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Inference using {num_gpus} GPUs")

    # --- data ---
    dcfg = cfg["data"]
    _, _, preprocess = open_clip.create_model_and_transforms(
        mcfg.get("clip_model_name", "ViT-B-32"),
        pretrained=mcfg.get("pretrained", "laion2b_s34b_b79k"),
    )

    dataset = RecipeCLIPDataset(
        img_path  = dcfg["img_path"],
        data_path = dcfg["data_path"],
        partition = args.partition,
        transform = preprocess,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = bs,
        shuffle     = False,
        num_workers = workers,
        pin_memory  = True,
    )
    print(f"Evaluating on '{args.partition}' split: {len(dataset):,} samples")

    # --- embed ---
    im_vecs, rec_vecs, names = extract_embeddings(model, loader, device)
    print(f"Embedding shapes: images={im_vecs.shape}, recipes={rec_vecs.shape}")

    # --- results dir ---
    exp_name = cfg.get("experiment", "recipeclip")
    results_dir = os.path.join(args.results_dir, exp_name, args.partition)
    os.makedirs(results_dir, exist_ok=True)

    # --- save embeddings ---
    with open(os.path.join(results_dir, "img_embeds.pkl"),  "wb") as f:
        pickle.dump(im_vecs, f)
    with open(os.path.join(results_dir, "rec_embeds.pkl"),  "wb") as f:
        pickle.dump(rec_vecs, f)
    with open(os.path.join(results_dir, "rec_ids.pkl"),     "wb") as f:
        pickle.dump(names, f)
    print(f"Embeddings saved to: {results_dir}")

    if args.save_embeds_only:
        return

    # --- compute metrics ---
    print(f"\nComputing metrics (N={medr_N}, trials={n_trials})...")

    m_i2r = compute_ranks(im_vecs, rec_vecs, medr_N, n_trials, direction="i2r")
    m_r2i = compute_ranks(im_vecs, rec_vecs, medr_N, n_trials, direction="r2i")

    # --- print table ---
    header = f"\n{'Metric':<12} {'Image→Recipe':>15} {'Recipe→Image':>15}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key in ["MedR", "MeanR", "R@1", "R@5", "R@10"]:
        print(f"{key:<12} {m_i2r[key]:>15.4f} {m_r2i[key]:>15.4f}")
    print(sep)

    # --- save text report ---
    report_path = os.path.join(results_dir, "metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Partition  : {args.partition}\n")
        f.write(f"N samples  : {len(dataset)}\n")
        f.write(f"medr_N     : {medr_N}\n")
        f.write(f"n_trials   : {n_trials}\n\n")
        f.write(f"{'Metric':<12} {'Image→Recipe':>15} {'Recipe→Image':>15}\n")
        f.write("-" * 44 + "\n")
        for key in ["MedR", "MeanR", "R@1", "R@5", "R@10"]:
            f.write(f"{key:<12} {m_i2r[key]:>15.4f} {m_r2i[key]:>15.4f}\n")

    print(f"\nMetric report saved to: {report_path}")

    # --- save pickle for ablation table aggregation ---
    with open(os.path.join(results_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({"i2r": m_i2r, "r2i": m_r2i}, f)


if __name__ == "__main__":
    main()
