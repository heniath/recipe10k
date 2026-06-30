#!/usr/bin/env python
"""
train_recipeclip.py — Training script for RecipeCLIP-10K.

Usage
=====
    # E2: frozen backbone, projection heads only
    python train_recipeclip.py --config configs/recipeclip.yaml

    # E3: unfreeze backbone after warm-up (set in config or via CLI)
    python train_recipeclip.py --config configs/recipeclip.yaml \\
        --freeze_backbone false --lr_backbone 1e-5

    # Resume from checkpoint
    python train_recipeclip.py --config configs/recipeclip.yaml \\
        --resume snapshots/recipeclip/E2_best.pth

Features
========
* Reads all hyperparameters from a YAML config (configs/recipeclip.yaml)
* CLI flags override YAML values (handy for ablation sweeps)
* Multi-GPU via torch.nn.DataParallel
* AdamW + linear-warmup + cosine-decay learning-rate schedule
* Learnable InfoNCE temperature
* Checkpoint saved on every MedR improvement (best model only)
* Periodic val evaluation with Recall@1/5/10, MedR, MeanR — both directions
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from tqdm import tqdm

# Project imports
from models.recipeclip import RecipeCLIP
from models.losses import SymmetricInfoNCE
from datasets.clip_loader import RecipeCLIPDataset

try:
    import open_clip
except ImportError as e:
    raise ImportError("Install open_clip_torch: pip install open_clip_torch") from e


# ============================================================
# CLI parsing
# ============================================================

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RecipeCLIP-10K")
    p.add_argument("--config", default="configs/recipeclip.yaml",
                   help="Path to YAML config file")

    # --- overrides (optional) ---
    p.add_argument("--experiment",      type=str,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr_head",         type=float, default=None)
    p.add_argument("--lr_backbone",     type=float, default=None)
    p.add_argument("--weight_decay",    type=float, default=None)
    p.add_argument("--embed_dim",       type=int,   default=None)
    p.add_argument("--freeze_backbone", type=str,   default=None,
                   help="true / false")
    p.add_argument("--val_freq",        type=int,   default=None)
    p.add_argument("--resume",          type=str,   default="",
                   help="Path to checkpoint (.pth) to resume from")
    p.add_argument("--gpu_ids",         type=str,   default="",
                   help="Comma-separated GPU IDs, e.g. '0,1'. Empty = all visible.")
    p.add_argument("--workers",         type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--no_cuda",         action="store_true")
    return p


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply non-None CLI arguments on top of the YAML config."""
    def _set(section, key, val):
        if val is not None:
            cfg.setdefault(section, {})[key] = val

    _set("training", "epochs",       args.epochs)
    _set("training", "batch_size",   args.batch_size)
    _set("training", "lr_head",      args.lr_head)
    _set("training", "lr_backbone",  args.lr_backbone)
    _set("training", "weight_decay", args.weight_decay)
    _set("training", "val_freq",     args.val_freq)
    _set("training", "workers_override", args.workers)
    _set("training", "seed_override",    args.seed)
    _set("model",    "embed_dim",    args.embed_dim)

    if args.freeze_backbone is not None:
        cfg.setdefault("model", {})["freeze_backbone"] = \
            args.freeze_backbone.lower() == "true"
    if args.experiment is not None:
        cfg["experiment"] = args.experiment

    return cfg


# ============================================================
# Evaluation helpers
# ============================================================

def rank(im_vecs: np.ndarray, rec_vecs: np.ndarray, names: np.ndarray,
         N: int = 1000, n_trials: int = 10, direction: str = "i2r",
         seed: int = 42) -> tuple[float, float, dict]:
    """
    Compute MedR, MeanR, and Recall@{1,5,10} over `n_trials` random draws.

    Args:
        im_vecs    : (total, D) image embeddings
        rec_vecs   : (total, D) recipe embeddings
        names      : (total,)  recipe ID strings (for matching)
        N          : subset size per trial (medr_N in config)
        n_trials   : number of random draws
        direction  : "i2r" (image→recipe) or "r2i" (recipe→image)
        seed       : random seed for reproducibility

    Returns:
        med_r, mean_r, recall_dict  {1: float, 5: float, 10: float}
    """
    random.seed(seed)
    total = len(names)
    N = min(N, total)

    glob_rank   = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}

    for _ in range(n_trials):
        ids = random.sample(range(total), N)
        im_sub  = im_vecs[ids]
        rec_sub = rec_vecs[ids]

        if direction == "i2r":
            sims = im_sub @ rec_sub.T      # (N, N)
        else:
            sims = rec_sub @ im_sub.T      # (N, N)

        med_rank = []
        recall   = {1: 0.0, 5: 0.0, 10: 0.0}

        for ii in range(N):
            sim     = sims[ii]
            sorting = np.argsort(sim)[::-1].tolist()
            pos     = sorting.index(ii)    # rank of the positive (0-indexed)

            if pos + 1 <= 1:  recall[1]  += 1
            if pos + 1 <= 5:  recall[5]  += 1
            if pos + 1 <= 10: recall[10] += 1
            med_rank.append(pos + 1)

        for k in recall:
            recall[k] /= N

        for k in glob_recall:
            glob_recall[k] += recall[k]
        glob_rank.append(np.median(med_rank))

    for k in glob_recall:
        glob_recall[k] /= n_trials

    return float(np.median(glob_rank)), float(np.mean(glob_rank)), glob_recall


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict,
) -> tuple[float, dict, dict]:
    """
    Embed all val samples, then compute retrieval metrics.

    Returns:
        med_r_i2r  : median rank for image→recipe direction
        metrics_i2r: dict with MedR, MeanR, R@1, R@5, R@10
        metrics_r2i: same for recipe→image direction
    """
    model.eval()
    ecfg = cfg["eval"]
    N      = ecfg.get("medr_N", 1000)
    trials = ecfg.get("n_trials", 10)

    all_img_emb  = []
    all_rec_emb  = []
    all_rec_ids  = []

    with torch.no_grad():
        for imgs, recs, _, rec_ids in tqdm(val_loader, desc="  Embedding", leave=False):
            imgs = imgs.to(device)
            recs = recs.to(device)
            ie, re = model(imgs, recs)
            all_img_emb.append(ie.cpu().numpy())
            all_rec_emb.append(re.cpu().numpy())
            all_rec_ids.extend(rec_ids)

    im_vecs  = np.concatenate(all_img_emb, axis=0)
    rec_vecs = np.concatenate(all_rec_emb, axis=0)
    names    = np.array(all_rec_ids)

    # Sort by ID so trials are deterministic regardless of loader order.
    sort_idx  = np.argsort(names)
    im_vecs   = im_vecs[sort_idx]
    rec_vecs  = rec_vecs[sort_idx]
    names     = names[sort_idx]

    med_i2r, mean_i2r, rec_i2r = rank(im_vecs, rec_vecs, names, N, trials, "i2r")
    med_r2i, mean_r2i, rec_r2i = rank(im_vecs, rec_vecs, names, N, trials, "r2i")

    metrics_i2r = {"MedR": med_i2r, "MeanR": mean_i2r, **{f"R@{k}": v for k, v in rec_i2r.items()}}
    metrics_r2i = {"MedR": med_r2i, "MeanR": mean_r2i, **{f"R@{k}": v for k, v in rec_r2i.items()}}

    return med_i2r, metrics_i2r, metrics_r2i


# ============================================================
# Learning-rate schedule
# ============================================================

def get_lr_schedule(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Returns a LambdaLR with linear warmup then cosine decay.
    One scheduler governs all param groups proportionally.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


# ============================================================
# Main
# ============================================================

def main():
    args   = get_parser().parse_args()
    cfg    = load_config(args.config)
    cfg    = apply_overrides(cfg, args)

    # --- seeds ---
    seed = cfg["training"].get("seed_override") or cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- GPU setup ---
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device   = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
    else:
        device   = torch.device("cpu")
        num_gpus = 0

    print(f"Device: {device}  |  GPUs: {num_gpus}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    mcfg = cfg["model"]
    model = RecipeCLIP(
        clip_model_name  = mcfg.get("clip_model_name",  "ViT-B-32"),
        pretrained       = mcfg.get("pretrained",       "laion2b_s34b_b79k"),
        embed_dim        = mcfg.get("embed_dim",        256),
        proj_hidden_dim  = mcfg.get("proj_hidden_dim",  512),
        dropout          = mcfg.get("dropout",          0.1),
        text_dim         = mcfg.get("text_dim",         384),
        freeze_backbone  = mcfg.get("freeze_backbone",  True),
    )
    model.to(device)

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # Unwrap for method access
    inner = model.module if isinstance(model, nn.DataParallel) else model

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    lcfg     = cfg["loss"]
    loss_fn  = SymmetricInfoNCE(
        temperature = lcfg.get("temperature",    0.07),
        learnable   = lcfg.get("learnable_temp", True),
    ).to(device)

    # --------------------------------------------------------
    # Optimiser
    # --------------------------------------------------------
    tcfg = cfg["training"]
    lr_head     = tcfg.get("lr_head",     3e-4)
    lr_backbone = tcfg.get("lr_backbone", 0.0)
    wd          = tcfg.get("weight_decay", 0.01)

    param_groups = inner.get_param_groups(lr_head, lr_backbone)
    # Add loss temperature to the head param group
    param_groups[0]["params"] += list(loss_fn.parameters())

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    total_epochs  = tcfg.get("epochs", 30)
    warmup_epochs = tcfg.get("warmup_epochs", 2)
    scheduler     = get_lr_schedule(optimizer, warmup_epochs, total_epochs)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    dcfg = cfg["data"]
    workers = tcfg.get("workers_override") or dcfg.get("workers", 4)

    # open_clip image pre-processing transform
    _, _, preprocess = open_clip.create_model_and_transforms(
        mcfg.get("clip_model_name", "ViT-B-32"),
        pretrained=mcfg.get("pretrained", "laion2b_s34b_b79k"),
    )

    batch_size = tcfg.get("batch_size", 128)

    train_dataset = RecipeCLIPDataset(
        img_path  = dcfg["img_path"],
        data_path = dcfg["data_path"],
        partition = "train",
        transform = preprocess,
    )
    val_dataset = RecipeCLIPDataset(
        img_path  = dcfg["img_path"],
        data_path = dcfg["data_path"],
        partition = "val",
        transform = preprocess,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = workers,
        pin_memory  = True,
        drop_last   = True,   # keep batch size constant for contrastive loss
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = tcfg.get("batch_size", 256),
        shuffle     = False,
        num_workers = workers,
        pin_memory  = True,
    )

    print(f"Train: {len(train_dataset):,} recipes | Val: {len(val_dataset):,} recipes")
    print(f"Batch size: {batch_size} | Effective (×GPUs): {batch_size * max(num_gpus, 1)}")

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
    start_epoch = 0
    best_med_r  = float("inf")
    valtrack    = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_med_r  = ckpt.get("best_med_r", float("inf"))
        valtrack    = ckpt.get("valtrack", 0)
        print(f"  Resumed at epoch {start_epoch}, best MedR {best_med_r:.2f}")

    # --------------------------------------------------------
    # Snapshot directory
    # --------------------------------------------------------
    exp_name  = cfg.get("experiment", "recipeclip")
    snap_dir  = tcfg.get("snapshots", "snapshots/recipeclip/")
    snap_dir  = os.path.join(snap_dir, exp_name)
    os.makedirs(snap_dir, exist_ok=True)

    val_freq  = tcfg.get("val_freq", 2)
    patience  = tcfg.get("patience", 5)
    grad_clip = tcfg.get("grad_clip", 1.0)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, total_epochs):
        model.train()
        inner.image_encoder.backbone.eval()  # keep BN in eval if backbone is frozen

        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]",
                    leave=True, dynamic_ncols=True)

        for imgs, recs, _, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            recs = recs.to(device, non_blocking=True)

            img_emb, rec_emb = model(imgs, recs)
            loss = loss_fn(img_emb, rec_emb)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    max_norm=grad_clip,
                )
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{np.mean(losses):.4f}",
                "τ":    f"{loss_fn.temperature:.4f}",
            })

        scheduler.step()

        avg_loss = float(np.mean(losses))
        lr0 = optimizer.param_groups[0]["lr"]
        lr1 = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:3d}  loss={avg_loss:.4f}  "
              f"lr_head={lr0:.2e}  lr_bb={lr1:.2e}  τ={loss_fn.temperature:.4f}")

        # ---- validation ------------------------------------------------
        if (epoch + 1) % val_freq == 0:
            med_r, m_i2r, m_r2i = validate(model, val_loader, device, cfg)
            model.train()

            print(
                f"  [Val i2r] MedR={m_i2r['MedR']:.1f}  MeanR={m_i2r['MeanR']:.1f}"
                f"  R@1={m_i2r['R@1']:.3f}  R@5={m_i2r['R@5']:.3f}  R@10={m_i2r['R@10']:.3f}"
            )
            print(
                f"  [Val r2i] MedR={m_r2i['MedR']:.1f}  MeanR={m_r2i['MeanR']:.1f}"
                f"  R@1={m_r2i['R@1']:.3f}  R@5={m_r2i['R@5']:.3f}  R@10={m_r2i['R@10']:.3f}"
            )

            # ---- patience / best-model tracking ------------------------
            is_best = med_r < best_med_r
            if is_best:
                best_med_r = med_r
                valtrack   = 0
            else:
                valtrack += 1

            # ---- checkpoint --------------------------------------------
            state = {
                "epoch":      epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_med_r": best_med_r,
                "valtrack":   valtrack,
                "metrics_i2r": m_i2r,
                "metrics_r2i": m_r2i,
            }
            save_checkpoint(
                state,
                os.path.join(snap_dir, "last.pth"),
            )
            if is_best:
                save_checkpoint(
                    state,
                    os.path.join(snap_dir, f"best_e{epoch+1:03d}_medR{best_med_r:.1f}.pth"),
                )

            # ---- early stopping ----------------------------------------
            if valtrack >= patience:
                print(f"Early stopping triggered (no improvement for {patience} val checks).")
                break

    print(f"\nTraining complete. Best val MedR (i2r): {best_med_r:.2f}")
    print(f"Checkpoints saved in: {snap_dir}")


if __name__ == "__main__":
    main()
