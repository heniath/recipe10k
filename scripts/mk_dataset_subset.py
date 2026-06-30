#!/usr/bin/env python
"""
mk_dataset_subset.py - Build LMDB datasets for im2recipe training from Recipe1M+ subset data.

Replaces the original mk_dataset.py pipeline by:
  1. Using sentence-transformers instead of skip-thought vectors (no .t7 files needed)
  2. Using gensim for Word2Vec training (no C compilation needed)
  3. Reading subset JSON files directly (no renaming required)
  4. Filtering recipes to only those with physically-existing images on disk.

Usage (from the scripts/ directory):
    python mk_dataset_subset.py --img_path /path/to/recipe1M+_images

Or with custom paths:
    python mk_dataset_subset.py \
        --img_path /path/to/recipe1M+_images \
        --layer1 ../data/layer1_subset.json \
        --layer2 ../data/layer2_subset.json \
        --det_ingrs ../data/det_ingrs.json \
        --classes ../data/classes1M.pkl \
        --output_dir ../data \
        --st_model all-MiniLM-L6-v2
"""

import argparse
import glob
import os
import pickle
import random
import shutil
import sys
import time

import lmdb
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Build LMDB dataset for im2recipe (subset)")

    # Input files
    p.add_argument("--layer1", default="../data/layer1_subset (1).json",
                   help="Path to layer1 JSON (recipes with instructions/ingredients)")
    p.add_argument("--layer2", default="../data/layer2_subset (1).json",
                   help="Path to layer2 JSON (recipe-to-image mapping)")
    p.add_argument("--det_ingrs", default="../data/det_ingrs.json",
                   help="Path to detected ingredients JSON")
    p.add_argument("--classes", default="../data/classes1M.pkl",
                   help="Path to classes pickle (class_dict + id2class)")
    p.add_argument("--remove_ids", default="remove1M.txt",
                   help="Path to file with recipe IDs to exclude")

    # Image root — required for existence filtering
    p.add_argument("--img_path", required=True,
                   help="Root directory of Recipe1M+ images in four-level folder format "
                        "(e.g. /data/images/  ->  /data/images/3/e/2/3/3e23a9b850.jpg).")

    # Output
    p.add_argument("--output_dir", default="../data",
                   help="Directory where LMDBs, keys, and vocab files are written")

    # Sentence-transformer model for instruction encoding
    p.add_argument("--st_model", default="all-MiniLM-L6-v2",
                   help="Sentence-transformer model name (determines embedding dim)")

    # Word2Vec
    p.add_argument("--w2v_dim", default=300, type=int,
                   help="Word2Vec embedding dimension")
    p.add_argument("--w2v_min_count", default=10, type=int,
                   help="Word2Vec minimum word count")
    p.add_argument("--w2v_window", default=10, type=int,
                   help="Word2Vec context window size")
    p.add_argument("--w2v_epochs", default=10, type=int,
                   help="Word2Vec training epochs")

    # Dataset params
    p.add_argument("--maxlen", default=20, type=int,
                   help="Max number of instructions / ingredients per recipe")
    p.add_argument("--max_imgs", default=5, type=int,
                   help="Max number of images per recipe stored in LMDB")
    p.add_argument("--batch_size_encode", default=256, type=int,
                   help="Batch size for sentence-transformer encoding")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Image-path helpers
# ---------------------------------------------------------------------------

from pathlib import Path

def image_path_from_id(img_id: str, image_root) -> Path:
    """Build the 4-level filesystem path for a Recipe1M+ image.

    Example:
        img_id = '3e23a9b850.jpg'
        -> <image_root>/3/e/2/3/3e23a9b850.jpg
    """
    stem = img_id  # keep the extension; index into the stem characters
    return Path(image_root) / stem[0] / stem[1] / stem[2] / stem[3] / img_id


def get_valid_images(recipe: dict, image_root) -> list:
    """Return only the image dicts whose files physically exist on disk.

    The returned list has the same dict schema as the input (id, url, …),
    so the dataloader serialization format is unchanged.
    """
    valid = []
    for img in recipe.get("images", []):
        img_id = img.get("id")
        if not img_id:
            continue
        if image_path_from_id(img_id, image_root).exists():
            valid.append(img)
    return valid


# ---------------------------------------------------------------------------
# Ingredient detection (same logic as original proc.py, but inlined)
# ---------------------------------------------------------------------------

def detect_ingrs(recipe, vocab):
    """Detect ingredients in a recipe using the vocabulary."""
    try:
        ingr_names = [ingr["text"] for ingr in recipe["ingredients"] if ingr["text"]]
    except Exception:
        ingr_names = []

    detected = set()
    for name in ingr_names:
        name = name.replace(" ", "_")
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)

    return list(detected) + [vocab["</i>"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ------------------------------------------------------------------
    # 1. Load data files
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading data files")
    print("=" * 60)

    import simplejson as json

    t0 = time.time()

    print(f"  Loading layer1 from: {args.layer1}")
    with open(args.layer1) as f:
        layer1 = json.load(f)
    print(f"    -> {len(layer1)} recipes")

    print(f"  Loading layer2 from: {args.layer2}")
    with open(args.layer2) as f:
        layer2 = json.load(f)
    print(f"    -> {len(layer2)} image entries")

    print(f"  Loading det_ingrs from: {args.det_ingrs}")
    with open(args.det_ingrs) as f:
        det_ingrs = json.load(f)
    print(f"    -> {len(det_ingrs)} ingredient entries")

    print(f"  Loading classes from: {args.classes}")
    with open(args.classes, "rb") as f:
        class_dict = pickle.load(f)
        id2class = pickle.load(f)
    print(f"    -> {len(class_dict)} class assignments, {len(id2class)} class labels")

    # Build lookup dicts
    l2_by_id = {entry["id"]: entry for entry in layer2}
    det_by_id = {entry["id"]: entry for entry in det_ingrs}

    # Merge into a single dataset (same as utils.Layer.merge)
    dataset = []
    for entry in layer1:
        rid = entry["id"]
        merged = dict(entry)
        if rid in l2_by_id:
            merged.update(l2_by_id[rid])
        if rid in det_by_id:
            merged.update(det_by_id[rid])
        dataset.append(merged)

    print(f"  Merged dataset: {len(dataset)} recipes")

    # Load remove IDs
    remove_ids = {}
    if os.path.exists(args.remove_ids):
        with open(args.remove_ids, "r") as f:
            remove_ids = {w.rstrip(): i for i, w in enumerate(f)}
        print(f"  Loaded {len(remove_ids)} IDs to remove")

    print(f"  Done loading in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # 2. Train Word2Vec on ingredient text & build vocab
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Training Word2Vec and building ingredient vocabulary")
    print("=" * 60)

    from gensim.models import Word2Vec

    # Collect tokenized instruction text for w2v training (same as tokenize_instructions.py)
    print("  Tokenizing instructions for Word2Vec training...")
    w2v_sentences = []
    tok_chars = [",", ".", ";", "(", ")", "?", "!", "&", "%", ":", "*", '"']

    all_ingr_names = set()
    for entry in tqdm(dataset, desc="  Tokenizing"):
        if entry["partition"] != "train":
            continue
        # Build instruction text
        instrs_text = ""
        for instr in entry.get("instructions", []):
            instrs_text += instr["text"] + "\t"

        # Get detected ingredients and replace with underscored versions
        det_entry = det_by_id.get(entry["id"], {})
        det_ingr_list = det_entry.get("ingredients", [])
        valid_list = det_entry.get("valid", [])
        for j, det_ingr in enumerate(det_ingr_list):
            if j < len(valid_list) and not valid_list[j]:
                continue
            det_text = det_ingr["text"]
            underscore_text = det_text.replace(" ", "_")
            all_ingr_names.add(underscore_text)
            instrs_text = instrs_text.replace(det_text, underscore_text)

        # Tokenize
        for tc in tok_chars:
            instrs_text = instrs_text.replace(tc, " " + tc + " ")

        # Split into sentences (by tab = instruction boundaries)
        for sent in instrs_text.split("\t"):
            words = sent.strip().split()
            if words:
                w2v_sentences.append(words)

    print(f"  Collected {len(w2v_sentences)} sentences for Word2Vec training")
    print(f"  Found {len(all_ingr_names)} unique ingredient names")

    # Train Word2Vec
    print(f"  Training Word2Vec (dim={args.w2v_dim}, window={args.w2v_window}, "
          f"min_count={args.w2v_min_count}, epochs={args.w2v_epochs})...")
    w2v_model = Word2Vec(
        sentences=w2v_sentences,
        vector_size=args.w2v_dim,
        window=args.w2v_window,
        min_count=args.w2v_min_count,
        sg=1,  # skip-gram (same as original -cbow 0)
        hs=1,  # hierarchical softmax (same as original -hs 1)
        negative=0,  # no negative sampling (same as original -negative 0)
        workers=4,
        epochs=args.w2v_epochs,
    )

    # Save in formats compatible with the training code
    os.makedirs(os.path.join(args.output_dir, "text"), exist_ok=True)

    # Save as word2vec binary format (compatible with torchwordemb.load_word2vec_bin)
    vocab_bin_path = os.path.join(args.output_dir, "text", "vocab.bin")
    w2v_model.wv.save_word2vec_format(vocab_bin_path, binary=True)
    print(f"  Saved vocab.bin to: {vocab_bin_path}")

    # Save vocab.txt (list of words, one per line)
    vocab_txt_path = os.path.join(args.output_dir, "text", "vocab.txt")
    with open(vocab_txt_path, "w") as f:
        f.write("\n".join(w2v_model.wv.index_to_key))
    print(f"  Saved vocab.txt to: {vocab_txt_path} ({len(w2v_model.wv)} words)")

    # Build ingredient vocab dict (same as original mk_dataset.py)
    with open(vocab_txt_path) as f_vocab:
        ingr_vocab = {w.rstrip(): i + 2 for i, w in enumerate(f_vocab)}
        ingr_vocab["</i>"] = 1
    print(f"  Ingredient vocab size: {len(ingr_vocab)}")

    print()

    # ------------------------------------------------------------------
    # 3. Encode instructions with sentence-transformers
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Encoding instructions with sentence-transformers")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {args.st_model}")
    st_model = SentenceTransformer(args.st_model)
    st_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {st_dim}")

    # Collect all instruction sentences per recipe
    # We encode each instruction sentence separately (matching skip-thoughts behavior)
    recipe_instructions = {}  # id -> list of instruction strings
    for entry in dataset:
        rid = entry["id"]
        instrs = [instr["text"] for instr in entry.get("instructions", []) if instr.get("text")]
        recipe_instructions[rid] = instrs

    # Flatten all sentences for batch encoding
    all_sentences = []
    sentence_map = []  # (recipe_id, instr_index) for each sentence
    for rid, instrs in recipe_instructions.items():
        for idx, sent in enumerate(instrs):
            all_sentences.append(sent)
            sentence_map.append((rid, idx))

    print(f"  Total instruction sentences to encode: {len(all_sentences)}")
    print(f"  Encoding in batches of {args.batch_size_encode}...")

    all_embeddings = st_model.encode(
        all_sentences,
        batch_size=args.batch_size_encode,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Encoded shape: {all_embeddings.shape}")

    # Reassemble per-recipe: dict of id -> np.array of shape (n_instrs, st_dim)
    recipe_encs = {}
    for i, (rid, idx) in enumerate(sentence_map):
        if rid not in recipe_encs:
            recipe_encs[rid] = []
        recipe_encs[rid].append(all_embeddings[i])

    for rid in recipe_encs:
        recipe_encs[rid] = np.array(recipe_encs[rid], dtype=np.float32)

    print(f"  Encoded instructions for {len(recipe_encs)} recipes")
    print(f"  Instruction embedding dim: {st_dim}")
    print()

    # ------------------------------------------------------------------
    # 4. Build LMDBs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Building LMDB datasets")
    print("=" * 60)

    # Clean up existing LMDBs
    for part in ["train", "val", "test"]:
        lmdb_path = os.path.join(args.output_dir, f"{part}_lmdb")
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)

    envs = {
        "train": lmdb.open(os.path.join(args.output_dir, "train_lmdb"), map_size=int(1e11)),
        "val": lmdb.open(os.path.join(args.output_dir, "val_lmdb"), map_size=int(1e11)),
        "test": lmdb.open(os.path.join(args.output_dir, "test_lmdb"), map_size=int(1e11)),
    }

    keys = {"train": [], "val": [], "test": []}
    num_skipped = 0  # legacy total (kept for compat)

    # Per-reason skip counters
    n_skip_remove      = 0  # in remove_ids list
    n_skip_no_image    = 0  # no physically existing image on disk
    n_skip_text        = 0  # too long / no ingrs / no class / no encoding
    n_skip_bad_part    = 0  # missing or unknown partition label
    n_total            = 0  # total recipes examined
    n_with_valid_image = 0  # recipes that have at least one valid image

    for entry in tqdm(dataset, desc="  Building LMDBs"):
        rid = entry["id"]
        n_total += 1

        # Skip if in remove list
        if rid in remove_ids:
            n_skip_remove += 1
            num_skipped += 1
            continue

        # Validate partition label
        partition = entry.get("partition")
        if partition not in ("train", "val", "test"):
            n_skip_bad_part += 1
            num_skipped += 1
            continue

        # Get instruction count
        ninstrs = len(entry.get("instructions", []))

        # Detect ingredients
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)

        # Filter text/class/encoding
        if ninstrs >= args.maxlen or ningrs >= args.maxlen or ningrs == 0:
            n_skip_text += 1
            num_skipped += 1
            continue

        if rid not in class_dict:
            n_skip_text += 1
            num_skipped += 1
            continue

        if rid not in recipe_encs:
            n_skip_text += 1
            num_skipped += 1
            continue

        # --- Image existence filtering ---
        all_imgs    = entry.get("images") or []
        valid_imgs  = get_valid_images(entry, args.img_path)
        if len(valid_imgs) == 0:
            n_skip_no_image += 1
            num_skipped += 1
            continue

        n_with_valid_image += 1

        # Build ingredient vector
        ingr_vec = np.zeros(args.maxlen, dtype="uint16")
        ingr_vec[:ningrs] = ingr_detections

        # Get instruction embeddings for this recipe
        intrs = recipe_encs[rid]

        # Store only the filtered (existing) image dicts — same list-of-dicts schema.
        # Cap to max_imgs after filtering.
        stored_imgs = valid_imgs[:args.max_imgs]

        # Serialize sample (same format as original, with optional coverage metadata)
        serialized_sample = pickle.dumps({
            "ingrs": ingr_vec,
            "intrs": intrs,  # shape: (n_instrs, st_dim)
            "classes": class_dict[rid] + 1,
            "imgs": stored_imgs,          # only physically-existing image dicts
            "original_img_count": len(all_imgs),   # informational
            "valid_img_count": len(valid_imgs),    # informational
        })

        with envs[partition].begin(write=True) as txn:
            txn.put(rid.encode("latin1"), serialized_sample)

        keys[partition].append(rid)

    # Close LMDB environments
    for env in envs.values():
        env.close()

    # Save keys
    for part in keys:
        keys_path = os.path.join(args.output_dir, f"{part}_keys.pkl")
        with open(keys_path, "wb") as f:
            pickle.dump(keys[part], f)

    total_written = len(keys["train"]) + len(keys["val"]) + len(keys["test"])
    img_coverage_pct = 100.0 * n_with_valid_image / n_total if n_total > 0 else 0.0

    print(f"\n  ===== Preprocessing Summary =====")
    print(f"  Total recipes examined           : {n_total}")
    print(f"  Recipes with valid image(s)      : {n_with_valid_image} / {n_total} "
          f"({img_coverage_pct:.2f}%)")
    print(f"  Skipped – in remove_ids list     : {n_skip_remove}")
    print(f"  Skipped – no valid image on disk : {n_skip_no_image}")
    print(f"  Skipped – text/class/encoding    : {n_skip_text}")
    print(f"  Skipped – bad/missing partition  : {n_skip_bad_part}")
    print(f"  ─────────────────────────────────")
    print(f"  Written to LMDB:")
    print(f"    train : {len(keys['train'])}")
    print(f"    val   : {len(keys['val'])}")
    print(f"    test  : {len(keys['test'])}")
    print(f"    total : {total_written}")
    print(f"  Final image coverage             : 100.00% ✓")
    print(f"  (All written samples have at least one physically-existing image.)")
    print()

    # ------------------------------------------------------------------
    # 5. Save config for training
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 5: Summary")
    print("=" * 60)

    config = {
        "st_model": args.st_model,
        "st_dim": st_dim,
        "w2v_dim": args.w2v_dim,
        "num_train": len(keys["train"]),
        "num_val": len(keys["val"]),
        "num_test": len(keys["test"]),
        "num_classes": len(id2class),
        "vocab_size": len(w2v_model.wv),
    }
    config_path = os.path.join(args.output_dir, "dataset_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print(f"  Sentence-transformer model: {args.st_model}")
    print(f"  Instruction embedding dim:  {st_dim}")
    print(f"  Word2Vec dim:               {args.w2v_dim}")
    print(f"  Vocab size:                 {len(w2v_model.wv)}")
    print(f"  Number of classes:          {len(id2class)}")
    print()
    print("  Generated files:")
    print(f"    {os.path.join(args.output_dir, 'train_lmdb/')}")
    print(f"    {os.path.join(args.output_dir, 'val_lmdb/')}")
    print(f"    {os.path.join(args.output_dir, 'test_lmdb/')}")
    print(f"    {os.path.join(args.output_dir, 'train_keys.pkl')}")
    print(f"    {os.path.join(args.output_dir, 'val_keys.pkl')}")
    print(f"    {os.path.join(args.output_dir, 'test_keys.pkl')}")
    print(f"    {vocab_bin_path}")
    print(f"    {vocab_txt_path}")
    print(f"    {config_path}")
    print()
    print("  To train, run:")
    print(f"    python train.py --img_path data/images/ --data_path data/ "
          f"--ingrW2V data/text/vocab.bin --stDim {st_dim} "
          f"--batch_size 64 --workers 4 --valfreq 10")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
