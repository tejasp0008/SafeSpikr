#!/usr/bin/env python3
"""
Merge StateFarm (alert, distracted) + DDD (drowsy) into a unified 3-class dataset
and generate synthetic PPG files.

Usage:
  python tools/merge_datasets.py \
    --statefarm_root Data/imgs/train \
    --statefarm_csv Data/driver_imgs_list.csv \
    --ddd_root DDD \
    --out_root Data_unified \
    --img_size 128 \
    --num_per_class 5000
"""

import argparse
import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def resize_and_copy(src_path, dst_path, size):
    img = Image.open(src_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    img.save(dst_path, quality=95)

def synthetic_ppg_for_label(label, length=100, seed=0):
    rnd = np.random.RandomState(seed)
    t = np.linspace(0, 1, length)
    if label == "alert":
        base = 0.6 * np.sin(2 * np.pi * 3 * t)
        noise = 0.05 * rnd.randn(length)
        return base + noise
    elif label == "drowsy":
        base = 0.35 * np.sin(2 * np.pi * 1.8 * t)
        noise = 0.08 * rnd.randn(length)
        drift = 0.05 * np.linspace(0, 1, length)
        return base + noise - drift
    else:  # distracted
        base = 0.5 * np.sin(2 * np.pi * 3 * t)
        noise = 0.12 * rnd.randn(length)
        for i in range(3):
            pos = rnd.randint(5, length-5)
            base[pos:pos+3] += rnd.uniform(-1.0, 1.0)
        return base + noise

# ----------------------------
# Collect images
# ----------------------------
def collect_statefarm(statefarm_root, statefarm_csv):
    """Map StateFarm c0->alert, c1..c9->distracted"""
    df = pd.read_csv(statefarm_csv)
    items = []
    for _, row in df.iterrows():
        classname = str(row["classname"])
        img_name = str(row["img"])
        candidate1 = Path(statefarm_root) / classname / img_name
        candidate2 = Path(statefarm_root) / img_name
        if classname == "c0":
            label = "alert"
        else:
            label = "distracted"
        if candidate1.exists():
            items.append((str(candidate1), label))
        elif candidate2.exists():
            items.append((str(candidate2), label))
    return items

def collect_ddd(ddd_root):
    """Map DDD non_drowsy->alert, drowsy->drowsy"""
    items = []
    ddd_root = Path(ddd_root)
    for f in (ddd_root / "non_drowsy").glob("*.*"):
        items.append((str(f), "alert"))
    for f in (ddd_root / "drowsy").glob("*.*"):
        items.append((str(f), "drowsy"))
    return items

# ----------------------------
# Build dataset
# ----------------------------
def build_dataset(statefarm_root, statefarm_csv, ddd_root, out_root,
                  img_size=128, num_per_class=None, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    out_root = Path(out_root)
    images_out = out_root / "images"
    ppg_out = out_root / "ppg"
    ensure_dir(images_out)
    ensure_dir(ppg_out)

    # collect
    sf_items = collect_statefarm(statefarm_root, statefarm_csv)
    ddd_items = collect_ddd(ddd_root)

    unified = sf_items + ddd_items
    print(f"Total collected: {len(unified)}")

    grouped = {"alert": [], "drowsy": [], "distracted": []}
    for path, lbl in unified:
        grouped[lbl].append(path)

    for k in grouped:
        print(f"{k}: {len(grouped[k])} samples")

    if num_per_class:
        for k in grouped:
            random.shuffle(grouped[k])
            grouped[k] = grouped[k][:num_per_class]

    # split train/val/test
    all_records = []
    for lbl, paths in grouped.items():
        random.shuffle(paths)
        n = len(paths)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        splits = (["train"] * (n - n_val - n_test) +
                  ["val"] * n_val +
                  ["test"] * n_test)
        for split, src in zip(splits, paths):
            all_records.append((src, split, lbl))

    random.shuffle(all_records)

    # copy images + generate ppg
    rows = []
    cnt = 0
    for src, split, label in tqdm(all_records, desc="Processing"):
        cnt += 1
        dst_img_dir = images_out / split / label
        dst_ppg_dir = ppg_out / split / label
        ensure_dir(dst_img_dir)
        ensure_dir(dst_ppg_dir)

        fname = f"{label}_{cnt:06d}.jpg"
        dst_img_path = dst_img_dir / fname
        resize_and_copy(src, dst_img_path, img_size)

        ppg = synthetic_ppg_for_label(label, length=100, seed=cnt)
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        ppg_fname = f"{label}_{cnt:06d}_ppg.csv"
        dst_ppg_path = dst_ppg_dir / ppg_fname
        np.savetxt(dst_ppg_path, ppg, fmt="%.6f", delimiter=",")

        rows.append({
            "split": split,
            "img_path": os.path.relpath(dst_img_path, out_root).replace("\\", "/"),
            "ppg_path": os.path.relpath(dst_ppg_path, out_root).replace("\\", "/"),
            "label": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "labels.csv", index=False)
    print(f"Saved {len(df)} samples to {out_root}/labels.csv")

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--statefarm_root", type=str, required=True)
    p.add_argument("--statefarm_csv", type=str, required=True)
    p.add_argument("--ddd_root", type=str, required=True)
    p.add_argument("--out_root", type=str, default="Data_unified")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_per_class", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_dataset(args.statefarm_root, args.statefarm_csv, args.ddd_root,
                  args.out_root, img_size=args.img_size,
                  num_per_class=args.num_per_class,
                  val_frac=args.val_frac, test_frac=args.test_frac,
                  seed=args.seed)

if __name__ == "__main__":
    main()
