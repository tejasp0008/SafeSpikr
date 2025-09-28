# model/dataset_unified.py
"""
Dataset loader for Data_unified produced by tools/merge_datasets.py

CSV format expected (relative to the dataset root passed in):
  split,img_path,ppg_path,label

Example row:
  train,images/train/alert/alert_000001.jpg,ppg/train/alert/alert_000001_ppg.csv,alert

Usage:
  from model.dataset_unified import UnifiedDataset
  ds = UnifiedDataset(csv_path="Data_unified/labels.csv", split="train", max_items=None)
  img, ppg, label = ds[0]   # img: torch.FloatTensor (3,H,W), ppg: torch.FloatTensor (ppg_len,), label: int
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# default image size and PPG length (match merge script)
DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_PPG_LEN = 100

LABEL_TO_IDX = {"alert": 0, "drowsy": 1, "distracted": 2}

class UnifiedDataset(Dataset):
    def __init__(self, csv_path="Data_unified/labels.csv", root=None, split="train",
                 img_size=DEFAULT_IMG_SIZE, ppg_len=DEFAULT_PPG_LEN, max_items=None, transforms=None):
        """
        Args:
          csv_path: path to Data_unified/labels.csv (absolute or relative)
          root: optional base folder for CSV-relative paths (default: parent dir of csv_path)
          split: 'train'|'val'|'test' (filters rows)
          img_size: (W,H) or (H,W) tuple, will be used with PIL resize (expects square usually)
          ppg_len: expected PPG length (if CSV shorter/longer, we pad/truncate)
          max_items: optional int to limit dataset for quick tests
          transforms: optional callable on PIL image (after resize) -> np array
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"labels CSV not found: {csv_path}")
        self.root = Path(root) if root else self.csv_path.parent
        self.split = split
        self.img_size = tuple(img_size)
        self.ppg_len = ppg_len
        self.transforms = transforms

        df = pd.read_csv(self.csv_path)
        # filter by split
        df = df[df["split"].astype(str) == str(split)]
        df = df.reset_index(drop=True)
        if max_items is not None:
            df = df.iloc[:max_items].reset_index(drop=True)

        # Store absolute paths for speed
        rows = []
        for _, r in df.iterrows():
            img_rel = str(r["img_path"])
            ppg_rel = str(r["ppg_path"])
            lbl = str(r["label"])
            img_abs = (self.root / img_rel).resolve()
            ppg_abs = (self.root / ppg_rel).resolve()
            rows.append({
                "img_path": str(img_abs),
                "ppg_path": str(ppg_abs),
                "label": lbl
            })
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path):
        # load, resize, normalize to [-1,1], convert to C,H,W float32 tensor
        img = Image.open(path).convert("RGB")
        img = img.resize(self.img_size, Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        # normalize to [-1,1]
        img_np = (img_np / 127.5) - 1.0
        # channel first
        img_np = np.transpose(img_np, (2, 0, 1))
        return torch.from_numpy(img_np).float()

    def _load_ppg(self, path):
        # expects CSV with one column or a single row of values
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()
        # pad / trim to ppg_len
        if arr.size < self.ppg_len:
            pad = np.zeros(self.ppg_len - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        elif arr.size > self.ppg_len:
            arr = arr[: self.ppg_len]
        # normalize zero mean, unit var (small eps)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return torch.from_numpy(arr).float()

    def _map_label(self, label_str):
        if label_str not in LABEL_TO_IDX:
            # fallback heuristics
            l = label_str.lower()
            if "drows" in l or "closed" in l:
                return LABEL_TO_IDX["drowsy"]
            if "c0" in l or "alert" in l or "open" in l:
                return LABEL_TO_IDX["alert"]
            return LABEL_TO_IDX["distracted"]
        return LABEL_TO_IDX[label_str]

    def __getitem__(self, idx):
        entry = self.rows[idx]
        img_path = entry["img_path"]
        ppg_path = entry["ppg_path"]
        label_str = entry["label"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(ppg_path):
            raise FileNotFoundError(f"PPG file not found: {ppg_path}")

        img_t = self._load_image(img_path)
        ppg_t = self._load_ppg(ppg_path)
        label_idx = self._map_label(label_str)
        return img_t, ppg_t, label_idx

if __name__ == "__main__":
    # quick smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Data_unified/labels.csv")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_items", type=int, default=5)
    args = parser.parse_args()

    ds = UnifiedDataset(csv_path=args.csv, split=args.split, max_items=args.max_items)
    print("Dataset length:", len(ds))
    for i in range(min(5, len(ds))):
        img, ppg, lbl = ds[i]
        print(f"{i}: img={tuple(img.shape)}, ppg={tuple(ppg.shape)}, label={lbl}")
