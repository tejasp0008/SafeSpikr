# model/dataset_from_csv.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# CONFIG (adjust if your Data folder is somewhere else)
ROOT = os.path.join(os.getcwd(), "Data", "imgs", "train")  # where images are stored
CSV_PATH = os.path.join(os.getcwd(), "Data", "driver_imgs_list.csv")
IMG_SIZE = (128, 128)
PPG_LEN = 100

# Map StateFarm classname -> our label. If CSV uses numeric classes (c0..c9),
# map them here. Default: c0 -> alert, everything else -> distracted.
CLASS_MAP = {
    # example mappings (adjust if your csv uses different names)
    "c0": "alert",
    "c1": "distracted",
    "c2": "distracted",
    "c3": "distracted",
    "c4": "distracted",
    "c5": "distracted",
    "c6": "distracted",
    "c7": "distracted",
    "c8": "distracted",
    "c9": "distracted",
}
LABEL_TO_IDX = {"alert": 0, "drowsy": 1, "distracted": 2}

def synthetic_ppg_for_label(label, length=PPG_LEN, seed=0):
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

class StateFarmCSVDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, root=ROOT, img_size=IMG_SIZE,
                 ppg_len=PPG_LEN, transform=None, max_items=None):
        assert os.path.exists(csv_path), f"CSV missing: {csv_path}"
        self.df = pd.read_csv(csv_path)
        # CSV format may be: classname,img,subject  (common StateFarm)
        # We keep rows as-is and compute image path.
        self.root = root
        self.img_size = img_size
        self.ppg_len = ppg_len
        self.transform = transform
        if max_items:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _map_label(self, classname):
        if classname in CLASS_MAP:
            return CLASS_MAP[classname]
        # fallback heuristics
        lname = str(classname).lower()
        if "safe" in lname or "c0" in lname:
            return "alert"
        return "distracted"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        classname = str(row['classname'])
        img_name = row['img']  # adjust if CSV column is named differently
        # image path: try foldered layout first: <root>/<classname>/<img>
        candidate1 = os.path.join(self.root, classname, img_name)
        candidate2 = os.path.join(self.root, img_name)  # flat layout
        if os.path.exists(candidate1):
            img_path = candidate1
        elif os.path.exists(candidate2):
            img_path = candidate2
        else:
            raise FileNotFoundError(f"Image not found for row {idx}: tried {candidate1} and {candidate2}")

        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size)
        img_np = np.array(img).astype("float32") / 127.5 - 1.0
        img_np = np.transpose(img_np, (2,0,1))  # C,H,W
        mapped = self._map_label(classname)
        # create deterministic synthetic ppg
        ppg = synthetic_ppg_for_label(mapped, length=self.ppg_len, seed=idx)
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        img_t = torch.tensor(img_np).float()
        ppg_t = torch.tensor(ppg).float()
        label_idx = LABEL_TO_IDX.get(mapped, 2)
        return img_t, ppg_t, label_idx
