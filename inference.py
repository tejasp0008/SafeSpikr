import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- Add root directory to path ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# --- Import model and preprocessing ---
from model.snn_model import SNNDriverStateClassifier
from utils.preprocess import load_dataset

def create_torch_dataloader(batch_size=1):
    print("[INFO] Loading and preprocessing dataset...")
    X_img, X_ppg, y_cat, _ = load_dataset()

    X_img_tensor = torch.from_numpy(X_img).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Fix starts here 👇
    if X_ppg.ndim == 4:
        X_ppg = np.squeeze(X_ppg, axis=-1)  # from (B, 100, 1, 1) → (B, 100, 1)
    X_ppg_tensor = torch.from_numpy(X_ppg).float()
    # Fix ends here 👆

    y_tensor = torch.from_numpy(y_cat).float()

    dataset = TensorDataset(X_img_tensor, X_ppg_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("[INFO] DataLoader created successfully.")
    return loader


def run_inference():
    """
    Load model and run a forward pass on one sample.
    """
    MODEL_PATH = os.path.join(ROOT_DIR, "snn_model.pth")
    device = torch.device("cpu")

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = SNNDriverStateClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Get one sample
    loader = create_torch_dataloader(batch_size=1)
    img_input, ppg_input, _ = next(iter(loader))

    print(f"[DEBUG] Image Input Shape: {img_input.shape}")  # Expect: (1, 3, H, W)
    print(f"[DEBUG] PPG Input Shape: {ppg_input.shape}")    # Expect: (1, 100, 1)

    # Inference
    with torch.no_grad():
        output = model(img_input, ppg_input)
        prediction = torch.argmax(output, dim=1).item()

    print(f"[DEBUG] Raw Model Output: {output}")

    # Human-readable prediction
    class_labels = {
        0: "🟢 Awake",
        1: "⚠️ Drowsy",
        2: "❌ Distracted"
    }
    print(f"[RESULT] Predicted Driver State: {class_labels.get(prediction, 'Unknown')}")

if __name__ == "__main__":
    run_inference()
