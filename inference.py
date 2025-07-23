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

    if X_ppg.ndim == 4:
        X_ppg = np.squeeze(X_ppg, axis=-1)  # (B, 100, 1)

    X_ppg_tensor = torch.from_numpy(X_ppg).float()
    y_tensor = torch.from_numpy(y_cat).float()

    dataset = TensorDataset(X_img_tensor, X_ppg_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("[INFO] DataLoader created successfully.")
    return loader


def save_spike_tensor_to_mem(spike_tensor, filename="spike_tensor.mem", binary=False):
    tensor_flat = spike_tensor.view(-1).cpu().numpy()  # Flatten to 1D
    output_path = os.path.join(ROOT_DIR, filename)

    with open(output_path, "w") as f:
        for val in tensor_flat:
            int_val = int(val) if not np.isnan(val) else 0
            if binary:
                f.write(f"{int_val:08b}\n")   # 8-bit binary
            else:
                f.write(f"{int_val:02X}\n")   # 2-digit hex

    print(f"[INFO] Spike tensor saved to {output_path} (binary={binary})")


def run_inference():
    MODEL_PATH = os.path.join(ROOT_DIR, "snn_model.pth")
    device = torch.device("cpu")

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = SNNDriverStateClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    loader = create_torch_dataloader(batch_size=1)
    img_input, ppg_input, _ = next(iter(loader))

    print(f"[DEBUG] Image Input Shape: {img_input.shape}")  # (1, 3, H, W)
    print(f"[DEBUG] PPG Input Shape: {ppg_input.shape}")    # (1, 100, 1)

    # --- Get spike tensor from image encoder ---
    with torch.no_grad():
        spike_tensor = model.img_encoder(img_input)  # shape: (1, 128)

    print(f"[DEBUG] Spike Tensor Shape: {spike_tensor.shape}")
    print(f"[DEBUG] Spike Tensor Preview: {spike_tensor[0, :8]}")

    # --- Save spike tensor to .mem file for Verilog ---
    save_spike_tensor_to_mem(spike_tensor, "spike_tensor.mem", binary=False)

    # --- Inference with full model ---
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
