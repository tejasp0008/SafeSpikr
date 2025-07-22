import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- Add root directory to path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.preprocess import load_dataset


def create_torch_dataloader(batch_size=1):
    print("Loading and preprocessing dataset...")
    X_img, X_ppg, y_cat, _ = load_dataset()

    X_img_tensor = torch.from_numpy(X_img).permute(0, 3, 1, 2).float()
    X_ppg_tensor = torch.from_numpy(X_ppg).float()
    y_tensor = torch.from_numpy(y_cat).float()

    dataset = TensorDataset(X_img_tensor, X_ppg_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("DataLoader created successfully.")
    return loader


def quantize_and_export():
    from model.snn_model import SNNDriverStateClassifier

    MODEL_PATH = "snn_model.pth"
    device = torch.device("cpu")

    print(f"Loading model from: {MODEL_PATH}")
    float_model = SNNDriverStateClassifier().to(device)
    float_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    float_model.eval()

    print("\nQuantizing model...")
    quantized_model = torch.quantization.quantize_dynamic(
        float_model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    print("--- Model Quantized Successfully ---")

    verilog_dir = "verilog"
    if not os.path.exists(verilog_dir):
        os.makedirs(verilog_dir)

    try:   
        # ✅ These are quantized (nn.Linear)
        weights_classifier_fc = quantized_model.fc.weight().int_repr().detach().numpy()
        np.savetxt(os.path.join(verilog_dir, "weights_classifier_fc.txt"), weights_classifier_fc, fmt='%x')

        weights_ppg_fc = quantized_model.ppg_encoder.fc.weight().int_repr().detach().numpy()
        np.savetxt(os.path.join(verilog_dir, "weights_ppg_fc.txt"), weights_ppg_fc, fmt='%x')

        # ❌ These are still float (nn.Conv2d), so use .data
        weights_img_conv1 = quantized_model.img_encoder.conv1.weight.data.numpy()
        np.savetxt(os.path.join(verilog_dir, "weights_img_conv1.txt"), weights_img_conv1.flatten(), fmt='%f')

        weights_img_conv2 = quantized_model.img_encoder.conv2.weight.data.numpy()
        np.savetxt(os.path.join(verilog_dir, "weights_img_conv2.txt"), weights_img_conv2.flatten(), fmt='%f')

        print("--- Weights exported successfully ---")

    

    except AttributeError as e:
        print("\nERROR: Could not find a layer. This can happen if the model structure changes.")
        print(f"Original error: {e}")
        return

    print("\n--- Generating Test Vector ---")
    test_loader = create_torch_dataloader(batch_size=1)
    img_input, ppg_input, _ = next(iter(test_loader))

    img_vector = img_input.numpy().flatten()
    ppg_vector = ppg_input.numpy().flatten()
    np.savetxt(os.path.join(verilog_dir, "test_vector_img_input.txt"), img_vector, fmt='%f')
    np.savetxt(os.path.join(verilog_dir, "test_vector_ppg_input.txt"), ppg_vector, fmt='%f')
    print(f"Saved test input vectors to {verilog_dir}/")


if __name__ == "__main__":
    quantize_and_export()
