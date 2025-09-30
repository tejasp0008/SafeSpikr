"""
Export quantized weights + test vectors from a trained SNN model
for FPGA/Verilog simulation.

Usage:
  python -m model.export_traces_statefarm \
    --model_path outputs/unified_run1/snn_model_best.pth \
    --csv_path Data_unified/labels.csv \
    --split test \
    --out_dir verilog \
    --num_samples 10 \
    --q_int 1 --q_frac 15
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset_unified import UnifiedDataset
from model.snn_model_statefarm import SNNDriverStateClassifier  # same model used in training

LABELS = {0: "alert", 1: "drowsy", 2: "distracted"}


def quantize_tensor(t, q_int=1, q_frac=15):
    """Quantize float tensor into fixed-point Qm.n format"""
    scale = 2 ** q_frac
    t_q = torch.clamp(
        torch.round(t * scale),
        -(2 ** (q_int + q_frac - 1)),
        2 ** (q_int + q_frac - 1) - 1
    )
    return t_q.int().cpu().numpy()


def export_weights(model, weights_dir, q_int=1, q_frac=15):
    """Export all model weights into txt files"""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_files = {}
    for name, param in model.state_dict().items():
        q_arr = quantize_tensor(param, q_int=q_int, q_frac=q_frac)
        fname = f"weights_{name.replace('.', '_')}.txt"
        fpath = weights_dir / fname
        np.savetxt(fpath, q_arr.flatten(), fmt="%d")
        weight_files[name] = str(fpath)
    return weight_files


def export_test_vectors(ds, model, test_dir, num_samples=5, q_int=1, q_frac=15, device="cpu"):
    """Export a few sample image+ppg pairs + logits + golden reference"""
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    vectors = []
    golden = []

    with torch.no_grad():
        for idx, (img, ppg, label) in enumerate(dl):
            if idx >= num_samples:
                break

            img = img.to(device)
            ppg = ppg.to(device)
            outputs = model(img, ppg)
            if isinstance(outputs, tuple):  # handle (logits, spikes)
                logits = outputs[0]
            else:
                logits = outputs

            # float32 logits (before softmax)
            logits_np = logits.cpu().numpy()[0]

            # quantized logits
            logits_q = quantize_tensor(logits, q_int=q_int, q_frac=q_frac).flatten()

            # save logits file (quantized)
            logits_file = test_dir / f"test_vector_logits_{idx}.txt"
            np.savetxt(logits_file, logits_q.reshape(1, -1), fmt="%d")

            # softmax probs (for CSV reference only)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            true_label = int(label.item())

            # quantize + save inputs
            img_q = quantize_tensor(img, q_int=q_int, q_frac=q_frac).flatten()
            ppg_q = quantize_tensor(ppg, q_int=q_int, q_frac=q_frac).flatten()

            img_file = test_dir / f"test_vector_img_{idx}.txt"
            ppg_file = test_dir / f"test_vector_ppg_{idx}.txt"
            np.savetxt(img_file, img_q, fmt="%d")
            np.savetxt(ppg_file, ppg_q, fmt="%d")

            vectors.append({
                "idx": idx,
                "img_file": str(img_file),
                "ppg_file": str(ppg_file),
                "logits_file": str(logits_file),
                "true_label": true_label,
                "pred_label": pred
            })

            golden.append({
                "idx": idx,
                "true_label_idx": true_label,
                "true_label": LABELS[true_label],
                "pred_label_idx": pred,
                "pred_label": LABELS[pred],
                "logits_float": logits_np.tolist(),
                "logits_quant": logits_q.tolist(),
                "prob0": float(probs[0]),
                "prob1": float(probs[1]),
                "prob2": float(probs[2]),
            })

    # save golden reference CSV
    import pandas as pd
    golden_df = pd.DataFrame(golden)
    golden_path = test_dir / "golden_reference.csv"
    golden_df.to_csv(golden_path, index=False)

    return vectors, str(golden_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--csv_path", type=str, default="Data_unified/labels.csv")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default="verilog")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--q_int", type=int, default=1)
    p.add_argument("--q_frac", type=int, default=15)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    ds = UnifiedDataset(csv_path=args.csv_path, split=args.split, max_items=args.num_samples)

    # model
    model = SNNDriverStateClassifier().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir)
    weights_dir = out_dir / "weights"
    test_dir = out_dir / "test_vectors"

    # export
    weight_files = export_weights(model, weights_dir, q_int=args.q_int, q_frac=args.q_frac)
    vectors, golden_path = export_test_vectors(
        ds, model, test_dir,
        num_samples=args.num_samples,
        q_int=args.q_int,
        q_frac=args.q_frac,
        device=device
    )

    # manifest
    manifest = {
        "model_path": args.model_path,
        "csv_path": args.csv_path,
        "split": args.split,
        "num_samples": args.num_samples,
        "q_format": f"Q{args.q_int}.{args.q_frac}",
        "weights_dir": str(weights_dir),
        "test_vectors_dir": str(test_dir),
        "weights": weight_files,
        "test_vectors": vectors,
        "golden_reference": golden_path
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Export complete. Outputs saved in {out_dir}")
    print(f"   Weights in: {weights_dir}")
    print(f"   Test vectors in: {test_dir}")
    print(f"   Golden reference: {golden_path}")


if __name__ == "__main__":
    main()
