# model/eval_test.py
"""
Evaluate the trained SNN on the test split and save metrics + confusion matrix.

Usage examples (from repo root):
  python -m model.eval_test --model_path outputs/unified_run1/snn_model_best.pth --csv_path Data_unified/labels.csv --split test

Outputs (in outputs/eval_<ts>/):
  - metrics.txt          (accuracy, loss, per-class precision/recall/f1)
  - confusion_matrix.png
  - predictions.csv      (idx, img_path, true_label, pred_label, probabilities)
  - metrics.json         (machine-readable metrics)
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import dataset & model
from model.dataset_unified import UnifiedDataset
from model.snn_model_statefarm import SNNDriverStateClassifier  # use same model as training

LABELS = {0: "alert", 1: "drowsy", 2: "distracted"}

def evaluate(model_path, csv_path="Data_unified/labels.csv", split="test",
             batch_size=32, max_items=None, device=None, out_dir="outputs"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(out_dir) / f"eval_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # dataset
    ds = UnifiedDataset(csv_path=csv_path, split=split, max_items=max_items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # load model
    model = SNNDriverStateClassifier().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_true, all_pred, all_probs, all_losses, rows = [], [], [], [], []
    sample_idx = 0

    with torch.no_grad():
        for imgs, ppgs, labels in dl:
            imgs, ppgs, labels = imgs.to(device), ppgs.to(device), labels.to(device)
            outputs = model(imgs, ppgs)
            if isinstance(outputs, tuple):  # handle (logits, spikes)
                logits = outputs[0]
            else:
                logits = outputs
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(labels_np.shape[0]):
                lbl = int(labels_np[i])
                pred = int(preds[i])
                pr = probs[i].tolist()
                all_true.append(lbl)
                all_pred.append(pred)
                all_probs.append(pr)
                all_losses.append(float(loss.item()))  # batch loss
                rows.append({
                    "idx": sample_idx,
                    "true_label_idx": lbl,
                    "true_label": LABELS[lbl],
                    "pred_label_idx": pred,
                    "pred_label": LABELS[pred],
                    "prob0": pr[0],
                    "prob1": pr[1],
                    "prob2": pr[2]
                })
                sample_idx += 1

    # metrics
    acc = accuracy_score(all_true, all_pred)
    labels_all = [0, 1, 2]
    target_names = [LABELS[i] for i in labels_all]
    cls_report = classification_report(
        all_true,
        all_pred,
        labels=labels_all,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    conf_mat = confusion_matrix(all_true, all_pred, labels=labels_all)

    # save predictions
    df = pd.DataFrame(rows)
    csv_path_out = outdir / "predictions.csv"
    df.to_csv(csv_path_out, index=False)

    # save metrics json
    metrics = {
        "num_samples": len(all_true),
        "accuracy": float(acc),
        "avg_loss": float(np.mean(all_losses)) if len(all_losses) else None,
        "classification_report": cls_report
    }
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save metrics text
    text_path = outdir / "metrics.txt"
    with open(text_path, "w") as f:
        f.write(f"Test samples: {len(all_true)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(pd.DataFrame(cls_report).transpose().to_string())
    print(f"Saved metrics -> {metrics_path}")

    # confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=[LABELS[i] for i in labels_all],
                yticklabels=[LABELS[i] for i in labels_all])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = outdir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Saved predictions CSV -> {csv_path_out}")
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved human metrics text -> {text_path}")
    print(f"Full JSON metrics -> {metrics_path}")
    return outdir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="outputs/run1/snn_model_best.pth")
    p.add_argument("--csv_path", type=str, default="Data_unified/labels.csv")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()

    outdir = evaluate(
        model_path=args.model_path,
        csv_path=args.csv_path,
        split=args.split,
        batch_size=args.batch_size,
        max_items=args.max_items,
        device=args.device,
        out_dir=args.out_dir
    )
    print("Evaluation outputs in:", outdir)
