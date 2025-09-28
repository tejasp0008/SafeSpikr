# model/export_traces_statefarm.py
"""
Export quantized weights & test vectors from a trained PyTorch SNN
Usage (example):
  python -m model.export_traces_statefarm --model_path outputs/run1/snn_model_best.pth \
      --out_dir verilog --num_samples 5 --q_int 1 --q_frac 15
"""
import os
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

# import your dataset & model (package-style)
from model.dataset_from_csv import StateFarmCSVDataset
from model.snn_model_statefarm import SNNDriverStateClassifier

# fixed-point helpers (simple, local implementations)
def float_to_fixed_array(x: np.ndarray, q_int=1, q_frac=15):
    """
    Convert numpy float array `x` to signed fixed-point integers.
    q_int: number of integer bits (excluding sign)
    q_frac: number of fractional bits
    """
    scale = 2 ** q_frac
    # total bits = sign + q_int + q_frac
    max_int = 2 ** (q_int + q_frac) - 1
    min_int = -2 ** (q_int + q_frac)
    y = np.round(x * scale).astype(np.int64)
    y = np.clip(y, min_int, max_int)
    return y.astype(np.int32)

def save_vector_txt(path: Path, vec: np.ndarray, fmt="%d"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vec.flatten():
            f.write((fmt % int(v)) + "\n")

def resolve_model_path(given_path: str) -> Path:
    """
    Resolve model path:
      - If given_path is absolute, use it.
      - If relative, resolve relative to the script directory (__file__).
    """
    p = Path(given_path)
    if p.is_absolute():
        return p
    # relative -> resolve relative to this script file
    base_dir = Path(__file__).resolve().parent
    return (base_dir / p).resolve()

def load_checkpoint(model_path: Path):
    """
    Load a torch checkpoint and return a state_dict suitable for model.load_state_dict.
    Handles common wrappers like {'model_state_dict':..., 'state_dict':...}
    """
    try:
        print(f"Attempting to load checkpoint from: {model_path}")
        ckpt = torch.load(str(model_path), map_location="cpu")
    except FileNotFoundError:
        print(f"ERROR: File not found: {model_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to torch.load('{model_path}'): {e}")
        raise

    # If the checkpoint *is* already a state_dict (mapping of parameter names -> tensors)
    if isinstance(ckpt, dict):
        # common keys that wrap the actual state dict
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        # else assume this dict *is* a state_dict
        # Heuristic: check if values are tensors
        any_tensor = any(torch.is_tensor(v) for v in ckpt.values())
        if any_tensor:
            return ckpt
        # else we don't know how to interpret this dict
        # still return it and allow model.load_state_dict to error with a clear message
        return ckpt
    else:
        # checkpoint is not a dict (unexpected), return as-is
        return ckpt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="verilog")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--q_int", type=int, default=1)
    p.add_argument("--q_frac", type=int, default=15)
    p.add_argument("--max_items", type=int, default=None)
    args = p.parse_args()

    # Resolve model path robustly
    model_path = resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": str(model_path),
        "q_format": {"q_int": args.q_int, "q_frac": args.q_frac},
        "files": {}
    }

    # Useful diagnostics for debugging path issues
    print("Current working directory:", Path.cwd())
    print("Script file location:", Path(__file__).resolve())
    print("Resolved model_path:", model_path)
    parent = model_path.parent
    if parent.exists():
        try:
            print("Parent directory contents (first 30 entries):")
            for i, pth in enumerate(sorted(parent.iterdir())):
                if i >= 30:
                    print("  ...")
                    break
                print("  ", pth.name)
        except Exception:
            print("  (Could not list parent directory contents)")
    else:
        print("Parent directory does not exist:", parent)

    if not model_path.exists():
        print(f"ERROR: model file not found at: {model_path}")
        sys.exit(2)

    # load dataset & model
    ds = StateFarmCSVDataset(max_items=args.max_items)
    model = SNNDriverStateClassifier()

    # load checkpoint and extract state_dict
    try:
        state_dict = load_checkpoint(model_path)
    except Exception as e:
        print("Exiting due to load error.")
        sys.exit(3)

    # attempt to load into model
    try:
        # If checkpoint contains a top-level object that isn't a state_dict, this may raise
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # often mismatch between keys (e.g., 'module.' prefixes from DataParallel)
        msg = str(e)
        print("RuntimeError while loading state_dict:", msg)
        # try removing 'module.' prefix if present
        new_sd = {}
        changed = False
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
                changed = True
            else:
                new_sd[k] = v
        if changed:
            try:
                print("Retrying load_state_dict after stripping 'module.' prefix...")
                model.load_state_dict(new_sd)
            except Exception as e2:
                print("Second attempt failed:", e2)
                print("You may need to inspect the checkpoint keys vs. model keys.")
                sys.exit(4)
        else:
            print("No 'module.' prefix found; cannot auto-fix mismatch. Inspect keys manually.")
            sys.exit(4)
    except Exception as e:
        print("Unexpected error while loading the state dict:", e)
        sys.exit(4)

    model.eval()

    # Export every state_dict tensor (weights/biases)
    sd = model.state_dict()
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    print("Exporting weights...")
    for k, v in sd.items():
        arr = v.cpu().numpy().astype(np.float32)
        flat = arr.flatten()
        q = float_to_fixed_array(flat, q_int=args.q_int, q_frac=args.q_frac)
        fname = weights_dir / f"weights_{k.replace('.', '_')}.txt"
        save_vector_txt(fname, q)
        manifest["files"][f"weights::{k}"] = {
            "path": str(fname.relative_to(out_dir)),
            "shape": arr.shape,
            "dtype": "int32_fixed"
        }
        print(f"  Wrote {fname}  shape={arr.shape}")

    # Run N samples, export inputs and expected outputs
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    n = min(args.num_samples, len(ds))
    print(f"Exporting {n} samples (inputs + expected outputs)...")
    for i in range(n):
        img_t, ppg_t, label = ds[i]
        img_np = img_t.numpy().astype(np.float32)  # C,H,W
        ppg_np = ppg_t.numpy().astype(np.float32)  # (ppg_len,)

        # Forward pass to get float outputs (spikes/logits)
        with torch.no_grad():
            img_b = torch.tensor(img_np).unsqueeze(0)  # [1,C,H,W]
            ppg_b = torch.tensor(ppg_np).unsqueeze(0)  # [1,ppg_len]
            logits, spikes = model(img_b, ppg_b)
            logits_f = logits.cpu().numpy().squeeze().astype(np.float32)
            # spikes may be tensor or float; turn into numpy 1D
            try:
                spikes_f = spikes.cpu().numpy().squeeze().astype(np.float32)
            except Exception:
                # if spikes is tuple/state, try first element
                spikes_f = np.array(spikes).astype(np.float32).squeeze()

        # Save quantized inputs
        img_flat = img_np.flatten()
        ppg_flat = ppg_np.flatten()
        q_img = float_to_fixed_array(img_flat, q_int=args.q_int, q_frac=args.q_frac)
        q_ppg = float_to_fixed_array(ppg_flat, q_int=args.q_int, q_frac=args.q_frac)
        q_spikes = float_to_fixed_array(spikes_f.flatten(), q_int=args.q_int, q_frac=args.q_frac)
        q_logits = float_to_fixed_array(logits_f.flatten(), q_int=args.q_int, q_frac=args.q_frac)

        sample_prefix = f"sample_{i}"
        img_file = samples_dir / f"{sample_prefix}_img.txt"
        ppg_file = samples_dir / f"{sample_prefix}_ppg.txt"
        spikes_file = samples_dir / f"{sample_prefix}_spikes_expected.txt"
        logits_file = samples_dir / f"{sample_prefix}_logits_expected.txt"
        meta_file = samples_dir / f"{sample_prefix}_meta.json"

        save_vector_txt(img_file, q_img)
        save_vector_txt(ppg_file, q_ppg)
        save_vector_txt(spikes_file, q_spikes)
        save_vector_txt(logits_file, q_logits)

        # Save human metadata for this sample
        meta = {
            "label_index": int(label),
            "label_map": {"0": "alert", "1": "drowsy", "2": "distracted"},
            "img_shape": img_np.shape,
            "ppg_shape": ppg_np.shape,
            "spikes_shape": spikes_f.shape,
            "logits_shape": logits_f.shape,
            "files": {
                "img": str(img_file.relative_to(out_dir)),
                "ppg": str(ppg_file.relative_to(out_dir)),
                "spikes_expected": str(spikes_file.relative_to(out_dir)),
                "logits_expected": str(logits_file.relative_to(out_dir))
            }
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        manifest["files"][f"sample::{i}"] = {
            "meta": str(meta_file.relative_to(out_dir))
        }
        print(f"  Sample {i}: label={label} -> wrote inputs + expected")

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Export complete. Manifest:", manifest_path)
    print("Files written to:", out_dir.resolve())

if __name__ == "__main__":
    main()
