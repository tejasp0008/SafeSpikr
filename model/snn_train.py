# model/snn_train.py
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from model.dataset_from_csv import StateFarmCSVDataset
from model.snn_model_statefarm import SNNDriverStateClassifier
import numpy as np

# -----------------------
# Config / hyperparams
# -----------------------
DEFAULTS = {
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "max_items": None,     # limit dataset for quick runs (set None to use all)
    "val_fraction": 0.15,
    "output_dir": ".",
    "seed": 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, ppgs, labels in loader:
        imgs = imgs.to(device)
        ppgs = ppgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(imgs, ppgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, ppgs, labels in loader:
            imgs = imgs.to(device)
            ppgs = ppgs.to(device)
            labels = labels.to(device)
            logits, _ = model(imgs, ppgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset
    ds = StateFarmCSVDataset(max_items=args.max_items)
    N = len(ds)
    if N == 0:
        raise RuntimeError("Dataset is empty. Check CSV path and image layout.")
    n_val = max(1, int(args.val_fraction * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"Dataset size: {N}  Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model, loss, optimizer
    model = SNNDriverStateClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # logging & checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # TensorBoard scalars
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        # checkpoint last
        last_path = os.path.join(args.output_dir, "snn_model_last.pth")
        torch.save(model.state_dict(), last_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = os.path.join(args.output_dir, "snn_model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (val_acc={best_val_acc:.4f}) -> {best_path}")

    print("Training complete. Best val acc:", best_val_acc, "at epoch", best_epoch)
    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--max_items", type=int, default=DEFAULTS["max_items"])
    p.add_argument("--val_fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = p.parse_args()
    main(args)
