# model/snn_train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from model.dataset_unified import UnifiedDataset
from model.snn_model_statefarm import SNNDriverStateClassifier  # assuming you already have this


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for imgs, ppgs, labels in loader:
        imgs, ppgs, labels = imgs.to(device), ppgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, ppgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, ppgs, labels in loader:
            imgs, ppgs, labels = imgs.to(device), ppgs.to(device), labels.to(device)
            outputs = model(imgs, ppgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
   
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="Data_unified/labels.csv",
                        help="Path to labels.csv from unified dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets
    train_ds = UnifiedDataset(csv_path=args.csv_path, split="train", max_items=args.max_items)
    val_ds = UnifiedDataset(csv_path=args.csv_path, split="val", max_items=args.max_items)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    model = SNNDriverStateClassifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "snn_model_best.pth"))
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")

if __name__ == "__main__":
    main()
