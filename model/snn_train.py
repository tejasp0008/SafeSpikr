import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Fix module import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import load_dataset
from snn_model import SNNDriverStateClassifier

# -------------------------------
# Load preprocessed data
# -------------------------------
X_img, X_ppg, y_cat, class_names = load_dataset()

# Convert to torch tensors
X_img_torch = torch.tensor(X_img, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: [B, C, H, W]
X_ppg_torch = torch.tensor(X_ppg, dtype=torch.float32)
y_torch = torch.tensor(y_cat, dtype=torch.float32)

# Split
X_img_train, X_img_val, X_ppg_train, X_ppg_val, y_train, y_val = train_test_split(
    X_img_torch, X_ppg_torch, y_torch, test_size=0.2, random_state=42
)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_img_train, X_ppg_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_img_val, X_ppg_val, y_val), batch_size=32)

# -------------------------------
# Init model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNDriverStateClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(10):
    model.train()
    total_loss = 0
    for img, ppg, label in train_loader:
        img, ppg, label = img.to(device), ppg.to(device), label.to(device)

        # Forward pass
        out = model(img, ppg)

        # Use argmax to convert one-hot to class index
        loss = criterion(out, torch.argmax(label, dim=1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "snn_model.pth")
print("✅ Model saved as snn_model.pth")

