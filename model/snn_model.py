import torch
import torch.nn as nn
import norse.torch as snn

class SpikingImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)      # Output: (B, 12, 64, 64)
        self.pool = nn.MaxPool2d(2)                      # Output: (B, 12, 32, 32)

        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)     # Output: (B, 24, 32, 32)
        self.pool2 = nn.MaxPool2d(2)                     # Output: (B, 24, 16, 16)

        flat_dim1 = 12 * 32 * 32
        flat_dim2 = 24 * 16 * 16

        self.lif1 = snn.LIFRecurrentCell(flat_dim1, 128)
        self.lif2 = snn.LIFRecurrentCell(flat_dim2, 128)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.pool(torch.relu(self.conv1(x)))         # (B, 12, 32, 32)
        x_flat1 = x.view(batch_size, -1)                 # (B, 12*32*32)
        spk1, _ = self.lif1(x_flat1)

        x = self.pool2(torch.relu(self.conv2(x)))        # (B, 24, 16, 16)
        x_flat2 = x.view(batch_size, -1)                 # (B, 24*16*16)
        spk2, _ = self.lif2(x_flat2)

        return spk2  # shape: (B, 128)


class SpikingPPGEncoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lif = snn.LIFRecurrentCell(hidden_dim, 64)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # (B, 100, 1)
        x = x.view(batch_size, -1)        # Flatten PPG: (B, 100)
        x = self.fc(x)                    # (B, 64)
        spk, _ = self.lif(x)              # (B, 64)
        return spk


class SNNDriverStateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = SpikingImageEncoder()     # Output: (B, 128)
        self.ppg_encoder = SpikingPPGEncoder()       # Output: (B, 64)
        self.fc = nn.Linear(128 + 64, 3)             # Combined features → 3 classes

    def forward(self, img, ppg):
        img_feat = self.img_encoder(img)
        ppg_feat = self.ppg_encoder(ppg)
        combined = torch.cat([img_feat, ppg_feat], dim=1)  # (B, 192)
        out = self.fc(combined)
        return out
