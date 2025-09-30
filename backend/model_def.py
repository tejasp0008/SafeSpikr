# backend/model_def.py
"""
Model definition wrapper for SafeSpikr backend.
This mirrors your training model (snn_model_statefarm.py) and exposes get_model().
"""

import torch
import torch.nn as nn

# try norse
try:
    import norse.torch as norse
    NORSE_AVAILABLE = True
except Exception:
    NORSE_AVAILABLE = False

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

class PPGEncoder(nn.Module):
    def __init__(self, n_input=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.fc(x)

class SNNDriverStateClassifier(nn.Module):
    def __init__(self, ppg_input_len=100, n_classes=3, use_norse=NORSE_AVAILABLE):
        super().__init__()
        self.img_enc = ImgEncoder()
        self.ppg_enc = PPGEncoder(n_input=ppg_input_len)
        self.fc = nn.Linear(32 + 64, 128)
        self.use_norse = use_norse
        if use_norse:
            # LIFCell returns spikes, state etc.
            self.lif = norse.LIFCell()
        else:
            self.lif = nn.ReLU()
        self.out_fc = nn.Linear(128, n_classes)

    def forward(self, img, ppg):
        """
        Forward signature matches training: (img, ppg) -> (logits, spikes/hidden)
        img: (B,3,H,W), ppg: (B, ppg_input_len)
        """
        i_feat = self.img_enc(img)
        p_feat = self.ppg_enc(ppg)
        x = torch.cat([i_feat, p_feat], dim=1)  # (B, 96)
        x = self.fc(x)                           # (B, 128)
        if self.use_norse:
            s, _ = self.lif(x)
        else:
            s = self.lif(x)
        out = self.out_fc(s)
        return out, s

def get_model(ppg_input_len: int = 100, n_classes: int = 3, use_norse: bool | None = None):
    """
    Return an instance of the model with the same defaults used in training.
    - use_norse: if None, tries to use NORSE_AVAILABLE flag
    """
    if use_norse is None:
        use_norse = NORSE_AVAILABLE
    m = SNNDriverStateClassifier(ppg_input_len=ppg_input_len, n_classes=n_classes, use_norse=use_norse)
    return m
