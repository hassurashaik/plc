# src/model/simple_model.py
import torch.nn as nn
import torch

class SimplePLCNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [B, T] -> [B,1,T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.enc(x)
        out = self.dec(h)
        return out.squeeze(1)  # [B,T]
