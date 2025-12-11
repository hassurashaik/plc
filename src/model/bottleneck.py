# src/model/bottleneck.py
"""
F–T–GRU Bottleneck module for BS-PLCNet2.

Processes 4D features [B, C, T, F] using GRU in BOTH time and frequency domains.
This increases temporal and spectral context without heavy convolution.

Pipeline:
    1) Project input to hidden channels
    2) GRU over time dimension (per frequency bin)
    3) GRU over frequency dimension (per time frame)
    4) Residual connection + normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FTGRU(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, bidirectional=False):
        """
        Args:
            in_ch: input channels (C)
            hidden_ch: channels in GRU hidden state
            bidirectional: if True, uses bi-GRU (paper uses uni-GRU for causal mode)
        """
        super().__init__()

        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.bidirectional = bidirectional
        mult = 2 if bidirectional else 1

        # Project input → hidden channels
        self.pre = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)

        # GRU over time: process sequence T for each frequency bin independently
        self.gru_time = nn.GRU(
            input_size=hidden_ch,
            hidden_size=hidden_ch,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # GRU over frequency: process sequence F for each time step independently
        self.gru_freq = nn.GRU(
            input_size=hidden_ch * mult,
            hidden_size=hidden_ch,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Project back to original channel dimension
        self.post = nn.Conv2d(hidden_ch * mult, in_ch, kernel_size=1)

        self.norm = nn.BatchNorm2d(in_ch)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, mode="causal") -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]
            mode: "causal" or "noncausal" (GRU itself is noncausal, but mode kept for API)
        """
        B, C, T, F = x.shape

        # 1) Project channels: [B, C, T, F] → [B, H, T, F]
        h = self.pre(x)

        # 2) GRU over TIME — treat freq bins separately
        # Reshape to [B*F, T, H]
        h_t = h.permute(0, 3, 2, 1).contiguous()  # [B, F, T, H]
        h_t = h_t.view(B * F, T, self.hidden_ch)
        h_time_out, _ = self.gru_time(h_t)
        # reshape back to [B, H*mult, T, F]
        h_time_out = h_time_out.view(B, F, T, -1).permute(0, 3, 2, 1)

        # 3) GRU over FREQUENCY — treat time frames separately
        # Reshape to [B*T, F, H*mult]
        _, Ht, _, _ = h_time_out.shape  # Ht = hidden_ch * mult
        h_f = h_time_out.permute(0, 2, 3, 1).contiguous()  # [B, T, F, Ht]
        h_f = h_f.view(B * T, F, Ht)
        h_freq_out, _ = self.gru_freq(h_f)
        # reshape back to [B, H*mult, T, F]
        h_freq_out = h_freq_out.view(B, T, F, -1).permute(0, 3, 1, 2)

        # 4) Project back + residual + norm + activation
        out = self.post(h_freq_out)
        out = out + x  # residual connection
        out = self.norm(out)
        out = self.act(out)

        return out


if __name__ == "__main__":
    # Self-test
    B, C, T, F = 2, 32, 128, 64
    x = torch.randn(B, C, T, F)
    bott = FTGRU(in_ch=C, hidden_ch=32)
    o = bott(x)
    print("Output shape:", o.shape)  # Expected [2, 32, 128, 64]
