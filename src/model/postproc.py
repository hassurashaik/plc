# src/model/postproc.py
"""
PostProcessModule
-----------------
Lightweight post-processing module for BS-PLCNet2 (operates on low-band, e.g. 0-8 kHz).

Architecture (paper-inspired, simplified and robust):
  - 1x1 Conv -> Tanh (project channels)
  - GRU over time (features per time step = channels * freq_bins_low)
  - Project GRU output back to (hidden * freq) and reshape -> Conv2d -> BatchNorm
  - Final 1x1 Conv -> Tanh to produce same channel count as input

Inputs:
  - x: tensor [B, C, T, F] (C typically 2 for complex real+imag, or 1 for mag)
  - sr: sample rate (for computing low-frequency bin index)
  - n_fft: FFT length used for STFT
  - low_freq: upper bound frequency in Hz to process (default 8000)

Notes:
  - The module only modifies the low-band portion and returns the full-band tensor with the low-band replaced by the processed result.
  - Designed to be small (suitable for running in real-time).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PostProcessModule(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        hidden: int = 128,
        sr: int = 48000,
        n_fft: int = 1024,
        low_freq: int = 8000,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.hidden = hidden
        self.sr = sr
        self.n_fft = n_fft
        self.low_freq = low_freq

        # number of freq bins (rfft)
        self.n_bins = n_fft // 2 + 1
        # compute low-band cutoff bin (inclusive)
        freq_res = sr / n_fft
        self.low_bin = min(self.n_bins, max(1, int(math.ceil(low_freq / freq_res))))

        # layers
        # project input channels -> hidden channels (1x1 conv)
        self.proj_in = nn.Conv2d(in_ch, hidden, kernel_size=1)

        # small conv after projection to mix freq/time locally
        self.pre_conv = nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1))
        self.pre_bn = nn.BatchNorm2d(hidden)
        self.pre_act = nn.Tanh()

        # GRU: will take a flattened vector per time step (hidden * low_bin)
        self.gru = nn.GRU(input_size=hidden * self.low_bin, hidden_size=hidden, batch_first=True)

        # after GRU project back to hidden * low_bin features
        self.post_fc = nn.Linear(hidden, hidden * self.low_bin)

        # post conv to smooth and reduce channels
        self.post_conv = nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1))
        self.post_bn = nn.BatchNorm2d(hidden)
        self.post_act = nn.ELU()

        # final linear 1x1 to map back to original input channels
        self.out_proj = nn.Conv2d(hidden, in_ch, kernel_size=1)
        self.out_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]   (C is channels, e.g. 2 for real+imag)
        Returns:
            out: [B, C, T, F]  (low-band processed, high-band unchanged)
        """
        B, C, T, F = x.shape
        # clamp F to expected bins if mismatch
        assert C == self.in_ch, f"PostProc expected {self.in_ch} channels, got {C}"

        low_bin = min(self.low_bin, F)  # if actual F smaller, use it
        if low_bin <= 0:
            return x

        # split low and high bands
        low = x[:, :, :, :low_bin]   # [B, C, T, low_bin]
        high = x[:, :, :, low_bin:] if low_bin < F else None

        # project in channels
        h = self.proj_in(low)        # [B, hidden, T, low_bin]
        h = self.pre_conv(h)
        h = self.pre_bn(h)
        h = self.pre_act(h)

        # prepare sequence for GRU: flatten freq into features per time step
        # shape -> [B, T, hidden * low_bin]
        h_seq = h.permute(0, 2, 1, 3).contiguous().view(B, T, -1)

        # GRU
        gru_out, _ = self.gru(h_seq)   # [B, T, hidden]

        # project GRU outputs back to hidden * low_bin per time
        projected = self.post_fc(gru_out)  # [B, T, hidden * low_bin]
        # reshape back to [B, hidden, T, low_bin]
        projected = projected.view(B, T, self.hidden, low_bin).permute(0, 2, 1, 3).contiguous()

        # post conv smoothing
        p = self.post_conv(projected)
        p = self.post_bn(p)
        p = self.post_act(p)

        # map back to original channels
        out_low = self.out_proj(p)
        out_low = self.out_act(out_low)  # [-1,1] bounded output

        # combine with high band
        if high is None:
            out = out_low
        else:
            out = torch.cat([out_low, high], dim=-1)

        return out


if __name__ == "__main__":
    # quick test
    B, C, T, F = 2, 2, 128, 513
    x = torch.randn(B, C, T, F)
    pp = PostProcessModule(in_ch=2, hidden=64, sr=48000, n_fft=1024, low_freq=8000)
    y = pp(x)
    print("input:", x.shape, "output:", y.shape)
