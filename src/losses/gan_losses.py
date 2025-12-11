# src/losses/gan_losses.py
"""
LSGAN losses + simple MPD and MFD discriminators.

This file provides:
 - lsgan_d_loss(real_pred, fake_pred)
 - lsgan_g_loss(fake_pred)
 - simple PeriodDiscriminator and MFD (2D conv) implementations
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# LSGAN losses
# -------------------------
def lsgan_discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor):
    """
    Least Squares discriminator loss:
        L_D = (D(x) - 1)^2 + (D(G(z)))^2
    real_pred: discriminator outputs on real data
    fake_pred: outputs on fake data
    """
    loss_real = torch.mean((real_pred - 1.0) ** 2)
    loss_fake = torch.mean((fake_pred) ** 2)
    return 0.5 * (loss_real + loss_fake)


def lsgan_generator_loss(fake_pred: torch.Tensor):
    """
    Generator loss: (D(G(z)) - 1)^2
    """
    return 0.5 * torch.mean((fake_pred - 1.0) ** 2)


# -------------------------
# Simple Period Discriminator (MPD)
# -------------------------
class SubPeriodDiscriminator(nn.Module):
    """
    A single period discriminator that reshapes waveform into (B, 1, L//period, period)
    and applies 2D convs.
    """
    def __init__(self, period: int, n_layers: int = 4, base_ch: int = 32):
        super().__init__()
        self.period = period
        layers = []
        in_ch = 1
        ch = base_ch
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch
            ch = min(ch * 2, 512)
        # final conv to scalar map
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: waveform [B, 1, L]
        B, C, L = x.shape
        p = self.period
        if L % p != 0:
            # pad to multiple of period
            pad = p - (L % p)
            x = F.pad(x, (0, pad), "reflect")
            L = L + pad
        x = x.view(B, 1, L // p, p)  # reshape
        out = self.net(x)
        return out


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.subs = nn.ModuleList([SubPeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        """
        x: waveform [B, 1, L]
        returns list of outputs per sub-discriminator
        """
        outs = []
        for sub in self.subs:
            outs.append(sub(x))
        return outs


# -------------------------
# Simple Multi-Frequency Discriminator (MFD)
# -------------------------
class SimpleMFD(nn.Module):
    """
    Applied on spectrograms: expects [B, 2, T, F] (real+imag) or [B, 1, T, F] (magnitude)
    A few 2D conv layers produce a map; used as discriminator.
    """
    def __init__(self, in_ch: int = 2, n_layers: int = 4, base_ch: int = 32):
        super().__init__()
        layers = []
        ch = base_ch
        in_channels = in_ch
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_channels, ch, kernel_size=(3, 3), padding=(1, 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = ch
            ch = min(ch * 2, 512)
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, spec):
        # spec: [B, C, T, F]
        out = self.net(spec)
        return out


# -------------------------
# Convenience wrapper functions
# -------------------------
def discriminator_forward_and_loss(discriminator: nn.Module, real, fake, loss_fn=lsgan_discriminator_loss):
    """
    Run discriminator on real and fake, compute loss.
    Returns: loss, real_pred, fake_pred
    """
    real_pred = discriminator(real)
    fake_pred = discriminator(fake.detach())
    loss = loss_fn(real_pred, fake_pred)
    return loss, real_pred, fake_pred


def generator_adv_loss(discriminator: nn.Module, fake, loss_fn=lsgan_generator_loss):
    """
    Run discriminator on fake (no detach) and compute generator adversarial loss.
    """
    fake_pred = discriminator(fake)
    return loss_fn(fake_pred)


if __name__ == "__main__":
    # quick smoke test
    B, L = 2, 16384
    wav = torch.randn(B, 1, L)
    mpd = MultiPeriodDiscriminator()
    outs = mpd(wav)
    print("MPD outputs count:", len(outs))
    # test MFD
    spec = torch.randn(B, 2, 128, 513)
    mfd = SimpleMFD(in_ch=2)
    print("MFD out:", mfd(spec).shape)
