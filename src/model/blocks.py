# src/model/blocks.py
"""
Core convolution blocks for BS-PLCNet2.
Implements:
    - DepthwiseSeparableConv2d
    - GatedConv2d
    - DualPathDepthwiseConv (causal + non-causal)
    - TFDCM: Time-Frequency Dilated Conv Module

Shapes: [B, C, T, F]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ---------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------

def pad_tf(x, kernel_t, kernel_f, dilation_t, dilation_f, causal=False):
    """
    Pads time and frequency dimensions correctly so Conv2D keeps shape.
    Time:
        causal: pad only left side
        noncausal: symmetric padding
    Frequency:
        always symmetric padding
    """
    pad_t = (kernel_t - 1) * dilation_t
    pad_f = (kernel_f - 1) * dilation_f

    # Frequency padding (symmetric)
    pad_f_left = pad_f // 2
    pad_f_right = pad_f - pad_f_left

    if causal:
        pad_t_left = pad_t
        pad_t_right = 0
    else:
        pad_t_left = pad_t // 2
        pad_t_right = pad_t - pad_t_left

    # F.pad format: (f_left, f_right, t_left, t_right)
    return F.pad(x, (pad_f_left, pad_f_right, pad_t_left, pad_t_right))


# ---------------------------------------------------------
# Depthwise separable 2D convolution
# ---------------------------------------------------------

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), dilation=(1, 1), bias=True):
        super().__init__()
        k_t, k_f = kernel_size
        d_t, d_f = dilation

        self.depth = nn.Conv2d(
            in_ch, in_ch, 
            kernel_size=(k_t, k_f),
            dilation=(d_t, d_f),
            groups=in_ch,
            bias=False
        )

        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x


# ---------------------------------------------------------
# Gated convolution module
# ---------------------------------------------------------

class GatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), dilation=(1,1)):
        super().__init__()
        self.f = DepthwiseSeparableConv2d(in_ch, out_ch, kernel_size, dilation)
        self.g = DepthwiseSeparableConv2d(in_ch, out_ch, kernel_size, dilation)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(torch.tanh(self.f(x)) * torch.sigmoid(self.g(x)))


# ---------------------------------------------------------
# Dual-path depthwise conv (Causal + Non-Causal branches)
# ---------------------------------------------------------

class DualPathDepthwiseConv(nn.Module):
    """
    Two completely separate branches:
        - causal DP conv
        - noncausal DP conv
    No weight sharing.
    """

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), dilation=(1,1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.causal = GatedConv2d(in_ch, out_ch, kernel_size, dilation)
        self.noncausal = GatedConv2d(in_ch, out_ch, kernel_size, dilation)

    def forward(self, x, mode):
        k_t, k_f = self.kernel_size
        d_t, d_f = self.dilation

        if mode == "causal":
            x_pad = pad_tf(x, k_t, k_f, d_t, d_f, causal=True)
            return self.causal(x_pad)
        else:
            x_pad = pad_tf(x, k_t, k_f, d_t, d_f, causal=False)
            return self.noncausal(x_pad)


# ---------------------------------------------------------
# TFDCM - Time-Frequency Dilated Convolution Module
# ---------------------------------------------------------

class TFDCM(nn.Module):
    """
    Stacked dilated depthwise convs in the TIME dimension.
    """

    def __init__(self, ch, n_layers=4, base_dilation=1):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            dilation_t = base_dilation * (2 ** i)
            self.layers.append(
                DualPathDepthwiseConv(ch, ch, kernel_size=(3, 3), dilation=(dilation_t, 1))
            )

        self.out_norm = nn.BatchNorm2d(ch)
        self.act = nn.ELU()

    def forward(self, x, mode):
        """
        Applies dual-path dilated conv stacks with residuals.
        """
        h = x
        for layer in self.layers:
            out = layer(h, mode)
            h = h + out
            h = self.act(h)

        return self.out_norm(h)
