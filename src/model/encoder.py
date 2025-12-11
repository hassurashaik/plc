# src/model/encoder.py
"""
DualPathEncoder for BS-PLCNet2.

This encoder creates two separate convolutional branches per stage:
 - causal branch (left-time padding only)
 - noncausal branch (symmetric padding)

It returns:
 - feats_c: final causal features [B, C, T, F]
 - feats_nc: final noncausal features [B, C, T, F]
 - skips_c: list of causal skip tensors (per stage)
 - skips_nc: list of noncausal skip tensors (per stage)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import DualPathDepthwiseConv, TFDCM


class DualPathEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        n_stages: int = 3,
        tfdcm_layers_per_stage: int = 2,
    ):
        """
        Args:
            in_ch: input channels (1 for magnitude, or 2 for complex stacked)
            base_ch: channels for stage 0
            n_stages: number of encoder stages
            tfdcm_layers_per_stage: number of TFDCM layers per stage
        """
        super().__init__()
        self.n_stages = n_stages
        self.base_ch = base_ch

        # initial projection
        self.initial_proj = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        # create per-stage modules for causal and noncausal branches
        self.causal_blocks = nn.ModuleList()
        self.noncausal_blocks = nn.ModuleList()
        self.tfdcm_causal = nn.ModuleList()
        self.tfdcm_noncausal = nn.ModuleList()
        self.proj_layers = nn.ModuleList()  # project to expected channel for each stage if needed

        for i in range(n_stages):
            ch = base_ch * (2 ** i)

            # For the first stage, input channel equals base_ch after initial_proj.
            # For deeper stages we keep channel width = ch and optionally project.
            self.proj_layers.append(nn.Conv2d(ch, ch, kernel_size=1))

            self.causal_blocks.append(
                DualPathDepthwiseConv(in_ch=ch, out_ch=ch, kernel_size=(3, 3), dilation=(1, 1))
            )
            self.noncausal_blocks.append(
                DualPathDepthwiseConv(in_ch=ch, out_ch=ch, kernel_size=(3, 3), dilation=(1, 1))
            )

            # TFDCM modules (use same config but separate instances for causal/noncausal)
            self.tfdcm_causal.append(TFDCM(ch=ch, n_layers=tfdcm_layers_per_stage))
            self.tfdcm_noncausal.append(TFDCM(ch=ch, n_layers=tfdcm_layers_per_stage))

        # final projection (identity-like)
        final_ch = base_ch * (2 ** (n_stages - 1))
        self.output_proj = nn.Conv2d(final_ch, final_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        x: [B, in_ch, T, F]
        returns (feats_c, feats_nc, skips_c, skips_nc)
        """
        # initial projection
        h_init = self.initial_proj(x)  # [B, base_ch, T, F]

        h_c = h_init
        h_nc = h_init

        skips_c = []
        skips_nc = []

        # iterate stages
        for i in range(self.n_stages):
            expected_ch = self.base_ch * (2 ** i)

            # If current channel dimension doesn't match expected (shouldn't often), project
            if h_c.shape[1] != expected_ch:
                proj = self.proj_layers[i].to(h_c.device)
                h_c = proj(h_c)
                h_nc = proj(h_nc)

            # dual-path depthwise convs (each branch pads appropriately inside)
            h_c = self.causal_blocks[i](h_c, mode="causal")
            h_nc = self.noncausal_blocks[i](h_nc, mode="noncausal")

            # collect skips before TFDCM
            skips_c.append(h_c)
            skips_nc.append(h_nc)

            # TFDCM refinement
            h_c = self.tfdcm_causal[i](h_c, mode="causal")
            h_nc = self.tfdcm_noncausal[i](h_nc, mode="noncausal")

        feats_c = self.output_proj(h_c)
        feats_nc = self.output_proj(h_nc)

        return feats_c, feats_nc, skips_c, skips_nc


# self-test when executed directly
if __name__ == "__main__":
    B, T, F = 2, 64, 257
    x = torch.randn(B, 1, T, F)
    enc = DualPathEncoder(in_ch=1, base_ch=16, n_stages=3, tfdcm_layers_per_stage=1)
    feats_c, feats_nc, skips_c, skips_nc = enc(x)
    print("feats_c", feats_c.shape, "feats_nc", feats_nc.shape)
    print("n skips:", len(skips_c), [s.shape for s in skips_c])
