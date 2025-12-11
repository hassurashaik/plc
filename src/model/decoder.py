# src/model/decoder.py
"""
Single Decoder module for BS-PLCNet2:
 - Decodes bottleneck features into wide-band complex spectrogram (real + imag)
 - Provides a high-band head for reconstructing high-frequency content
 - Supports causal and noncausal modes (for dual-path convs)

Expect input shapes:
    feats: [B, C, T, F]  (bottleneck output / encoder last features)
    skips: optional list of skip tensors from encoder: each [B, C_i, T, F]

Outputs:
    wb_real: [B, 1, T, F_wb]
    wb_imag: [B, 1, T, F_wb]
    hb: [B, 1, T, F_hb]   (if computed; else zeros)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import DualPathDepthwiseConv, TFDCM, DepthwiseSeparableConv2d


class SingleDecoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int = 64,
        n_layers: int = 3,
        use_tfdcm: bool = True,
        hb_enabled: bool = True,
        hb_ratio: float = 1.0
    ):
        """
        Args:
            in_ch: number of channels in bottleneck input
            mid_ch: internal channel width
            n_layers: number of decoder stages
            use_tfdcm: whether to use TFDCM blocks in decoder
            hb_enabled: whether to produce a high-band head output
            hb_ratio: if hb_enabled, relative freq bins for high-band head (not enforced here)
        """
        super().__init__()
        self.n_layers = n_layers
        self.hb_enabled = hb_enabled

        # initial projection from bottleneck channels to mid channels
        self.init_proj = nn.Conv2d(in_ch, mid_ch, kernel_size=1)

        # decoder stacks: DualPathDepthwiseConv (noncausal/causal switch in forward)
        self.dec_blocks = nn.ModuleList()
        self.tfdcms = nn.ModuleList()
        for i in range(n_layers):
            self.dec_blocks.append(DualPathDepthwiseConv(mid_ch, mid_ch, kernel_size=(3, 3), dilation=(1, 1)))
            if use_tfdcm:
                # small TFDCM at each stage
                self.tfdcms.append(TFDCM(ch=mid_ch, n_layers=2))
            else:
                self.tfdcms.append(nn.Identity())

        # optional skip projection if skip channels differ
        self.skip_projs = nn.ModuleList()
        for i in range(n_layers):
            # project any incoming skip to mid_ch
            self.skip_projs.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=1))

        # final refine convs
        self.refine = nn.Sequential(
            DepthwiseSeparableConv2d(mid_ch, mid_ch, kernel_size=(3, 3)),
            nn.ELU(),
            nn.BatchNorm2d(mid_ch)
        )

        # heads for wideband real & imag
        self.wb_real_head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(mid_ch // 2, 1, kernel_size=1)
        )
        self.wb_imag_head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(mid_ch // 2, 1, kernel_size=1)
        )

        # high-band head (optional)
        if self.hb_enabled:
            self.hb_head = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1),
                nn.ELU(),
                nn.Conv2d(mid_ch // 2, 1, kernel_size=1)
            )
        else:
            self.hb_head = None

        # small output normalization
        self.out_norm = nn.BatchNorm2d(1)

    def forward(
        self,
        feats: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        mode: str = "causal"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: [B, C, T, F]
            skips: list of skip tensors (len == n_layers) each [B, C_skip, T, F] (optional)
            mode: 'causal' or 'noncausal' - passed to DualPathDepthwiseConv layers
        Returns:
            wb_real, wb_imag, hb  (hb may be zeros if hb_enabled==False)
        """
        h = self.init_proj(feats)  # [B, mid_ch, T, F]

        # iterate decoder stacks. If skips provided, combine via addition (after projecting)
        for i in range(self.n_layers):
            # dual-path conv with chosen mode
            h = self.dec_blocks[i](h, mode=mode)  # internal padding handled inside block
            # combine skip if available
            if skips is not None and i < len(skips):
                s = skips[i]
                # if skip channel differs, project (assume skip already same ch or use identity)
                if s.shape[1] != h.shape[1]:
                    # project skip to mid_ch
                    s = F.interpolate(s, size=(h.shape[2], h.shape[3]), mode='nearest') if s.shape[2:] != h.shape[2:] else s
                    s = self.skip_projs[i](s)
                h = h + s
            # optional TFDCM refinement (choose causal vs noncausal inside TFDCM)
            tfdcm = self.tfdcms[i]
            # TFDCM in our blocks expects mode argument (we used same TFDCM class)
            if isinstance(tfdcm, TFDCM):
                h = tfdcm(h, mode=mode)
            else:
                h = tfdcm(h)

        # refinement conv
        h = self.refine(h)

        # heads
        wb_real = self.wb_real_head(h)
        wb_imag = self.wb_imag_head(h)

        if self.hb_enabled and (self.hb_head is not None):
            hb = self.hb_head(h)
        else:
            hb = torch.zeros_like(wb_real)

        # normalize outputs (optional)
        wb_real = self.out_norm(wb_real)
        wb_imag = self.out_norm(wb_imag)
        hb = self.out_norm(hb)

        return wb_real, wb_imag, hb


if __name__ == "__main__":
    # quick test
    B, C, T, F = 2, 64, 128, 64
    feats = torch.randn(B, C, T, F)
    # create some dummy skips with same mid channels
    skips = [torch.randn(B, 64, T, F) for _ in range(3)]
    dec = SingleDecoder(in_ch=C, mid_ch=64, n_layers=3, use_tfdcm=True, hb_enabled=True)
    wb_r, wb_i, hb = dec(feats, skips=skips, mode="noncausal")
    print("wb_r", wb_r.shape, "wb_i", wb_i.shape, "hb", hb.shape)
