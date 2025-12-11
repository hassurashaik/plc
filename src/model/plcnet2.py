# src/model/plcnet2.py
"""
Top-level BS-PLCNet2 generator wiring together:
  - DualPathEncoder (src.model.encoder.DualPathEncoder)
  - FT-GRU bottleneck (src.model.bottleneck.FTGRU)
  - SingleDecoder (src.model.decoder.SingleDecoder)
  - PostProcessModule (src.model.postproc.PostProcessModule) - optional import, fallback provided

Provides:
  - forward(mode='causal'|'noncausal') -> during training you call both modes to compute distillation loss
  - infer(...) wrapper is left to src/infer code

Assumes STFT parameters (for mapping bins) external:
  - n_fft = 1024, hop = 256, window = 1024 (paper default)

Outputs:
  - wideband complex spectrogram parts (real, imag)
  - highband estimate
  - encoder features (causal, noncausal) for distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try imports from the model package
try:
    from src.model.decoder import DualPathEncoder
    from src.model.bottleneck import FTGRU
    from src.model.decoder import SingleDecoder
    # optional external postproc file
    from src.model.postproc import PostProcessModule
except Exception as e:
    # Provide fallback/basic implementations if the modules are not yet present.
    # Encoder fallback: a tiny proxy that projects input to hidden channels
    print("Warning: some model modules not found, using minimal fallbacks:", e)

    class DualPathEncoder(nn.Module):
        def __init__(self, in_ch=1, base_ch=32, n_stages=3, **kw):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, base_ch, kernel_size=1)
            self.out_ch = base_ch

        def forward(self, x):
            h = self.proj(x)
            # return same for causal and noncausal, no skips
            return h, h, [], []

    class FTGRU(nn.Module):
        def __init__(self, in_ch, hidden_ch, **kw):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        def forward(self, x, mode="causal"):
            return x

    class SingleDecoder(nn.Module):
        def __init__(self, in_ch, mid_ch=64, n_layers=3, **kw):
            super().__init__()
            self.head_r = nn.Conv2d(in_ch, 1, kernel_size=1)
            self.head_i = nn.Conv2d(in_ch, 1, kernel_size=1)
            self.head_h = nn.Conv2d(in_ch, 1, kernel_size=1)

        def forward(self, feats, skips=None, mode="causal"):
            return self.head_r(feats), self.head_i(feats), self.head_h(feats)

    class PostProcessModule(nn.Module):
        def __init__(self, in_ch, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, in_ch, 3, padding=1)
            )

        def forward(self, x):
            # x expected [B, 1, T, F] -> merge time+freq for a 1D conv fallback
            B, C, T, F = x.shape
            y = x.view(B, C * F, T)
            y = self.net(y)
            y = y.view(B, C, T, F)
            return y

# Top level generator
class BS_PLCNet2(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        encoder_stages: int = 4,
        tfdcm_layers_per_stage: int = 2,
        bottleneck_hidden: int = 64,
        decoder_mid_ch: int = 64,
        hb_enabled: bool = True,
        postproc_enabled: bool = True,
        n_fft: int = 1024,
        hop_length: int = 256
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Encoder: returns causal & noncausal features
        self.encoder = DualPathEncoder(in_ch=in_ch, base_ch=base_ch, n_stages=encoder_stages, tfdcm_layers_per_stage=tfdcm_layers_per_stage)

        # Bottleneck: F-T-GRU
        self.bottleneck = FTGRU(in_ch=base_ch * (2 ** (encoder_stages - 1)), hidden_ch=bottleneck_hidden)

        # Decoder: convert back to complex spectrogram heads + hb
        self.decoder = SingleDecoder(in_ch=bottleneck_hidden, mid_ch=decoder_mid_ch, n_layers=3, use_tfdcm=True, hb_enabled=hb_enabled)

        # Post processing module applied on WB (0-8k band). If external module available use it else fallback above
        self.postproc_enabled = postproc_enabled
        if postproc_enabled:
            try:
                self.postproc = PostProcessModule(decoder_mid_ch, hidden=decoder_mid_ch)
            except Exception:
                self.postproc = PostProcessModule(bottleneck_hidden, hidden=decoder_mid_ch)
        else:
            self.postproc = nn.Identity()

    def forward(self, x: torch.Tensor, mode: str = "causal"):
        """
        Forward pass.

        Args:
            x: raw input tensor. Expect shape [B, 1, T, F] or [B, T] (waveform). For waveform input, user should
               compute STFT externally and feed TF features. For convenience we accept TF input.
            mode: 'causal' or 'noncausal'
        Returns:
            dict with:
                - wb_real, wb_imag, hb : decoder outputs (tensors)
                - feats_c, feats_nc : encoder features (for distillation)
        """
        # Expect TF features input; if input is 2D waveform, user must convert to TF externally.
        if x.dim() == 2:
            raise ValueError("Input waveform detected; please run STFT externally and provide TF features [B,1,T,F].")

        # Encoder forward -> both causal and noncausal features
        feats_c, feats_nc, skips_c, skips_nc = self.encoder(x)

        # Bottleneck uses causal features (student path). We also allow using causal or noncausal in bottleneck depending on mode
        if mode == "causal":
            bott_in = feats_c
        else:
            # use noncausal features for teacher forward if requested
            bott_in = feats_nc

        bott = self.bottleneck(bott_in, mode=mode)

        # Decoder uses features and causal skips by default (in training you could also feed noncausal skips)
        if mode == "causal":
            wb_real, wb_imag, hb = self.decoder(bott, skips=skips_c, mode="causal")
        else:
            wb_real, wb_imag, hb = self.decoder(bott, skips=skips_nc, mode="noncausal")

        # Post processing (applied on wideband complex pair by merging channels)
        # Stack real & imag into 2-channel representation for postproc if needed
        complex_wb = torch.cat([wb_real, wb_imag], dim=1)  # [B, 2, T, F]
        post_out = self.postproc(complex_wb) if self.postproc_enabled else complex_wb

        # split back (if postproc returns 2 channels)
        if post_out.shape[1] >= 2:
            wb_real_p = post_out[:, 0:1, :, :]
            wb_imag_p = post_out[:, 1:2, :, :]
        else:
            # fallback if postproc returns same shape
            wb_real_p, wb_imag_p = wb_real, wb_imag

        return {
            "wb_real": wb_real_p,
            "wb_imag": wb_imag_p,
            "hb": hb,
            "feats_c": feats_c,
            "feats_nc": feats_nc
        }

    def inference_simplified(self, complex_wb_masked, hb_masked, stft_helper):
        """
        Optional simplified inference helper that accepts complex WB representation and HB, then does ISTFT.
        stft_helper should provide istft and merge band helpers. This function is intentionally simple and
        left to the inference module to implement fully (see src/infer/inference.py).
        """
        raise NotImplementedError("Use src/infer/inference.py for complete real-time inference pipeline")

if __name__ == "__main__":
    # Quick shape sanity test using TF-like inputs
    B, T, F = 2, 256, 513  # for n_fft=1024, rfft outputs 513 freq bins
    x = torch.randn(B, 1, T, F)
    model = BS_PLCNet2(in_ch=1, base_ch=32, encoder_stages=3, tfdcm_layers_per_stage=2, bottleneck_hidden=64, decoder_mid_ch=64)
    out = model(x, mode="noncausal")
    print("wb_real", out["wb_real"].shape, "wb_imag", out["wb_imag"].shape, "hb", out["hb"].shape)
    print("feats_c", out["feats_c"].shape, "feats_nc", out["feats_nc"].shape)
