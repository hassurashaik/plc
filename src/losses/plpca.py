# src/losses/plpca.py
"""
PLCPA (Power-Law Compressed Phase-Aware) loss and related helpers.

Functions:
  - plpca_loss(pred_complex, target_complex, power=0.3): returns L_mag + L_phase
  - mae_time_loss(pred_wave, target_wave): L1 in time domain
  - f0_mae_loss(pred_f0, target_f0): L1 over frames
"""

import torch
import torch.nn.functional as F
import math


def _complex_components(spec: torch.Tensor):
    """
    Accepts either:
      - real/imag concatenated channels shape [B, 2, T, F], or
      - complex dtype (rare)
    Returns real, imag
    """
    if spec.is_complex():
        real = spec.real
        imag = spec.imag
    else:
        # assume channel-first real+imag
        if spec.shape[1] == 2:
            real = spec[:, 0, :, :]
            imag = spec[:, 1, :, :]
        else:
            raise ValueError("Expected complex-like input with 2 channels (real, imag).")
    return real, imag


def plpca_loss(pred_spec: torch.Tensor, target_spec: torch.Tensor, power: float = 0.3, eps: float = 1e-8):
    """
    Power-law compressed magnitude + simple phase-aware loss.

    pred_spec & target_spec: [B, 2, T, F] (real, imag)
    power: exponent for power-law compression (0.3 is common)
    """
    pred_r, pred_i = _complex_components(pred_spec)
    tgt_r, tgt_i = _complex_components(target_spec)

    # magnitudes
    pred_mag = torch.sqrt(pred_r ** 2 + pred_i ** 2 + eps)
    tgt_mag = torch.sqrt(tgt_r ** 2 + tgt_i ** 2 + eps)

    # power-law compression
    comp_pred = torch.pow(pred_mag + eps, power)
    comp_tgt = torch.pow(tgt_mag + eps, power)
    lmag = F.l1_loss(comp_pred, comp_tgt)

    # phase loss - encourage sin of phase diff to be small (bounded)
    pred_phase = torch.atan2(pred_i, pred_r)
    tgt_phase = torch.atan2(tgt_i, tgt_r)
    phase_diff = pred_phase - tgt_phase
    # wrap to [-pi, pi]
    phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi
    lphase = torch.mean(torch.abs(torch.sin(phase_diff)))

    return lmag + lphase


def mae_time_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor):
    """
    Simple time-domain MAE (L1) loss between predicted and target waveforms.
    Inputs: [B, T] or [B, 1, T]
    """
    if pred_wave.dim() == 3:
        pred_wave = pred_wave.squeeze(1)
    if target_wave.dim() == 3:
        target_wave = target_wave.squeeze(1)
    return torch.mean(torch.abs(pred_wave - target_wave))


def f0_mae_loss(pred_f0: torch.Tensor, target_f0: torch.Tensor):
    """
    L1 loss for f0 sequence: inputs [B, T_f0]
    """
    return torch.mean(torch.abs(pred_f0 - target_f0))
