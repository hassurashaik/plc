# src/train/train_stage1.py
"""
Stage-1 training script for BS-PLCNet2.

Features:
 - Loads PLC dataset pairs (clean / lossy) from a directory (see src/dataset/loader.py)
 - Computes STFT (torch) to obtain TF inputs
 - Feeds magnitude TF inputs to generator (BS_PLCNet2)
 - Computes PLPCA loss on predicted complex WB spectrogram vs clean spectrogram
 - Computes waveform MAE (ISTFT) between reconstructed waveform and clean wave
 - Adds distillation loss between encoder causal and noncausal features
 - Trains MPD and MFD discriminators (LSGAN) and applies adversarial loss to generator
 - Saves checkpoints periodically

Usage:
    python -m src.train.train_stage1 --data-dir data/plc_train --epochs 10 --batch-size 4 --save-dir experiments/run_001
"""

import os
import time
import argparse
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local imports (assumes running from project root)
from src.dataset.loader import PLCDataset
from src.model.plcnet2 import BS_PLCNet2
from src.losses.plpca import plpca_loss, mae_time_loss
from src.losses.gan_losses import MultiPeriodDiscriminator, SimpleMFD, lsgan_discriminator_loss, lsgan_generator_loss

# -----------------------
# STFT helpers (torch)
# -----------------------
def compute_stft_torch(wave: torch.Tensor, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, window=None, return_complex=True):
    """
    wave: [B, T] or [B, 1, T]
    returns: complex spectrogram [B, F, T] (when return_complex True) or (real, imag)
    We'll return torch.complex64 tensor shaped [B, T_frames, F_bins] (we'll transpose later)
    """
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)  # [B,1,T]
    if window is None:
        window = torch.hann_window(win_length).to(wave.device)
    # torch.stft returns [B, C, F, T] when return_complex False, but with return_complex True it returns complex tensor [B, C, F, T]
    spec = torch.stft(
        wave.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True, normalized=False, center=True
    )  # [B, F, T]
    return spec  # complex tensor [B, F, T]

def spec_to_realimag(spec_complex: torch.Tensor):
    # spec_complex: [B, F, T] complex
    real = spec_complex.real
    imag = spec_complex.imag
    # convert to [B, 2, T, F] expected by our model outputs (channels, time, freq)
    # reorder to [B, 2, T, F]
    real_t = real.permute(0, 2, 1).unsqueeze(1)  # [B,1,T,F]
    imag_t = imag.permute(0, 2, 1).unsqueeze(1)
    stacked = torch.cat([real_t, imag_t], dim=1)  # [B,2,T,F]
    return stacked

def spec_magnitude(spec_complex: torch.Tensor):
    mag = torch.abs(spec_complex)
    # return [B, 1, T, F]
    return mag.permute(0, 2, 1).unsqueeze(1)

def istft_from_complex(spec_complex: torch.Tensor, n_fft=1024, hop_length=256, win_length=1024, window=None, length=None):
    # spec_complex: [B, F, T] complex
    if window is None:
        window = torch.hann_window(win_length).to(spec_complex.device)
    # torch.istft expects [B, F, T] complex
    wave = torch.istft(spec_complex, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)
    # returns [B, T]
    return wave

# -----------------------
# Training function
# -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    # collect wav pairs
    wavs = sorted(glob(os.path.join(args.data_dir, "*_clean.wav")))
    # for each clean derive lossy path with same stem
    pairs = []
    for c in wavs:
        stem = Path(c).stem.replace("_clean", "")
        lossy = os.path.join(args.data_dir, f"{stem}_lossy.wav")
        if os.path.exists(lossy):
            pairs.append((c, lossy))
    if len(pairs) == 0:
        raise RuntimeError("No clean/lossy pairs found in data_dir; expected *_clean.wav and *_lossy.wav")

    # flatten to list suitable for our PLCDataset loader which expects [clean, lossy] ordering
    wav_list = []
    for c, l in pairs:
        wav_list.append(c)
        wav_list.append(l)

    dataset = PLCDataset(wav_list, sr=args.sr, chunk_seconds=args.chunk_seconds)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # model
    model = BS_PLCNet2(in_ch=1, base_ch=args.base_ch, encoder_stages=args.encoder_stages,
                  tfdcm_layers_per_stage=args.tfdcm_layers, bottleneck_hidden=args.bottleneck_hidden,
                  decoder_mid_ch=args.decoder_mid_ch, hb_enabled=True, postproc_enabled=False,
                  n_fft=args.n_fft, hop_length=args.hop)

    model = model.to(device)

    # discriminators
    mpd = MultiPeriodDiscriminator().to(device)
    mfd = SimpleMFD(in_ch=2).to(device)

    # optimizers
    g_params = list(model.parameters())
    d_params = list(mpd.parameters()) + list(mfd.parameters())
    optim_g = optim.Adam(g_params, lr=args.lr, betas=(0.9, 0.95))
    optim_d = optim.Adam(d_params, lr=args.lr_d, betas=(0.9, 0.95))

    # windows for stft/istft
    window = torch.hann_window(args.win_length).to(device)

    global_step = 0
    best_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # batch has numpy arrays
            # Accept either NumPy arrays or already-converted torch tensors from the DataLoader
            clean_item = batch["clean"]
            lossy_item = batch["lossy"]

# If DataLoader already converted to torch.Tensor, use it directly; otherwise convert
            if isinstance(clean_item, torch.Tensor):
                clean_wav = clean_item.to(device)
            else:
                clean_wav = torch.from_numpy(np.asarray(clean_item)).to(device)

            if isinstance(lossy_item, torch.Tensor):
                lossy_wav = lossy_item.to(device)
            else:
                lossy_wav = torch.from_numpy(np.asarray(lossy_item)).to(device)


            # compute STFTs (complex) [B, F, T]
            spec_clean = compute_stft_torch(clean_wav, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length, window=window)
            spec_lossy = compute_stft_torch(lossy_wav, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length, window=window)

            # compute magnitude inputs (B,1,T,F)
            mag_lossy = spec_magnitude(spec_lossy)  # [B,1,T,F]
            mag_clean = spec_magnitude(spec_clean)

            # model forward: causal student output (we train causal generator)
            out_causal = model(mag_lossy, mode="causal")
            wb_real = out_causal["wb_real"]   # [B,1,T,F]
            wb_imag = out_causal["wb_imag"]
            feats_c = out_causal["feats_c"]
            # teacher (noncausal) features for distillation
            with torch.no_grad():
                out_nc = model(mag_lossy, mode="noncausal")
                feats_nc = out_nc["feats_nc"]

            # target complex spectrogram for training (shape match)
            target_spec = spec_to_realimag_for_loss(spec_clean) if 'spec_to_realimag_for_loss' in globals() else None
            # create target  [B,2,T,F]
            tgt_real = spec_clean.real.permute(0, 2, 1).unsqueeze(1)  # [B,1,T,F]
            tgt_imag = spec_clean.imag.permute(0, 2, 1).unsqueeze(1)
            target_spec = torch.cat([tgt_real, tgt_imag], dim=1)

            # compute PLCPA loss (mag+phase)
            loss_plpca = plpca_loss(torch.cat([wb_real, wb_imag], dim=1), target_spec)

            # reconstruct waveform from predicted complex spec
            # build complex tensor [B, F, T]
            pred_complex = torch.complex(wb_real.squeeze(1).permute(0, 2, 1), wb_imag.squeeze(1).permute(0, 2, 1))
            rec_wav = istft_from_complex(pred_complex, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length, window=window, length=clean_wav.shape[1])
            loss_mae_time = mae_time_loss(rec_wav, clean_wav)

            # distillation loss between feats (MSE)
            # feats_c and feats_nc shapes should match [B, C, T, F]
            distill = nn.MSELoss()(feats_c, feats_nc)

            # Generator adversarial loss: compute discriminator outputs on waveform and spec
            # Prepare real waveform and fake waveform (for MPD)
            real_wave = clean_wav.unsqueeze(1)  # [B,1,T]
            fake_wave = rec_wav.unsqueeze(1).detach()  # detach for D update

            # Prepare real and fake specs for MFD (use target_spec and pred)
            real_spec = target_spec  # [B,2,T,F]
            fake_spec = torch.cat([wb_real.detach(), wb_imag.detach()], dim=1)

            # ---------------------
            # Update Discriminators
            # ---------------------
            optim_d.zero_grad()

            # MPD - runs on waveform
            d_real_mpd = mpd(real_wave)
            d_fake_mpd = mpd(fake_wave)
            # compute scalar losses by averaging outputs maps
            loss_d_mpd = 0.0
            for rr, ff in zip(d_real_mpd, d_fake_mpd):
                loss_d_mpd = loss_d_mpd + lsgan_discriminator_loss(rr, ff)
            loss_d_mpd = loss_d_mpd / len(d_real_mpd)

            # MFD - runs on spectrograms
            pred_spec_for_d = torch.cat([wb_real.detach(), wb_imag.detach()], dim=1)
            d_real_mfd = mfd(real_spec)
            d_fake_mfd = mfd(pred_spec_for_d)
            loss_d_mfd = lsgan_discriminator_loss(d_real_mfd, d_fake_mfd)

            loss_d = loss_d_mpd + loss_d_mfd
            loss_d.backward()
            optim_d.step()

            # ---------------------
            # Update Generator
            # ---------------------
            optim_g.zero_grad()

            # Recompute rec_wav without detach for gradient to flow
            pred_complex = torch.complex(wb_real.squeeze(1).permute(0, 2, 1), wb_imag.squeeze(1).permute(0, 2, 1))
            rec_wav = istft_from_complex(pred_complex, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length, window=window, length=clean_wav.shape[1])

            # Adversarial loss (generator) - MPD & MFD
            g_fake_mpd = mpd(rec_wav.unsqueeze(1))
            loss_g_mpd = 0.0
            for out in g_fake_mpd:
                loss_g_mpd = loss_g_mpd + lsgan_generator_loss(out)
            loss_g_mpd = loss_g_mpd / len(g_fake_mpd)

            g_fake_mfd = mfd(torch.cat([wb_real, wb_imag], dim=1))
            loss_g_mfd = lsgan_generator_loss(g_fake_mfd)

            # Total generator loss
            loss_g = loss_plpca * args.lambda_plpca + loss_mae_time * args.lambda_mae + distill * args.lambda_distill + loss_g_mpd * args.lambda_adv + loss_g_mfd * args.lambda_adv

            loss_g.backward()
            optim_g.step()

            # logging
            running_loss += loss_g.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = running_loss / max(1, args.log_interval)
                print(f"Epoch {epoch+1}/{args.epochs} Step {global_step} | loss_g {avg:.4f} (plpca {loss_plpca.item():.4f} mae {loss_mae_time.item():.4f} distill {distill.item():.6f} adv {loss_g_mpd.item():.4f})")
                running_loss = 0.0

            # save checkpoint
            if global_step % args.save_interval == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "mpd": mpd.state_dict(),
                    "mfd": mfd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "step": global_step,
                    "epoch": epoch
                }
                torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_step{global_step}.pt"))
                print("Saved checkpoint at step", global_step)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

    # final save
    torch.save({"model": model.state_dict()}, os.path.join(args.save_dir, "final_model.pth"))
    print("Training complete. Model saved to", args.save_dir)


# -----------------------
# Utility wrapper for target_spec conversion (if wanted)
# -----------------------
def spec_to_realimag_for_loss(spec_complex: torch.Tensor):
    # spec_complex: [B, F, T] complex
    real = spec_complex.real.permute(0, 2, 1).unsqueeze(1)
    imag = spec_complex.imag.permute(0, 2, 1).unsqueeze(1)
    return torch.cat([real, imag], dim=1)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/plc_train", help="Directory containing *_clean.wav and *_lossy.wav pairs")
    parser.add_argument("--save-dir", type=str, default="experiments/run_001", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--chunk-seconds", type=int, default=8)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--encoder-stages", type=int, default=3)
    parser.add_argument("--tfdcm-layers", type=int, default=2)
    parser.add_argument("--bottleneck-hidden", type=int, default=64)
    parser.add_argument("--decoder-mid-ch", type=int, default=64)
    parser.add_argument("--lambda-plpca", type=float, default=1.0)
    parser.add_argument("--lambda-mae", type=float, default=1.0)
    parser.add_argument("--lambda-distill", type=float, default=0.1)
    parser.add_argument("--lambda-adv", type=float, default=0.5)
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true")
    args = parser.parse_args()

    # small sanity of directories
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
