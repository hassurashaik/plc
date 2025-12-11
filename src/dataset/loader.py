# src/dataset/loader.py
import os
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

class PLCDataset(Dataset):
    """
    Dataset that returns fixed-length (chunk_seconds) numpy arrays for 'clean' and 'lossy' pairs.
    Expects files named like: <stem>_clean.wav and <stem>_lossy.wav in the same folder or provided list.
    """
    def __init__(self, wav_list, sr=48000, chunk_seconds=2):
        """
        wav_list: flattened list [clean1, lossy1, clean2, lossy2, ...]
        sr: target sample rate (must match files or librosa available for resampling)
        chunk_seconds: number of seconds per sample returned
        """
        assert len(wav_list) % 2 == 0, "wav_list must contain pairs of clean/lossy files"
        self.pairs = []
        for i in range(0, len(wav_list), 2):
            clean = Path(wav_list[i])
            lossy = Path(wav_list[i+1])
            if clean.exists() and lossy.exists():
                self.pairs.append((str(clean), str(lossy)))
        if len(self.pairs) == 0:
            raise RuntimeError("No valid clean/lossy pairs found in wav_list")

        self.sr = sr
        self.chunk_seconds = chunk_seconds
        self.chunk_len = int(sr * chunk_seconds)

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        # returns mono numpy float32 at target sample rate
        audio, fs = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            # to mono
            audio = np.mean(audio, axis=1)
        if fs != self.sr:
            if _HAS_LIBROSA:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=fs, target_sr=self.sr)
            else:
                raise RuntimeError(f"Sample rate mismatch for {path}: file {fs} != target {self.sr}. Install librosa or pre-resample.")
        # ensure float32 in [-1,1]
        audio = audio.astype(np.float32)
        # if amplitude large, normalize (guard)
        maxv = np.max(np.abs(audio)) if audio.size>0 else 1.0
        if maxv > 1.0:
            audio = audio / maxv
        return audio

    def _pad_or_crop(self, audio):
        L = len(audio)
        if L == self.chunk_len:
            return audio
        elif L > self.chunk_len:
            # random crop to chunk_len for training diversity
            start = np.random.randint(0, L - self.chunk_len + 1)
            return audio[start:start + self.chunk_len]
        else:
            # pad with zeros at the end
            pad = self.chunk_len - L
            return np.pad(audio, (0, pad), mode='constant', constant_values=0.0)

    def __getitem__(self, idx):
        clean_path, lossy_path = self.pairs[idx]
        clean = self._load(clean_path)
        lossy = self._load(lossy_path)

        clean = self._pad_or_crop(clean)
        lossy = self._pad_or_crop(lossy)

        # return consistent-length numpy arrays
        return {"clean": clean, "lossy": lossy}
