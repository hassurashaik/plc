import os
import random
import soundfile as sf
import numpy as np
from pathlib import Path

def gilbert_elliott(len_frames, p, q):
    """Generate loss flags (0=ok, 1=lost)"""
    flags = []
    state = 0
    for _ in range(len_frames):
        if state == 0:
            state = 1 if random.random() < p else 0
        else:
            state = 0 if random.random() < q else 1
        flags.append(state)
    return flags


def apply_packet_loss(audio, frame_size=960):
    """Zero-out lost frames"""
    num_frames = len(audio) // frame_size
    flags = gilbert_elliott(num_frames, p=0.3, q=0.3)

    out = audio.copy()
    for i, f in enumerate(flags):
        if f == 1:
            out[i*frame_size : (i+1)*frame_size] = 0.0
    return out


def generate_dataset(clean_dir="data/clean", out_dir="data/plc_train", sr=48000):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    wavs = list(clean_dir.rglob("*.wav"))
    print("Found:", len(wavs), "clean files.")

    for w in wavs:
        audio, sr = sf.read(w)
        lossy = apply_packet_loss(audio)

        name = w.stem
        sf.write(out_dir / f"{name}_clean.wav", audio, sr)
        sf.write(out_dir / f"{name}_lossy.wav", lossy, sr)

    print("Packet-loss dataset ready.")


if __name__ == "__main__":
    generate_dataset()
