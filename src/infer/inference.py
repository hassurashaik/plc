# src/infer/inference.py
import torch
import soundfile as sf
from src.model.simple_model import SimplePLCNet

def restore(degraded_path, ckpt="models/simple_stage1.pth", out="out_restored.wav"):
    device = torch.device("cpu")
    model = SimplePLCNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    audio, sr = sf.read(degraded_path)
    import numpy as np
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        y = model(x).squeeze(0).numpy()
    sf.write(out, y, sr)
    print("Saved", out)

if __name__ == "__main__":
    restore("assets/example_wav.wav")
