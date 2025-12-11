# src/utils/stft.py
import numpy as np
from numpy.fft import rfft, irfft
from numpy.lib.stride_tricks import as_strided

def hop2hsize(wind, hop):
    if hop >= 1:
        return int(hop)
    return int(len(wind) * hop)

def stana(sig, wind, hop, synth=False):
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize - fsize
    else:
        sstart = 0
    send = ssize
    nframe = int(np.ceil((send - sstart) / hsize))
    zpleft = -sstart
    zpright = (nframe - 1) * hsize + fsize - zpleft - ssize
    if zpleft > 0 or zpright > 0:
        sigpad = np.zeros(ssize + zpleft + zpright, dtype=sig.dtype)
        sigpad[zpleft:len(sigpad) - zpright] = sig
    else:
        sigpad = sig
    return as_strided(sigpad, shape=(nframe, fsize),
                      strides=(sig.itemsize * hsize, sig.itemsize)) * wind

def stft(sig, dft_size=512, hop_fraction=0.5):
    window = np.hamming(dft_size + 1)[:-1]
    frames = stana(sig, window, hop_fraction, synth=True)
    return rfft(frames, n=dft_size)

def istft(spec, dft_size=512, hop_fraction=0.5):
    # simple inverse for prototyping: use overlap-add naive
    frames = irfft(spec, n=dft_size)
    # TODO: implement perfect overlap-add for production
    return frames.flatten()
