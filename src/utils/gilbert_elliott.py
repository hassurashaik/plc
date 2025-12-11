# src/utils/gilbert_elliott.py
import numpy as np

def sample_ge_sequence(n_frames, p, q, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    state = 0
    seq = np.zeros(n_frames, dtype=np.int8)
    for t in range(n_frames):
        if state == 0:
            if rng.rand() < p:
                state = 1
        else:
            if rng.rand() < q:
                state = 0
        seq[t] = state
    return seq

def sample_pq_under_rate(max_attempts=100, target_max_r=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(max_attempts):
        p = rng.uniform(0.1, 0.9)
        q = rng.uniform(0.1, 0.9)
        r = p/(p+q)
        if r < target_max_r:
            return p, q, r
    return 0.1, 0.9, 0.1
