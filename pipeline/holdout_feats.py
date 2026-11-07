import os
import numpy as np
from scipy.signal import csd, welch
from fm_adapters import BIPOLAR_18, IDX

def feats(win, sub):
    w = np.load(os.path.join(win, f'{sub}.npy'))[:4].astype(float)
    f, p = welch(w, fs=200, nperseg=400, axis=-1)
    band = (f >= 1) & (f <= 45)
    F = np.log10(p.mean(0)[:, band] + 1e-30)
    b = np.stack([w[:, IDX[a]] - w[:, IDX[c]] for a, c in BIPOLAR_18], axis=1)
    _, pb = welch(b, fs=200, nperseg=400, axis=-1)
    B = np.log10(pb.mean(0)[:, band] + 1e-30)
    C = []
    for a, c in BIPOLAR_18:
        _, sxy = csd(w[:, IDX[a]], w[:, IDX[c]], fs=200, nperseg=400, axis=-1)
        coh = np.abs(sxy.mean(0)) ** 2 / (p[:, IDX[a]].mean(0) * p[:, IDX[c]].mean(0) + 1e-40)
        C.append(np.arctanh(np.sqrt(np.clip(coh[band], 0, 0.9999))))
    return F.ravel(), B.ravel(), np.ravel(C)


