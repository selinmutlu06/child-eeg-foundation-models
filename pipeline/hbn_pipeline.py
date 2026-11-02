"""HBN resting-state pipeline: standard EEG features vs. LaBraM embeddings.

Week 1 of the autism EEG foundation-model project. Everything here runs on
open Healthy Brain Network data (age and CBCL factors), no data agreement needed.
"""
import copy
import os

import boto3
import mne
import numpy as np
import pandas as pd
import torch
from botocore import UNSIGNED
from botocore.config import Config
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

mne.set_log_level("error")

# ---------------------------------------------------------------- settings
BUCKET = "fcp-indi"
PREFIX = "data/Projects/HBN/BIDS_EEG/cmi_bids_{release}/"

# EGI HydroCel 129 electrodes closest to the standard 10-20 positions
# (the mapping commonly used for HBN / EGI data).
EGI_TO_1020 = {
    "E22": "FP1", "E9": "FP2", "E33": "F7", "E24": "F3", "E11": "FZ",
    "E124": "F4", "E122": "F8", "E45": "T7", "E36": "C3", "Cz": "CZ",
    "E104": "C4", "E108": "T8", "E58": "P7", "E52": "P3", "E62": "PZ",
    "E92": "P4", "E96": "P8", "E70": "O1", "E83": "O2",
}
CH_1020 = list(EGI_TO_1020.values())

SFREQ = 200            # LaBraM expects 200 Hz
WIN_SEC = 15           # the braindecode LaBraM checkpoint takes 3000 samples
WIN = SFREQ * WIN_SEC
SKIP_SEC = 2           # skip 2 s after each "close your eyes" instruction
PTP_REJECT_UV = 500    # drop windows whose peak-to-peak exceeds this
LABRAM_SCALE = 1e4     # volts -> units of 0.1 mV (microvolts / 100), as in LaBraM

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}


# ---------------------------------------------------------------- download
def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED),
                        region_name="us-east-1")


def get_participants(release, data_dir):
    """Download and return participants.tsv for one HBN release."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"participants_{release}.tsv")
    if not os.path.exists(path):
        s3_client().download_file(BUCKET, PREFIX.format(release=release)
                                  + "participants.tsv", path)
    return pd.read_csv(path, sep="\t")


def download_resting_state(release, subject, data_dir):
    """Download one child's resting-state .set and events.tsv. Returns paths."""
    s3 = s3_client()
    base = PREFIX.format(release=release) + f"{subject}/eeg/{subject}_task-RestingState_"
    out = {}
    for suffix in ["eeg.set", "events.tsv"]:
        local = os.path.join(data_dir, f"{subject}_RestingState_{suffix}")
        if not os.path.exists(local):
            s3.download_file(BUCKET, base + suffix, local)
        out[suffix] = local
    return out


# ---------------------------------------------------------------- preprocessing
def eyes_closed_windows(set_path, events_path):
    """Return (n_windows, 19, 3000) eyes-closed windows in volts, plus counts."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    raw.filter(0.5, 75.0)
    raw.notch_filter(60.0)
    # Find noisy or flat electrodes (robust z-score of log standard deviation),
    # rebuild them from their neighbours, then average-reference.
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"))
    egi = [c for c in raw.ch_names if c != "Cz"]       # Cz is the recording reference
    log_sd = np.log(raw.get_data(picks=egi).std(axis=1) + 1e-12)
    mad = np.median(np.abs(log_sd - np.median(log_sd))) + 1e-12
    z = (log_sd - np.median(log_sd)) / (1.4826 * mad)
    raw.info["bads"] = [c for c, zz in zip(egi, z) if abs(zz) > 3]
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    raw.pick(list(EGI_TO_1020))
    raw.rename_channels(EGI_TO_1020)
    raw.reorder_channels(CH_1020)
    raw.resample(SFREQ)
    data = raw.get_data()

    events = pd.read_csv(events_path, sep="\t")
    onsets = events.loc[events["value"] == "instructed_toCloseEyes", "onset"].to_numpy()
    windows, n_total = [], 0
    for onset in onsets:
        start = int((onset + SKIP_SEC) * SFREQ)
        block_end = int((onset + 40) * SFREQ)  # each eyes-closed block is 40 s
        while start + WIN <= min(block_end, data.shape[1]):
            n_total += 1
            w = data[:, start:start + WIN]
            if np.ptp(w, axis=1).max() * 1e6 < PTP_REJECT_UV:
                windows.append(w)
            start += WIN
    windows = np.stack(windows) if windows else np.empty((0, 19, WIN))
    return windows, n_total


# ---------------------------------------------------------------- standard features
def standard_features(windows):
    """Band power, relative power, aperiodic slope/offset, alpha peak. One row per child."""
    from specparam import SpectralModel

    freqs, psd = welch(windows, fs=SFREQ, nperseg=2 * SFREQ, axis=-1)
    psd = psd.mean(axis=0)                      # average over windows -> (19, n_freqs)
    feats = {}
    total = psd[:, (freqs >= 1) & (freqs < 45)].sum(axis=1)
    for band, (lo, hi) in BANDS.items():
        p = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        for ch, abs_p, rel_p in zip(CH_1020, p, p / total):
            feats[f"logpow_{band}_{ch}"] = np.log10(abs_p)
            feats[f"relpow_{band}_{ch}"] = rel_p

    def params(fm, kind):
        for name in (kind, kind + "_params"):
            try:
                return fm.get_params(name)
            except Exception:
                continue
        return None

    for ch, spectrum in zip(CH_1020, psd):
        fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6, verbose=False)
        fm.fit(freqs, spectrum, [2, 40])
        offset, exponent = params(fm, "aperiodic")[:2]
        feats[f"aper_offset_{ch}"] = offset
        feats[f"aper_exponent_{ch}"] = exponent

    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6, verbose=False)
    fm.fit(freqs, psd.mean(axis=0), [2, 40])
    peaks = np.atleast_2d(params(fm, "peak"))
    alpha = peaks[(peaks[:, 0] >= 7) & (peaks[:, 0] <= 14)] if peaks.size else peaks
    feats["alpha_peak_freq"] = alpha[np.argmax(alpha[:, 1]), 0] if len(alpha) else np.nan
    return feats


# ---------------------------------------------------------------- LaBraM
def load_labram(random_weights=False):
    """Pretrained LaBraM, or the same architecture with re-initialized weights."""
    from braindecode.models import Labram

    model = Labram.from_pretrained("braindecode/labram-pretrained").eval()
    if random_weights:
        model = copy.deepcopy(model)
        torch.manual_seed(0)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for name, p in model.named_parameters():
            if any(k in name for k in ("position", "cls", "time_embed")):
                torch.nn.init.trunc_normal_(p, std=0.02)
    return model


@torch.no_grad()
def labram_embeddings(model, windows, batch_size=8):
    """Mean token embedding at every layer. Returns (n_layers, 200) for one child."""
    acts = []
    hooks = [b.register_forward_hook(lambda m, i, o: acts.append(o[:, 1:].mean(1)))
             for b in model.blocks]
    per_layer = []
    try:
        x_all = torch.tensor(windows * LABRAM_SCALE, dtype=torch.float32)
        for i in range(0, len(x_all), batch_size):
            acts.clear()
            model(x_all[i:i + batch_size], ch_names=CH_1020, return_features=True)
            per_layer.append(torch.stack(acts, dim=1))   # (batch, n_layers, 200)
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(per_layer).mean(0).numpy()          # average over windows


# ---------------------------------------------------------------- evaluation
ALPHAS = np.logspace(-2, 5, 15)


def _clip(X):
    return np.clip(X, -5, 5)


def scaler():
    """Robust scaling, then clip at 5 IQR-units so one noisy child can't dominate.

    The same treatment is applied to every feature set, so the comparison stays fair.
    """
    return [RobustScaler(), FunctionTransformer(_clip), StandardScaler()]


def ridge():
    return make_pipeline(*scaler(), RidgeCV(alphas=ALPHAS))


def cv_r2(X, y, n_splits=5, n_repeats=5, seed=0):
    """Repeated K-fold R^2 for ridge regression, one child per row."""
    scores = []
    for r in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed + r).split(X):
            m = ridge().fit(X[tr], y[tr])
            resid = y[te] - m.predict(X[te])
            scores.append(1 - (resid ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum())
    return float(np.mean(scores)), float(np.std(scores))


def robustness(X, y, n_splits=10, seed=0):
    """Carlson et al. (2024): split children in half, train on each, test on the other.

    P = mean held-out R^2, S = Spearman correlation of the two halves' coefficients,
    R = mean of max(P, 0) * S over random splits.
    """
    rng = np.random.default_rng(seed)
    alpha = ridge().fit(X, y)[-1].alpha_
    P, S = [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(y))
        a, b = idx[: len(y) // 2], idx[len(y) // 2:]
        coefs, perf = [], []
        for tr, te in ((a, b), (b, a)):
            m = make_pipeline(*scaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
            resid = y[te] - m.predict(X[te])
            perf.append(1 - (resid ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
            coefs.append(m[-1].coef_)
        P.append(np.mean(perf))
        S.append(spearmanr(coefs[0], coefs[1]).statistic)
    P, S = np.array(P), np.array(S)
    # Clip P at 0 so a model with no skill scores 0 (otherwise a negative P times a
    # negative S would look like positive robustness).
    return {"P": P.mean(), "S": S.mean(), "R": (np.clip(P, 0, None) * S).mean()}
