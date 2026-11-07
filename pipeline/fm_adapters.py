"""Adapters that turn cleaned HBN windows (volts, 19 x 3000 at 200 Hz, order CH_1020) into
per-layer embeddings for each EEG foundation model, following each model's own preprocessing.

Each adapter: prep(windows) -> tensor batch; the model is loaded once; per-layer outputs are captured
with forward hooks on the model's transformer/residual blocks and mean-pooled over all tokens.
"""
import copy, math
import mne, numpy as np, torch
from scipy.signal import butter, resample_poly, sosfiltfilt
import braindecode.models as M
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def manual(cls, repo, **kw):
    """Build cls(**kw) and load the repo's encoder weights, skipping any shape-mismatched head."""
    m = cls(**kw)
    sd = load_file(hf_hub_download(repo, 'model.safetensors'))
    own = m.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in keep]
    m.load_state_dict(keep, strict=False)
    m._fm_load_report = (len(keep), len(own), [k for k in missing][:8])
    return m

CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
NICE = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}
NAMES = [NICE.get(c, c) for c in CH]                       # standard capitalisation
OLD = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}      # TUH-style names
TUH_ORDER = ['C3', 'C4', 'Cz', 'F3', 'F4', 'Fp1', 'Fp2', 'F7', 'F8', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
IDX = {n: i for i, n in enumerate(NAMES)}
IDX.update({OLD[k]: IDX[k] for k in OLD})
BIPOLAR_18 = [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'), ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
              ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
              ('Fz', 'Cz'), ('Cz', 'Pz')]


def chs_info(names):
    mont = mne.channels.make_standard_montage('standard_1005')
    pos = {k.upper(): v for k, v in mont.get_positions()['ch_pos'].items()}
    out = []
    for n in names:
        p = pos[n.upper()] if n.upper() in pos else pos[{v: k for k, v in OLD.items()}.get(n, n).upper()]
        out.append({'ch_name': n, 'kind': 'eeg', 'loc': np.r_[p, np.zeros(9)]})
    return out


def resample(w, fs_out):
    if fs_out == 200:
        return w
    g = math.gcd(200, fs_out)
    return resample_poly(w, fs_out // g, 200 // g, axis=-1)


def segments(w, n):
    """Cut (k, C, T) windows into non-overlapping segments of n samples."""
    k = w.shape[-1] // n
    return np.concatenate([w[..., i * n:(i + 1) * n] for i in range(k)], axis=0)


def zscore(x, clip=None):
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
    return np.clip(x, -clip, clip) if clip else x


def reorder(w, names):
    return w[:, [IDX[n] for n in names]]


# ------------------------------------------------------------------ model specs
def spec(name):
    """Returns (loader, prep, forward). loader() -> nn.Module; prep(windows) -> np array batch."""
    if name == 'labram':
        import hbn_pipeline as hp
        load = hp.load_labram
        prep = lambda w: w * hp.LABRAM_SCALE                               # 0.1 mV units, 15 s at 200 Hz
        fwd = lambda m, x: m(x, ch_names=hp.CH_1020, return_features=True)
    elif name == 'cbramod':
        load = lambda: M.CBraMod.from_pretrained('braindecode/cbramod-pretrained', n_chans=19, n_times=3000, n_outputs=1)
        prep = lambda w: reorder(w, TUH_ORDER) * 1e4                       # 100 uV units, 15 s at 200 Hz
        fwd = lambda m, x: m(x)
    elif name == 'codebrain':
        load = lambda: manual(M.CodeBrain, 'braindecode/codebrain-pretrained', n_chans=19, n_times=3000, sfreq=200, n_outputs=1)
        prep = lambda w: reorder(w, TUH_ORDER) * 1e4
        fwd = lambda m, x: m(x)
    elif name == 'eegdino':
        load = lambda: M.EEGDINO.from_pretrained('braindecode/eegdino-small-pretrained', n_times=3000, n_outputs=1)
        prep = lambda w: reorder(w, TUH_ORDER) * 1e4
        fwd = lambda m, x: m(x)
    elif name == 'eegpt':
        def load():
            m = manual(M.EEGPT, 'braindecode/eegpt-pretrained', n_chans=19, n_times=1000, sfreq=250, chs_info=[{'ch_name': c} for c in CH], n_outputs=1, chan_proj_type='none')
            return m
        prep = lambda w: segments(resample(w, 250), 1000) * 1e3            # millivolts, 4 s at 250 Hz
        fwd = lambda m, x: m(x)
    elif name == 'biot':
        load = lambda: M.BIOT.from_pretrained('braindecode/biot-pretrained-six-datasets-18chs', n_outputs=1)
        def prep(w):
            b = np.stack([w[:, IDX[a]] - w[:, IDX[c]] for a, c in BIPOLAR_18], axis=1)
            s = segments(b, 1000)                                          # 5 s at 200 Hz
            return s / (np.quantile(np.abs(s), 0.95, axis=-1, keepdims=True) + 1e-12)
        fwd = lambda m, x: m(x)
    elif name == 'bendr':
        load = lambda: M.BENDR.from_pretrained('braindecode/braindecode-bendr', n_outputs=1)
        def prep(w):
            s = segments(resample(w, 250), 1000)                           # 4 s at 250 Hz
            lo, hi = s.min(axis=(1, 2), keepdims=True), s.max(axis=(1, 2), keepdims=True)
            scaled = 2 * (s - lo) / (hi - lo + 1e-12) - 1                  # per-segment min-max to [-1, 1]
            rel = np.broadcast_to((hi - lo) / 400e-6, (s.shape[0], 1, s.shape[2]))  # relative amplitude channel
            return np.concatenate([scaled, rel], axis=1)
        fwd = lambda m, x: m(x)
    elif name == 'signaljepa':
        load = lambda: M.SignalJEPA.from_pretrained('braindecode/signal-jepa', chs_info=chs_info(NAMES), n_chans=19, n_times=256, n_outputs=1)
        def prep(w):
            x = sosfiltfilt(butter(4, [0.5, 40], btype='band', fs=200, output='sos'), w, axis=-1)
            return segments(resample(x, 128), 256) * 1e6                   # microvolts, 2 s at 128 Hz
        fwd = lambda m, x: m(x)
    elif name == 'steegformer':
        load = lambda: M.STEEGFormer.from_pretrained('braindecode/STEEGFormer-small', n_chans=19, n_times=640, chs_info=chs_info(NAMES), n_outputs=1)
        prep = lambda w: zscore(segments(resample(w, 128), 640))           # 5 s at 128 Hz
        fwd = lambda m, x: m(x)
    elif name == 'luna':
        load = lambda: M.LUNA.from_pretrained('PulpBio/LUNA', filename='LUNA_base.safetensors', n_chans=19, n_times=1280, chs_info=chs_info(NAMES), n_outputs=1, embed_dim=64, num_queries=4, depth=8)
        prep = lambda w: zscore(segments(resample(w, 256), 1280))          # 5 s at 256 Hz
        fwd = lambda m, x: m(x)
    elif name == 'reve':
        load = lambda: M.REVE.from_pretrained('brain-bzh/reve-base', n_chans=19, n_times=3000, chs_info=chs_info(NAMES), n_outputs=1)
        prep = lambda w: zscore(w, clip=15)                                # 15 s at 200 Hz
        fwd = lambda m, x: m(x)
    else:
        raise ValueError(name)
    return load, prep, fwd


def block_candidates(model):
    out = []
    for n, mod in model.named_modules():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)) and len(mod) >= 2:
            if len({type(c).__name__ for c in mod}) == 1 and not isinstance(mod[0], (torch.nn.Linear, torch.nn.Dropout, torch.nn.GELU)):
                out.append((n, mod))
    return sorted(out, key=lambda x: -len(x[1]))


def hook_target(blk):
    """For a layer stored as ModuleList([attn, ff]) hook the attention's *input* (the residual stream)."""
    return (blk[0], True) if isinstance(blk, torch.nn.ModuleList) else (blk, False)


def attach(blk, fn):
    mod, pre = hook_target(blk)
    if pre:
        return mod.register_forward_pre_hook(lambda m, inp: fn(inp[0]))
    return mod.register_forward_hook(lambda m, inp, out: fn(out))


@torch.no_grad()
def block_list(model, fwd=None, x=None):
    """Longest repeated-block stack whose every block runs during a forward pass."""
    cands = block_candidates(model)
    if fwd is None or x is None:
        return cands[0] if cands else None
    fired = {}
    hooks = [attach(blk, lambda o, key=(n, j): fired.__setitem__(key, True)) for n, mod in cands for j, blk in enumerate(mod)]
    try:
        fwd(model, torch.tensor(np.ascontiguousarray(x[:2]), dtype=torch.float32))
    finally:
        for h in hooks:
            h.remove()
    for n, mod in cands:
        if len(mod) >= 4 and all(fired.get((n, j)) for j in range(len(mod))):
            return n, mod
    return None


def randomize(model, seed=0):
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in m.modules():
        if hasattr(mod, 'reset_parameters'):
            try:
                mod.reset_parameters()
            except Exception:
                pass
    for n, p in m.named_parameters():
        if any(k in n.lower() for k in ('pos', 'cls', 'token', 'embed')) and p.dim() >= 1:
            torch.nn.init.trunc_normal_(p, std=0.02)
    return m


def pooled(o, time_first=False):
    if isinstance(o, (tuple, list)):
        o = o[0]
    if isinstance(o, dict):
        o = next(v for v in o.values() if torch.is_tensor(v))
    o = o.float()
    if time_first:
        o = o.transpose(0, 1)
    if o.dim() == 2:
        return o
    b = o.shape[0]
    return o.reshape(b, -1, o.shape[-1]).mean(1) if o.shape[-1] <= 4096 else o.reshape(b, o.shape[1], -1).mean(-1)


@torch.no_grad()
def embed(model, fwd, x, batch=8, max_layers=12, time_first=False):
    """Mean-pooled output of every block, averaged over segments. Returns (n_layers, dim)."""
    found = getattr(model, '_fm_blocks', None) or block_list(model, fwd, x)
    model._fm_blocks = found
    name, blocks = found
    keep = np.unique(np.linspace(0, len(blocks) - 1, min(max_layers, len(blocks))).round().astype(int))
    acts = {}
    hooks = [attach(blocks[i], lambda o, i=i: acts.__setitem__(i, pooled(o, time_first))) for i in keep]
    per = []
    try:
        xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
        for s in range(0, len(xt), batch):
            acts.clear()
            fwd(model, xt[s:s + batch])
            per.append(torch.stack([acts[i] for i in keep], 1))
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(per).mean(0).numpy()
