"""What is BIOT's extra age signal? Remove one frequency band or one group of bipolar channels from
BIOT's input, re-embed, and measure how much of BIOT's held-out gain over the baselines remains.

Same split as analyze_holdout.py: fit on releases 1-3 (373 children), test once on releases 4-5 (612).
The baselines (referential spectrum + bipolar spectrum + coherence) are never ablated.
Usage: python ablate_biot.py TRAIN_WIN TEST_WIN RAW_TRAIN RAW_TEST OUT_JSON
"""
import glob, json, os, sys, importlib.util
import numpy as np, pandas as pd, torch
from scipy.signal import butter, sosfiltfilt

TRW, TEW, RTR, RTE, OUT = sys.argv[1:6]
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
sp = importlib.util.spec_from_file_location('afm', os.path.join(here, 'analyze_fm_funcs.py'))
A = importlib.util.module_from_spec(sp); sp.loader.exec_module(A)
sp2 = importlib.util.spec_from_file_location('hold_feats', os.path.join(here, 'holdout_feats.py'))
H = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(H)
import fm_adapters as F
torch.set_num_threads(4)

BANDS = {'delta 1-4 Hz': (1, 4), 'theta 4-8 Hz': (4, 8), 'alpha 8-13 Hz': (8, 13), 'beta 13-30 Hz': (13, 30), 'gamma 30-45 Hz': (30, 45)}
PAIRS = [f'{a}-{c}' for a, c in F.BIPOLAR_18]
GROUPS = {'left temporal chain': [0, 1, 2, 3], 'right temporal chain': [4, 5, 6, 7], 'left parasagittal chain': [8, 9, 10, 11],
          'right parasagittal chain': [12, 13, 14, 15], 'midline (Fz-Cz, Cz-Pz)': [16, 17],
          'frontal pairs': [0, 4, 8, 12, 16], 'posterior pairs': [3, 7, 11, 15, 17]}


def ids_ages(win, raw):
    P = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(raw, 'participants_R*.tsv'))]).drop_duplicates('participant_id').set_index('participant_id')
    ids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(win, 'sub-*.npy')))]
    ids = [s for s in ids if s in P.index and np.isfinite(P.loc[s, 'age'])]
    return ids, P.loc[ids, 'age'].to_numpy(float)


id_tr, y_tr = ids_ages(TRW, RTR); id_te, y_te = ids_ages(TEW, RTE)
cache = os.path.join(os.path.dirname(OUT), 'baseline_feats.npz')
if os.path.exists(cache):
    z = np.load(cache); Xb = z['X']
else:
    Xb = np.vstack([np.hstack(H.feats(TRW, s)) for s in id_tr] + [np.hstack(H.feats(TEW, s)) for s in id_te]); np.savez(cache, X=Xb)
Y = np.r_[y_tr, y_te]; tr = np.arange(len(y_tr)); te = np.arange(len(y_tr), len(Y))
print('train', len(tr), 'test', len(te), flush=True)

load, prep, fwd = F.spec('biot')
model = load().eval()


def embed_all(transform_windows=None, zero=None):
    out = []
    for win, ids in ((TRW, id_tr), (TEW, id_te)):
        for s in ids:
            w = np.load(os.path.join(win, f'{s}.npy'))[:4].astype(np.float64)
            if transform_windows is not None:
                w = transform_windows(w)
            x = prep(w)
            if zero is not None:
                x = x.copy(); x[:, zero] = 0.0
            out.append(F.embed(model, fwd, x))
    return np.stack(out)


def bandstop(lo, hi):
    sos = butter(4, [lo, hi], btype='bandstop', fs=200, output='sos')
    return lambda w: sosfiltfilt(sos, w, axis=-1)


def fit_predict(get, nl):
    best = 0
    if nl > 1:
        from sklearn.model_selection import KFold
        sc = np.zeros(nl)
        for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
            for l in range(nl):
                X = get(l); m = A.ridge().fit(X[tr[a]], Y[tr[a]])
                sc[l] += A.r2(Y[tr[b]], A.pred(m, X[tr[b]], Y[tr[a]]), Y[tr[a]].mean())
        best = int(np.argmax(sc))
    X = get(best); m = A.ridge().fit(X[tr], Y[tr])
    return A.pred(m, X[te], Y[tr])


rng = np.random.default_rng(0); boots = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
ybar = y_tr.mean()
def r2(p, i): return A.r2(y_te[i], p[i], ybar)
p_base = fit_predict(lambda l: Xb, 1)

results, gains = {}, {}
def evaluate(name, E):
    p_alone = fit_predict(lambda l: E[:, l], E.shape[1])
    p_comb = fit_predict(lambda l: np.hstack([Xb, E[:, l]]), E.shape[1])
    gains[name] = np.array([r2(p_comb, i) - r2(p_base, i) for i in boots])
    allidx = np.arange(len(te))
    results[name] = {'biot_alone_r2': round(r2(p_alone, allidx), 3),
                     'gain_over_baselines': round(r2(p_comb, allidx) - r2(p_base, allidx), 3),
                     'gain_ci95': [round(float(np.percentile(gains[name], q)), 3) for q in (2.5, 97.5)]}
    if name != 'none (full input)':
        d = gains['none (full input)'] - gains[name]
        results[name]['gain_lost'] = round(float(results['none (full input)']['gain_over_baselines'] - results[name]['gain_over_baselines']), 3)
        results[name]['gain_lost_ci95'] = [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]
    print(name, json.dumps(results[name]), flush=True)
    json.dump(results, open(OUT, 'w'), indent=1)


evaluate('none (full input)', embed_all())
for name, (lo, hi) in BANDS.items():
    evaluate(f'remove {name}', embed_all(transform_windows=bandstop(lo, hi)))
for name, idx in GROUPS.items():
    evaluate(f'remove {name}', embed_all(zero=idx))
print('ABLATION DONE', flush=True)
