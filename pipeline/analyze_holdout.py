"""Held-out test: fit on the original 373 children (releases 1-3), test once on new children (releases 4-5).

Every model choice (ridge penalty, layer, residualization) is made on the training children only.
Uncertainty comes from bootstrapping the held-out children (2,000 resamples).
Usage: python analyze_holdout.py TRAIN_WIN TEST_WIN FM_TRAIN FM_TEST RAW_TRAIN RAW_TEST OUT_JSON
"""
import glob, json, os, sys, importlib.util
import numpy as np, pandas as pd
from scipy.signal import csd, welch
from sklearn.model_selection import KFold

TRW, TEW, FTR, FTE, RTR, RTE, OUT = sys.argv[1:8]
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('afm', os.path.join(here, 'analyze_fm_funcs.py'))
A = importlib.util.module_from_spec(sp); sp.loader.exec_module(A)
sys.path.insert(0, here)
from fm_adapters import BIPOLAR_18, IDX


def parts(raw):
    p = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(raw, 'participants_R*.tsv'))])
    return p.drop_duplicates('participant_id').set_index('participant_id')


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


def load(fm_dir, win, raw, models):
    P = parts(raw)
    emb = {}
    for m in models:
        for mode in ('pretrained', 'random'):
            f = os.path.join(fm_dir, f'{m}_{mode}.npz')
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                emb[(m, mode)] = dict(zip(list(z['ids']), np.nan_to_num(z['emb'].astype(float))))
    ids = sorted(set.intersection(*[set(v) for v in emb.values()]))
    ids = [s for s in ids if s in P.index and np.isfinite(P.loc[s, 'age'])]
    fb = [feats(win, s) for s in ids]
    X = {'F': np.stack([x[0] for x in fb]), 'B': np.stack([x[1] for x in fb]), 'C': np.stack([x[2] for x in fb])}
    X['F+B+C'] = np.hstack([X['F'], X['B'], X['C']])
    E = {k: np.stack([v[s] for s in ids]) for k, v in emb.items()}
    return ids, P.loc[ids, 'age'].to_numpy(float), X, E


MODELS = sys.argv[8].split(',') if len(sys.argv) > 8 else ['biot']
id_tr, y_tr, X_tr, E_tr = load(FTR, TRW, RTR, MODELS)
id_te, y_te, X_te, E_te = load(FTE, TEW, RTE, MODELS)
assert not set(id_tr) & set(id_te)
print('train', len(id_tr), 'test', len(id_te), flush=True)

n_tr = len(y_tr)
Y = np.r_[y_tr, y_te]
tr, te = np.arange(n_tr), np.arange(n_tr, n_tr + len(y_te))


def stack(a, b): return np.vstack([a, b])


def pick_layer(get, nl):
    if nl == 1:
        return 0
    sc = np.zeros(nl)
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        for l in range(nl):
            Xl = get(l); m = A.ridge().fit(Xl[tr[a]], Y[tr[a]])
            sc[l] += A.r2(Y[tr[b]], A.pred(m, Xl[tr[b]], Y[tr[a]]), Y[tr[a]].mean())
    return int(np.argmax(sc))


def fit_predict(get, nl):
    l = pick_layer(get, nl)
    Xl = get(l); m = A.ridge().fit(Xl[tr], Y[tr])
    return A.pred(m, Xl[te], Y[tr]), l + 1


preds, layers = {}, {}
base = {k: stack(X_tr[k], X_te[k]) for k in X_tr}
for k in ['F', 'B', 'C', 'F+B+C']:
    preds[k], _ = fit_predict(lambda l, k=k: base[k], 1)
for m in MODELS:
    for mode in ('pretrained', 'random'):
        if (m, mode) not in E_tr:
            continue
        Em = stack(E_tr[(m, mode)], E_te[(m, mode)]); nl = Em.shape[1]
        tag = m if mode == 'pretrained' else f'{m} random'
        preds[tag], layers[tag] = fit_predict(lambda l: Em[:, l], nl)
        if True:
            preds[f'{tag} + F+B+C'], _ = fit_predict(lambda l: np.hstack([base['F+B+C'], Em[:, l]]), nl)
        for src in ['F', 'F+B+C']:
            res = [A.residualize(Em[:, l], base[src], tr) for l in range(nl)]
            preds[f'{tag} minus {src}'], _ = fit_predict(lambda l: res[l], nl)
    print(m, 'done', flush=True)

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y_te), len(y_te)) for _ in range(2000)]
ybar = y_tr.mean()
def r2(p, idx=None):
    idx = np.arange(len(y_te)) if idx is None else idx
    return A.r2(y_te[idx], p[idx], ybar)
out = {'n_train': n_tr, 'n_test': int(len(y_te)), 'layers': layers, 'r2': {}, 'compare': {}}
for k, p in preds.items():
    bs = np.array([r2(p, i) for i in boots])
    out['r2'][k] = {'r2': round(r2(p), 3), 'ci95': [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                    'p_le_0': round(float((bs <= 0).mean()), 4), 'mae': round(float(np.abs(y_te - p).mean()), 2)}
def cmp(a, b):
    d = np.array([r2(preds[a], i) - r2(preds[b], i) for i in boots])
    return {'diff': round(r2(preds[a]) - r2(preds[b]), 3), 'ci95': [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
            'p_two_sided': round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
for m in MODELS:
    for a, b in [(m, 'F'), (m, 'F+B+C'), (f'{m} + F+B+C', 'F+B+C'), (f'{m} random + F+B+C', 'F+B+C'), (f'{m} + F+B+C', f'{m} random + F+B+C'),
                 (m, f'{m} random'), (f'{m} minus F+B+C', f'{m} random minus F+B+C'), (f'{m} minus F', f'{m} random minus F')]:
        if a in preds and b in preds:
            out['compare'][f'{a} vs {b}'] = cmp(a, b)
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
