"""Does a nonlinear readout change the picture? Kernel ridge regression (RBF) instead of linear ridge.

Part 1: same 25 nested folds on the 373 children, every model vs. the full spectrum.
Part 2: held-out test (fit on 373, test once on 612) for the spectrum baselines, BIOT, and LaBraM.
Kernel width and penalty are tuned by 3-fold CV inside each training set; the layer is chosen the same way.
Usage: python analyze_nonlinear.py TRAIN_WIN TEST_WIN FM_TRAIN FM_TEST RAW_TRAIN RAW_TEST OUT_JSON
"""
import glob, json, os, sys, importlib.util
import numpy as np, pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline

TRW, TEW, FTR, FTE, RTR, RTE, OUT = sys.argv[1:8]
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
def mod(name, f):
    sp = importlib.util.spec_from_file_location(name, os.path.join(here, f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A = mod('afm', 'analyze_fm_funcs.py'); H = mod('hf', 'holdout_feats.py')


class KRR:
    """RBF kernel ridge with gamma scaled by feature count, tuned on the training data passed to fit."""
    def __init__(self): self.m = None
    def fit(self, X, y):
        d = X.shape[1]
        pipe = make_pipeline(*A.prep(), KernelRidge(kernel='rbf'))
        grid = {'kernelridge__alpha': [0.01, 0.1, 1.0, 10.0], 'kernelridge__gamma': [g / d for g in (0.03, 0.1, 0.3, 1.0)]}
        self.mu = y.mean()
        self.m = GridSearchCV(pipe, grid, cv=KFold(3, shuffle=True, random_state=0), scoring='r2').fit(X, y - self.mu)
        return self
    def predict(self, X): return self.m.predict(X) + self.mu


def run(get, nl, y, tr, te, seed):
    best = 0
    if nl > 1:
        sc = np.zeros(nl)
        for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
            for l in range(nl):
                X = get(l); m = KRR().fit(X[tr[a]], y[tr[a]])
                sc[l] += A.r2(y[tr[b]], A.pred(m, X[tr[b]], y[tr[a]]), y[tr[a]].mean())
        best = int(np.argmax(sc))
    X = get(best); m = KRR().fit(X[tr], y[tr])
    return A.pred(m, X[te], y[tr]), best


def parts(raw):
    return pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(raw, 'participants_R*.tsv'))]).drop_duplicates('participant_id').set_index('participant_id')


PART = os.environ.get('PART', 'both'); ONLY = os.environ.get('MODELS')
out = json.load(open(OUT)) if os.path.exists(OUT) else {'cv': {}, 'holdout': {}}
out.setdefault('holdout', {})
# ---------- Part 1: cross-validation, all models
P = parts(RTR)
models = sorted(os.path.basename(f).replace('_pretrained.npz', '') for f in glob.glob(os.path.join(FTR, '*_pretrained.npz')))
if ONLY: models = ONLY.split(',')
z0 = np.load(os.path.join(FTR, 'labram_pretrained.npz'), allow_pickle=True); ids = list(z0['ids'])
y = P.loc[ids, 'age'].to_numpy(float)
Fs = np.stack([H.feats(TRW, s)[0] for s in ids])
folds = [(tr, te) for r in range(5) for tr, te in KFold(5, shuffle=True, random_state=r).split(y)]
spec_r2 = []
for k, (tr, te) in enumerate(folds if PART != 'holdout' else []):
    p, _ = run(lambda l: Fs, 1, y, tr, te, k); spec_r2.append(A.r2(y[te], p, y[tr].mean()))
if spec_r2: out['cv']['full spectrum'] = {'r2': round(float(np.mean(spec_r2)), 3), 'sd': round(float(np.std(spec_r2)), 3)}

for m in (models if PART != 'holdout' else []):
    if m in out['cv']: continue
    z = np.load(os.path.join(FTR, f'{m}_pretrained.npz'), allow_pickle=True)
    assert list(z['ids']) == ids, m
    E = np.nan_to_num(z['emb'].astype(float)); r = []
    for k, (tr, te) in enumerate(folds):
        p, _ = run(lambda l: E[:, l], E.shape[1], y, tr, te, k); r.append(A.r2(y[te], p, y[tr].mean()))
    d = np.array(r) - np.array(spec_r2)
    out['cv'][m] = {'r2': round(float(np.mean(r)), 3), 'sd': round(float(np.std(r)), 3), 'vs_spectrum': round(float(d.mean()), 3),
                    'folds_better': int((d > 0).sum()), 'p': round(A.corrected_p(d), 4)}
    print(m, out['cv'][m], flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)

# ---------- Part 2: held-out
if PART == 'cv':
    print('NONLINEAR DONE', flush=True); sys.exit(0)
Pt = parts(RTE)
def load(fm, win, P_, ms):
    emb = {}
    for m in ms:
        for mode in ('pretrained', 'random'):
            z = np.load(os.path.join(fm, f'{m}_{mode}.npz'), allow_pickle=True)
            emb[(m, mode)] = dict(zip(list(z['ids']), np.nan_to_num(z['emb'].astype(float))))
    ii = sorted(set.intersection(*[set(v) for v in emb.values()]))
    ii = [s for s in ii if s in P_.index and np.isfinite(P_.loc[s, 'age'])]
    X = np.stack([np.hstack(H.feats(win, s)) for s in ii])
    return ii, P_.loc[ii, 'age'].to_numpy(float), X, {k: np.stack([v[s] for s in ii]) for k, v in emb.items()}

ms = ['biot', 'labram']
i1, y1, X1, E1 = load(FTR, TRW, P, ms); i2, y2, X2, E2 = load(FTE, TEW, Pt, ms)
Y = np.r_[y1, y2]; tr = np.arange(len(y1)); te = np.arange(len(y1), len(Y))
nF = X1.shape[1] // 3 + 1  # F block is first (19*89 = 1691)
Xall = np.vstack([X1, X2]); F_only = Xall[:, :1691]
preds = {'full spectrum': run(lambda l: F_only, 1, Y, tr, te, 0)[0], 'baselines (F+B+C)': run(lambda l: Xall, 1, Y, tr, te, 0)[0]}
for m in ms:
    for mode in ('pretrained', 'random'):
        Em = np.vstack([E1[(m, mode)], E2[(m, mode)]]); tag = m if mode == 'pretrained' else f'{m} random'
        preds[tag] = run(lambda l: Em[:, l], Em.shape[1], Y, tr, te, 0)[0]
        preds[f'baselines + {tag}'] = run(lambda l: np.hstack([Xall, Em[:, l]]), Em.shape[1], Y, tr, te, 0)[0]
    print(m, 'holdout done', flush=True)
rng = np.random.default_rng(0); boots = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
yb = y1.mean()
def r2(p, i): return A.r2(y2[i], p[i], yb)
allidx = np.arange(len(te))
for k, p in preds.items():
    bs = [r2(p, i) for i in boots]
    out['holdout'][k] = {'r2': round(r2(p, allidx), 3), 'ci95': [round(float(np.percentile(bs, q)), 3) for q in (2.5, 97.5)]}
def cmp(a, b):
    d = np.array([r2(preds[a], i) - r2(preds[b], i) for i in boots])
    return {'diff': round(r2(preds[a], allidx) - r2(preds[b], allidx), 3), 'ci95': [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]}
out['holdout_compare'] = {f'{a} vs {b}': cmp(a, b) for a, b in [
    ('biot', 'full spectrum'), ('labram', 'full spectrum'), ('baselines + biot', 'baselines (F+B+C)'),
    ('baselines + biot random', 'baselines (F+B+C)'), ('baselines + biot', 'baselines + biot random'),
    ('baselines + labram', 'baselines (F+B+C)'), ('baselines + labram random', 'baselines (F+B+C)')]}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out['holdout'], indent=1)); print(json.dumps(out['holdout_compare'], indent=1)); print('NONLINEAR DONE', flush=True)
