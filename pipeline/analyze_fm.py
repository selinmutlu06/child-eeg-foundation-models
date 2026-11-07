"""Compare many EEG foundation models with the full power spectrum, on the same children and windows.

For each model: nested CV R^2 for age (layer chosen inside training folds), the same for the
full spectrum computed from the same 4 windows, the combination, and the model's embeddings after
removing what the spectrum explains. Also attention and p-factor for each model.
Usage: python analyze_fm.py WINDOW_DIR FM_DIR RAW_DIR
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import t as tdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

WIN, FM, RAW = sys.argv[1:4]
ALPHAS = np.logspace(-2, 5, 15)
parts = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(RAW, 'participants_R*.tsv'))])
parts = parts.drop_duplicates('participant_id').set_index('participant_id')

ids_all = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(WIN, 'sub-*.npy')))]
def spectrum(sub):
    w = np.load(os.path.join(WIN, f'{sub}.npy'))[:4].astype(float)
    f, p = welch(w, fs=200, nperseg=400, axis=-1)
    band = (f >= 1) & (f <= 45)
    return np.log10(p.mean(0)[:, band] + 1e-30).ravel()
SPEC = {s: spectrum(s) for s in ids_all}


def clip5(X): return np.clip(X, -5, 5)
def prep(): return [RobustScaler(), FunctionTransformer(clip5), StandardScaler()]
def ridge(): return make_pipeline(*prep(), RidgeCV(alphas=ALPHAS))
def r2(yt, yp, ym): return 1 - ((yt - yp) ** 2).sum() / ((yt - ym) ** 2).sum()
def pred(m, X, ytr): return np.clip(m.predict(X), ytr.min(), ytr.max())


def residualize(target, source, tr):
    S = make_pipeline(*prep()).fit(source[tr]).transform(source)
    T = StandardScaler().fit(target[tr]).transform(target)
    return T - RidgeCV(alphas=ALPHAS).fit(S[tr], T[tr]).predict(S)


def run(get, n_l, y, tr, te, seed):
    best = 0
    if n_l > 1:
        sc = np.zeros(n_l)
        for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
            for l in range(n_l):
                X = get(l); m = ridge().fit(X[tr[a]], y[tr[a]])
                sc[l] += r2(y[tr[b]], pred(m, X[tr[b]], y[tr[a]]), y[tr[a]].mean())
        best = int(np.argmax(sc))
    X = get(best); m = ridge().fit(X[tr], y[tr])
    return r2(y[te], pred(m, X[te], y[tr]), y[tr].mean()), best + 1


def corrected_p(d, test_frac=0.2):
    d = np.asarray(d); J = len(d)
    t = d.mean() / np.sqrt((1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1) + 1e-30)
    return float(2 * tdist.sf(abs(t), J - 1))


results = {}
for f in sorted(glob.glob(os.path.join(FM, '*_pretrained.npz'))):
    name = os.path.basename(f).replace('_pretrained.npz', '')
    z = np.load(f, allow_pickle=True)
    ids = list(z['ids']); E = z['emb'].astype(float)
    E = np.where(np.isfinite(E), E, 0.0)
    full = np.stack([SPEC[s] for s in ids])
    rf = os.path.join(FM, f'{name}_random.npz')
    R = None
    if os.path.exists(rf):
        zr = np.load(rf, allow_pickle=True)
        if list(zr['ids']) == ids:
            R = np.where(np.isfinite(zr['emb']), zr['emb'], 0.0).astype(float)
    nl = E.shape[1]
    out = {'n_children': len(ids), 'n_layers': nl, 'dim': int(E.shape[2])}
    for target in ['age', 'attention', 'p_factor']:
        y_all = parts.loc[ids, target].to_numpy(float); ok = ~np.isnan(y_all)
        y, Eo, Fo = y_all[ok], E[ok], full[ok]
        Ro = R[ok] if R is not None else None
        rows = {k: [] for k in ['model', 'spectrum', 'model+spectrum', 'model minus spectrum', 'random weights']}
        layers = []
        for r in range(5):
            for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
                s = 100 * r + k
                m_r2, lay = run(lambda l: Eo[:, l], nl, y, tr, te, s); rows['model'].append(m_r2); layers.append(lay)
                rows['spectrum'].append(run(lambda l: Fo, 1, y, tr, te, s)[0])
                if Ro is not None and target == 'age':
                    rows['random weights'].append(run(lambda l: Ro[:, l], Ro.shape[1], y, tr, te, s)[0])
                if target == 'age':
                    rows['model+spectrum'].append(run(lambda l: np.hstack([Fo, Eo[:, l]]), nl, y, tr, te, s)[0])
                    res = [residualize(Eo[:, l], Fo, tr) for l in range(nl)]
                    rows['model minus spectrum'].append(run(lambda l: res[l], nl, y, tr, te, s)[0])
        o = {'n': int(ok.sum())}
        for k, v in rows.items():
            if v:
                o[k] = {'r2': round(float(np.mean(v)), 3), 'sd': round(float(np.std(v)), 3), 'p_vs_0': round(corrected_p(v), 4)}
        d = np.array(rows['model']) - np.array(rows['spectrum'])
        o['model vs spectrum'] = {'diff': round(float(d.mean()), 3), 'folds_better': int((d > 0).sum()), 'p': round(corrected_p(d), 4)}
        if rows['model+spectrum']:
            d2 = np.array(rows['model+spectrum']) - np.array(rows['spectrum'])
            o['model+spectrum vs spectrum'] = {'diff': round(float(d2.mean()), 3), 'folds_better': int((d2 > 0).sum()), 'p': round(corrected_p(d2), 4)}
        if rows['random weights']:
            d3 = np.array(rows['model']) - np.array(rows['random weights'])
            o['pretrained vs random'] = {'diff': round(float(d3.mean()), 3), 'folds_better': int((d3 > 0).sum()), 'p': round(corrected_p(d3), 4)}
        o['top_layer'] = int(np.bincount(layers).argmax())
        out[target] = o
    results[name] = out
    print(name, json.dumps(out['age']), flush=True)
    json.dump(results, open(os.path.join(FM, 'results_fm.json'), 'w'), indent=1)
print('ANALYSIS DONE', flush=True)
