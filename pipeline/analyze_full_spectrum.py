import glob, json, os, sys, numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

OUT, RAW = sys.argv[1], sys.argv[2]
parts = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(RAW, 'participants_R*.tsv'))])
parts = parts.drop_duplicates('participant_id').set_index('participant_id')
files = sorted(glob.glob(os.path.join(OUT, 'sub-*.npz')))
ids = [os.path.basename(f)[:-4] for f in files]
Z = [np.load(f) for f in files]
print(len(ids), 'children')
y = parts.loc[ids, 'age'].to_numpy(float)
std = np.stack([z['feats'] for z in Z]).astype(float)
std = np.where(np.isnan(std), np.nanmedian(std, axis=0), std)
pre = np.stack([z['labram'] for z in Z]).astype(float)
band = (Z[0]['freqs'] >= 1) & (Z[0]['freqs'] <= 45)
full = np.stack([z['log_psd'][:, band].ravel() for z in Z]).astype(float)
print('full spectrum:', full.shape[1], 'features')
n_layers = pre.shape[1]
ALPHAS = np.logspace(-2, 5, 15)

def clip5(X): return np.clip(X, -5, 5)
def prep(): return [RobustScaler(), FunctionTransformer(clip5), StandardScaler()]
def ridge(): return make_pipeline(*prep(), RidgeCV(alphas=ALPHAS))
def r2(yt, yp, ym): return 1 - ((yt - yp) ** 2).sum() / ((yt - ym) ** 2).sum()
def pred(m, X, ytr): return np.clip(m.predict(X), ytr.min(), ytr.max())

def residualize(target, source, tr):
    S = make_pipeline(*prep()).fit(source[tr]).transform(source)
    T = StandardScaler().fit(target[tr]).transform(target)
    return T - RidgeCV(alphas=ALPHAS).fit(S[tr], T[tr]).predict(S)

def run(get, n_l, tr, te, seed):
    best = 0
    if n_l > 1:
        sc = np.zeros(n_l)
        for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
            for l in range(n_l):
                X = get(l); m = ridge().fit(X[tr[a]], y[tr[a]])
                sc[l] += r2(y[tr[b]], pred(m, X[tr[b]], y[tr[a]]), y[tr[a]].mean())
        best = int(np.argmax(sc))
    X = get(best); m = ridge().fit(X[tr], y[tr]); p = pred(m, X[te], y[tr])
    return r2(y[te], p, y[tr].mean()), np.abs(y[te] - p).mean(), best + 1

names = ['standard', 'full spectrum', 'pretrained', 'full spectrum+pretrained',
         'pretrained minus full spectrum', 'full spectrum minus pretrained']
res = {k: {'r2': [], 'mae': [], 'layer': []} for k in names}
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
        seed = 100 * r + k
        pre_res = [residualize(pre[:, l], full, tr) for l in range(n_layers)]
        full_res = [residualize(full, pre[:, l], tr) for l in range(n_layers)]
        outs = [run(lambda l: std, 1, tr, te, seed),
                run(lambda l: full, 1, tr, te, seed),
                run(lambda l: pre[:, l], n_layers, tr, te, seed),
                run(lambda l: np.hstack([full, pre[:, l]]), n_layers, tr, te, seed),
                run(lambda l: pre_res[l], n_layers, tr, te, seed),
                run(lambda l: full_res[l], n_layers, tr, te, seed)]
        for name, o in zip(names, outs):
            res[name]['r2'].append(o[0]); res[name]['mae'].append(o[1]); res[name]['layer'].append(o[2])
    print('repeat', r + 1, 'of 5 done', flush=True)

def corrected_t(d, test_frac=0.2):
    d = np.asarray(d); J = len(d)
    t = d.mean() / np.sqrt((1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1))
    return float(t), float(2 * tdist.sf(abs(t), J - 1))

out = {'n': len(y)}
for name, v in res.items():
    t, p = corrected_t(v['r2'])
    out[name] = {'r2': round(float(np.mean(v['r2'])), 4), 'r2_sd': round(float(np.std(v['r2'])), 4),
                 'mae': round(float(np.mean(v['mae'])), 3), 'folds_r2_above_0': int((np.array(v['r2']) > 0).sum()),
                 'p_vs_0': round(p, 4), 'top_layer': int(np.bincount(v['layer']).argmax())}
for a, b in [('full spectrum', 'standard'), ('pretrained', 'full spectrum'),
             ('full spectrum+pretrained', 'full spectrum'), ('full spectrum+pretrained', 'pretrained')]:
    d = np.array(res[a]['r2']) - np.array(res[b]['r2']); t, p = corrected_t(d)
    out[f'{a} vs {b}'] = {'mean_diff_r2': round(float(d.mean()), 4), 'folds_better': int((d > 0).sum()), 'p': round(p, 4)}

json.dump(out, open(os.path.join(OUT, 'results_full_spectrum.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
