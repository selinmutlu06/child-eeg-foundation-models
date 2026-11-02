import json, os, numpy as np
from scipy.stats import t as tdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

PROJECT = '/content/drive/MyDrive/autism_eeg_project'
D = np.load(os.path.join(PROJECT, 'all_children.npz'), allow_pickle=True)
y = D['age']
std = D['feats'].astype(float)
std = np.where(np.isnan(std), np.nanmedian(std, axis=0), std)
pre, rand = D['labram'].astype(float), D['labram_random'].astype(float)
n_layers = pre.shape[1]

def clip5(X): return np.clip(X, -5, 5)
def ridge():
    return make_pipeline(RobustScaler(), FunctionTransformer(clip5), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 5, 15)))

# Each feature set is a function layer -> matrix, so "combined" can pair standard features with any LaBraM layer.
SETS = {
    'standard': (1, lambda l: std),
    'pretrained': (n_layers, lambda l: pre[:, l]),
    'random': (n_layers, lambda l: rand[:, l]),
    'standard+pretrained': (n_layers, lambda l: np.hstack([std, pre[:, l]])),
}

def r2(yt, yp, ym): return 1 - ((yt - yp) ** 2).sum() / ((yt - ym) ** 2).sum()
def pred(m, X, ytr): return np.clip(m.predict(X), ytr.min(), ytr.max())

def fit_select(nl, get, tr, seed):
    best = 0
    if nl > 1:
        sc = np.zeros(nl)
        for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
            for l in range(nl):
                X = get(l)
                m = ridge().fit(X[tr[a]], y[tr[a]])
                sc[l] += r2(y[tr[b]], pred(m, X[tr[b]], y[tr[a]]), y[tr[a]].mean())
        best = int(np.argmax(sc))
    return best, ridge().fit(get(best)[tr], y[tr])

res = {k: {'r2': [], 'mae': [], 'layer': []} for k in SETS}
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
        for name, (nl, get) in SETS.items():
            l, m = fit_select(nl, get, tr, 100 * r + k)
            p = pred(m, get(l)[te], y[tr])
            res[name]['r2'].append(r2(y[te], p, y[tr].mean()))
            res[name]['mae'].append(np.abs(y[te] - p).mean())
            res[name]['layer'].append(l + 1)
    print('repeat', r + 1, 'of 5 done')

def corrected_t(a, b, test_frac=0.2):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.array(a) - np.array(b); J = len(d)
    var = (1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * tdist.sf(abs(t), J - 1))

out = {'n': int(len(y)), 'age_mean': float(y.mean()), 'age_sd': float(y.std()),
       'baseline_mae_predict_mean': float(np.abs(y - y.mean()).mean())}
for name, v in res.items():
    out[name] = {'r2': float(np.mean(v['r2'])), 'r2_sd': float(np.std(v['r2'])),
                 'mae': float(np.mean(v['mae'])), 'mae_sd': float(np.std(v['mae'])),
                 'top_layer': int(np.bincount(v['layer']).argmax())}
for a, b in [('pretrained', 'standard'), ('pretrained', 'random'), ('standard+pretrained', 'standard'),
             ('standard+pretrained', 'pretrained'), ('random', 'standard')]:
    t, p = corrected_t(res[a]['r2'], res[b]['r2'])
    d = np.array(res[a]['r2']) - np.array(res[b]['r2'])
    out[f'{a} vs {b}'] = {'mean_diff_r2': float(d.mean()), 'folds_better': int((d > 0).sum()), 't': t, 'p': p}

json.dump(out, open(os.path.join(PROJECT, 'results_v3.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
