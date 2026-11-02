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
ALPHAS = np.logspace(-2, 5, 15)

def clip5(X): return np.clip(X, -5, 5)
def prep(): return [RobustScaler(), FunctionTransformer(clip5), StandardScaler()]
def ridge(): return make_pipeline(*prep(), RidgeCV(alphas=ALPHAS))
def r2(yt, yp, ym): return 1 - ((yt - yp) ** 2).sum() / ((yt - ym) ** 2).sum()
def pred(m, X, ytr): return np.clip(m.predict(X), ytr.min(), ytr.max())

def residualize(target, source, tr):
    """Remove from `target` everything a ridge regression on `source` can explain.
    Fit on training children only, then apply to everyone."""
    sc = make_pipeline(*prep()).fit(source[tr])
    S = sc.transform(source)
    T = StandardScaler().fit(target[tr]).transform(target)
    m = RidgeCV(alphas=ALPHAS).fit(S[tr], T[tr])
    return T - m.predict(S)

def run(get_layers, n_l, tr, te, seed):
    """Nested layer choice (3-fold inside the training set), then score on the test fold."""
    best = 0
    if n_l > 1:
        sc = np.zeros(n_l)
        for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
            for l in range(n_l):
                X = get_layers(l)
                m = ridge().fit(X[tr[a]], y[tr[a]])
                sc[l] += r2(y[tr[b]], pred(m, X[tr[b]], y[tr[a]]), y[tr[a]].mean())
        best = int(np.argmax(sc))
    X = get_layers(best)
    m = ridge().fit(X[tr], y[tr])
    p = pred(m, X[te], y[tr])
    return r2(y[te], p, y[tr].mean()), np.abs(y[te] - p).mean(), best + 1

names = ['pretrained minus standard', 'random minus standard', 'standard minus pretrained']
res = {k: {'r2': [], 'mae': [], 'layer': []} for k in names}
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
        seed = 100 * r + k
        # 1. LaBraM pretrained with the standard-feature information removed
        pre_res = [residualize(pre[:, l], std, tr) for l in range(n_layers)]
        out = run(lambda l: pre_res[l], n_layers, tr, te, seed)
        # 2. Same for random weights (control)
        rand_res = [residualize(rand[:, l], std, tr) for l in range(n_layers)]
        out2 = run(lambda l: rand_res[l], n_layers, tr, te, seed)
        # 3. Reverse: standard features with each LaBraM layer's information removed
        std_res = [residualize(std, pre[:, l], tr) for l in range(n_layers)]
        out3 = run(lambda l: std_res[l], n_layers, tr, te, seed)
        for name, o in zip(names, (out, out2, out3)):
            res[name]['r2'].append(o[0]); res[name]['mae'].append(o[1]); res[name]['layer'].append(o[2])
    print('repeat', r + 1, 'of 5 done')

def corrected_t(d, test_frac=0.2):
    d = np.asarray(d); J = len(d)
    var = (1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * tdist.sf(abs(t), J - 1))

summary = {}
for name, v in res.items():
    t, p = corrected_t(v['r2'])
    summary[name] = {'r2': float(np.mean(v['r2'])), 'r2_sd': float(np.std(v['r2'])),
                     'mae': float(np.mean(v['mae'])), 'folds_r2_above_0': int((np.array(v['r2']) > 0).sum()),
                     't_vs_0': t, 'p_vs_0': p, 'top_layer': int(np.bincount(v['layer']).argmax())}
d = np.array(res['pretrained minus standard']['r2']) - np.array(res['random minus standard']['r2'])
t, p = corrected_t(d)
summary['pretrained residual vs random residual'] = {'mean_diff_r2': float(d.mean()), 'folds_better': int((d > 0).sum()), 't': t, 'p': p}

json.dump(summary, open(os.path.join(PROJECT, 'results_residual.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
