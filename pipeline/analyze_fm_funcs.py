import numpy as np
from scipy.stats import t as tdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
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


