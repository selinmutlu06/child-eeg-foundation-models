"""Does movie-watching EEG predict CBCL factors (and age) where resting EEG does not?

Usage: python analyze_movie_here.py REST_DIR MOVIE_DIR RAW_DIR
Compares, on the same children, resting-state features with movie features
(standard, full spectrum, LaBraM) and inter-subject correlation (ISC) during The Present,
with Despicable Me as a replication for ISC and full spectrum.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

REST, MOVIE, RAW = sys.argv[1:4]
parts = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(RAW, 'participants_R*.tsv'))])
parts = parts.drop_duplicates('participant_id').set_index('participant_id')
TARGETS = ['age', 'attention', 'p_factor', 'internalizing', 'externalizing']
ALPHAS = np.logspace(-2, 5, 15)


def load_movie(task):
    files = sorted(glob.glob(os.path.join(MOVIE, f'*_{task}.npz')))
    ids = [os.path.basename(f).split('_')[0] for f in files]
    return ids, [np.load(f) for f in files]


def isc(arrs):
    """Leave-one-out inter-subject correlation per channel. arrs: (N, 19, T) with NaN for artifacts."""
    X = np.asarray(arrs, float)
    valid = ~np.isnan(X)
    S = np.nansum(X, axis=0); C = valid.sum(0)
    out = np.full(X.shape[:2], np.nan)
    for i in range(len(X)):
        others = (S - np.nan_to_num(X[i])) / np.maximum(C - valid[i], 1)
        for c in range(X.shape[1]):
            m = valid[i, c] & ((C[c] - valid[i, c]) > 0)
            if m.sum() > 50:
                out[i, c] = np.corrcoef(X[i, c, m], others[c, m])[0, 1]
    return out


def movie_sets(task):
    ids, Z = load_movie(task)
    std = np.stack([z['feats'] for z in Z]).astype(float)
    band = (Z[0]['freqs'] >= 1) & (Z[0]['freqs'] <= 45)
    full = np.stack([z['log_psd'][:, band].ravel() for z in Z]).astype(float)
    lab = np.stack([z['labram'] for z in Z]).astype(float)
    isc_bb = isc([z['sig50'] for z in Z])
    isc_a = isc([z['env_alpha'] for z in Z])
    isc_t = isc([z['env_theta'] for z in Z])
    iscf = np.hstack([isc_bb, isc_a, isc_t])
    frac_bad = np.array([float(z['frac_bad']) for z in Z])
    return ids, dict(std=std, full=full, labram=lab, isc=iscf), np.nanmean(isc_bb, axis=1), frac_bad


def load_rest():
    files = sorted(glob.glob(os.path.join(REST, 'sub-*.npz')))
    ids = [os.path.basename(f)[:-4] for f in files]
    Z = [np.load(f) for f in files]
    band = (Z[0]['freqs'] >= 1) & (Z[0]['freqs'] <= 45)
    return ids, dict(std=np.stack([z['feats'] for z in Z]).astype(float),
                     full=np.stack([z['log_psd'][:, band].ravel() for z in Z]).astype(float),
                     labram=np.stack([z['labram'] for z in Z]).astype(float))


def impute(X):
    X = np.where(np.isfinite(X), X, np.nan)
    return np.where(np.isnan(X), np.nanmedian(X, axis=0), X)


def clip5(X): return np.clip(X, -5, 5)
def ridge(): return make_pipeline(RobustScaler(), FunctionTransformer(clip5), StandardScaler(), RidgeCV(alphas=ALPHAS))
def r2(yt, yp, ym): return 1 - ((yt - yp) ** 2).sum() / ((yt - ym) ** 2).sum()
def pred(m, X, ytr): return np.clip(m.predict(X), ytr.min(), ytr.max())


def nested(X, y):
    """X is (n, features) or (n, layers, features). Returns 25 fold R^2 values."""
    layered = X.ndim == 3
    get = (lambda l: X[:, l]) if layered else (lambda l: X)
    nl = X.shape[1] if layered else 1
    scores = []
    for r in range(5):
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
            best = 0
            if nl > 1:
                sc = np.zeros(nl)
                for a, b in KFold(3, shuffle=True, random_state=100 * r + k).split(tr):
                    for l in range(nl):
                        m = ridge().fit(get(l)[tr[a]], y[tr[a]])
                        sc[l] += r2(y[tr[b]], pred(m, get(l)[tr[b]], y[tr[a]]), y[tr[a]].mean())
                best = int(np.argmax(sc))
            m = ridge().fit(get(best)[tr], y[tr])
            scores.append(r2(y[te], pred(m, get(best)[te], y[tr]), y[tr].mean()))
    return np.array(scores)


def corrected_p(d, test_frac=0.2):
    d = np.asarray(d); J = len(d)
    t = d.mean() / np.sqrt((1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1) + 1e-30)
    return float(2 * tdist.sf(abs(t), J - 1))


rest_ids, rest = load_rest()
pres_ids, pres, pres_isc_mean, pres_bad = movie_sets('ThePresent')
HAVE_DM = bool(glob.glob(os.path.join(MOVIE, '*_DespicableMe.npz')))
dm_ids, dm, dm_isc_mean, dm_bad = movie_sets('DespicableMe') if HAVE_DM else ([], {}, None, None)

common = sorted(set(rest_ids) & set(pres_ids))
ri = [rest_ids.index(s) for s in common]; pi = [pres_ids.index(s) for s in common]
print(f'{len(rest_ids)} rest, {len(pres_ids)} The Present, {len(dm_ids)} Despicable Me, {len(common)} with rest + The Present', flush=True)

SETS = {
    'rest: standard': impute(rest['std'][ri]),
    'rest: full spectrum': impute(rest['full'][ri]),
    'rest: LaBraM': rest['labram'][ri],
    'movie: standard': impute(pres['std'][pi]),
    'movie: full spectrum': impute(pres['full'][pi]),
    'movie: LaBraM': pres['labram'][pi],
    'movie: synchrony (ISC)': impute(pres['isc'][pi]),
    'movie: full spectrum + ISC': np.hstack([impute(pres['full'][pi]), impute(pres['isc'][pi])]),
}
meta = parts.loc[common]
results, folds = {}, {}
for target in TARGETS:
    y_all = meta[target].to_numpy(float); ok = ~np.isnan(y_all); y = y_all[ok]
    results[target] = {'n': int(ok.sum())}
    for name, X in SETS.items():
        s = nested(X[ok], y)
        folds[(target, name)] = s
        results[target][name] = {'r2': round(float(s.mean()), 3), 'sd': round(float(s.std()), 3),
                                 'p_vs_0': round(corrected_p(s), 4)}
    for a, b in [('movie: full spectrum', 'rest: full spectrum'), ('movie: LaBraM', 'rest: LaBraM'),
                 ('movie: full spectrum + ISC', 'movie: full spectrum')]:
        d = folds[(target, a)] - folds[(target, b)]
        results[target][f'{a} vs {b}'] = {'diff': round(float(d.mean()), 3), 'folds_better': int((d > 0).sum()),
                                          'p': round(corrected_p(d), 4)}
    print(target, 'done', flush=True)

# Replication with Despicable Me: ISC and full spectrum only, on its own children
dmeta = parts.loc[dm_ids]
rep = {}
for target in (TARGETS if HAVE_DM else []):
    y_all = dmeta[target].to_numpy(float); ok = ~np.isnan(y_all); y = y_all[ok]
    rep[target] = {'n': int(ok.sum())}
    for name, X in {'full spectrum': impute(dm['full']), 'synchrony (ISC)': impute(dm['isc'])}.items():
        s = nested(X[ok], y)
        rep[target][name] = {'r2': round(float(s.mean()), 3), 'sd': round(float(s.std()), 3), 'p_vs_0': round(corrected_p(s), 4)}

# Simple associations: mean broadband ISC vs age and CBCL (partialling out age and data quality)
def partial_spearman(x, y, covs):
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(covs).any(1)
    rx = x[ok] - LinearRegression().fit(covs[ok], x[ok]).predict(covs[ok])
    ry = y[ok] - LinearRegression().fit(covs[ok], y[ok]).predict(covs[ok])
    rho, p = spearmanr(rx, ry)
    return round(float(rho), 3), round(float(p), 4), int(ok.sum())

assoc = {}
for task, ids, iscm, bad in [('ThePresent', pres_ids, pres_isc_mean, pres_bad)] + ([('DespicableMe', dm_ids, dm_isc_mean, dm_bad)] if HAVE_DM else []):
    m = parts.loc[ids]
    rho, p = spearmanr(iscm, m['age'], nan_policy='omit')
    assoc[task] = {'mean ISC': round(float(np.nanmean(iscm)), 4),
                   'ISC vs age (spearman)': [round(float(rho), 3), round(float(p), 6)]}
    covs = np.column_stack([m['age'].to_numpy(float), bad])
    for target in TARGETS[1:]:
        assoc[task][f'ISC vs {target} | age, artifacts'] = partial_spearman(iscm, m[target].to_numpy(float), covs)

out = {'main (same children, rest vs The Present)': results, 'replication (Despicable Me)': rep, 'ISC associations': assoc}
json.dump(out, open(os.path.join(MOVIE, 'results_movie.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
