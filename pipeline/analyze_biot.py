"""Does BIOT's leftover age signal come from its bipolar input?

Builds three baselines from the same four windows per child:
  F  full referential spectrum (19 channels x 89 bins, 1-45 Hz at 0.5 Hz)
  B  spectrum of BIOT's 18 bipolar channels (18 x 89)
  C  magnitude-squared coherence between the two electrodes of each bipolar pair (18 x 89)
Then asks whether BIOT (pretrained and random weights) still predicts age after removing F,
after removing B, and after removing F+B+C together.
Usage: python analyze_biot.py WINDOW_DIR FM_DIR RAW_DIR OUT_JSON
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.signal import csd, welch
from sklearn.model_selection import KFold


WIN, FM, RAW, OUT = sys.argv[1:5]
import importlib.util
spec = importlib.util.spec_from_file_location('afm_funcs', os.path.join(os.path.dirname(__file__), 'analyze_fm_funcs.py'))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
from fm_adapters import BIPOLAR_18, IDX

parts = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(os.path.join(RAW, 'participants_R*.tsv'))])
parts = parts.drop_duplicates('participant_id').set_index('participant_id')


def feats(sub):
    w = np.load(os.path.join(WIN, f'{sub}.npy'))[:4].astype(float)
    f, p = welch(w, fs=200, nperseg=400, axis=-1)
    band = (f >= 1) & (f <= 45)
    F = np.log10(p.mean(0)[:, band] + 1e-30)
    b = np.stack([w[:, IDX[a]] - w[:, IDX[c]] for a, c in BIPOLAR_18], axis=1)
    _, pb = welch(b, fs=200, nperseg=400, axis=-1)
    B = np.log10(pb.mean(0)[:, band] + 1e-30)
    C = []
    for a, c in BIPOLAR_18:
        x, y = w[:, IDX[a]], w[:, IDX[c]]
        _, sxy = csd(x, y, fs=200, nperseg=400, axis=-1)
        sxx, syy = p[:, IDX[a]], p[:, IDX[c]]
        coh = np.abs(sxy.mean(0)) ** 2 / (sxx.mean(0) * syy.mean(0) + 1e-40)
        C.append(np.arctanh(np.sqrt(np.clip(coh[band], 0, 0.9999))))
    return F.ravel(), B.ravel(), np.ravel(C)


z = np.load(os.path.join(FM, 'biot_pretrained.npz'), allow_pickle=True)
ids = list(z['ids']); E = np.nan_to_num(z['emb'].astype(float))
zr = np.load(os.path.join(FM, 'biot_random.npz'), allow_pickle=True)
assert list(zr['ids']) == ids
R = np.nan_to_num(zr['emb'].astype(float))
FBC = [feats(s) for s in ids]
F = np.stack([x[0] for x in FBC]); B = np.stack([x[1] for x in FBC]); C = np.stack([x[2] for x in FBC])
X = np.hstack([F, B, C])
y = parts.loc[ids, 'age'].to_numpy(float)
print('features', F.shape, B.shape, C.shape, flush=True)

conds = {
    'full spectrum (F)': (lambda l: F, 1),
    'bipolar spectrum (B)': (lambda l: B, 1),
    'coherence (C)': (lambda l: C, 1),
    'F+B+C': (lambda l: X, 1),
    'BIOT': (lambda l: E[:, l], E.shape[1]),
    'BIOT + F+B+C': (lambda l: np.hstack([X, E[:, l]]), E.shape[1]),
}
resid = {'BIOT minus F': (E, F), 'BIOT minus B': (E, B), 'BIOT minus F+B+C': (E, X),
         'random BIOT minus F': (R, F), 'random BIOT minus F+B+C': (R, X)}
rows = {k: [] for k in list(conds) + list(resid)}
for r in range(5):
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(y)):
        s = 100 * r + k
        for name, (get, nl) in conds.items():
            rows[name].append(A.run(get, nl, y, tr, te, s)[0])
        for name, (emb, src) in resid.items():
            res = [A.residualize(emb[:, l], src, tr) for l in range(emb.shape[1])]
            rows[name].append(A.run(lambda l: res[l], len(res), y, tr, te, s)[0])
    print('repeat', r, 'done', flush=True)

out = {}
for k, v in rows.items():
    v = np.array(v)
    out[k] = {'r2': round(float(v.mean()), 3), 'sd': round(float(v.std()), 3), 'folds_above_0': int((v > 0).sum()), 'p_vs_0': round(A.corrected_p(v), 4)}
def paired(a, b):
    d = np.array(rows[a]) - np.array(rows[b])
    return {'diff': round(float(d.mean()), 3), 'folds_better': int((d > 0).sum()), 'p': round(A.corrected_p(d), 4)}
out['paired'] = {'BIOT vs F': paired('BIOT', 'full spectrum (F)'), 'BIOT vs F+B+C': paired('BIOT', 'F+B+C'),
                 'F+B+C vs F': paired('F+B+C', 'full spectrum (F)'), 'B vs F': paired('bipolar spectrum (B)', 'full spectrum (F)'),
                 'BIOT + F+B+C vs F+B+C': paired('BIOT + F+B+C', 'F+B+C'),
                 'BIOT minus F vs random BIOT minus F': paired('BIOT minus F', 'random BIOT minus F')}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
