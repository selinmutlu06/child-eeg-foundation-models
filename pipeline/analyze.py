"""Compare standard EEG features with LaBraM embeddings on open HBN targets (version 2).

Changes from version 1:
- The LaBraM layer is chosen inside each training set (nested cross-validation),
  so the test children never influence which layer is used.
- Robustness compares the two models' predictions on the same held-out children,
  which works for both interpretable features and arbitrary embedding dimensions.
- Paired comparison: on every test fold, how much better one feature set does than another.

Usage: python analyze.py /path/to/out_dir /path/to/participants.tsv [more participants.tsv ...]
Writes results.csv, layer_results.csv, paired_results.csv and two figures into out_dir.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import hbn_pipeline as hp

out_dir = sys.argv[1]
parts = pd.concat([pd.read_csv(p, sep="\t") for p in sys.argv[2:]])
parts = parts.drop_duplicates("participant_id").set_index("participant_id")

# ------------------------------------------------------------ load per-child results
files = sorted(glob.glob(os.path.join(out_dir, "sub-*.npz")))
sub_ids = [os.path.basename(f)[:-4] for f in files]
data = [np.load(f) for f in files]
meta = parts.loc[sub_ids]

X_std = np.stack([z["feats"] for z in data])
X_std = np.where(np.isnan(X_std), np.nanmedian(X_std, axis=0), X_std)   # alpha peak can be missing
FEATURE_SETS = {
    "Standard features": X_std[:, None, :],               # one "layer"
    "LaBraM pretrained": np.stack([z["labram"] for z in data]),
    "LaBraM random weights": np.stack([z["labram_random"] for z in data]),
}
TARGETS = {"age": "Age (years)", "attention": "Attention factor", "p_factor": "p-factor"}
print(f"{len(files)} children, {X_std.shape[1]} standard features, "
      f"LaBraM {FEATURE_SETS['LaBraM pretrained'].shape[1]} layers")


# ------------------------------------------------------------ helpers
def r2_score(y_true, y_pred, y_train_mean):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_train_mean) ** 2).sum()


def predict(model, X, y_train):
    """Predict, but never outside the range of the training targets.

    One unusual recording can push a linear model to absurd values (a 10-year-old
    predicted as 32). Capping at the training range is standard in brain-age work
    and is applied identically to every feature set.
    """
    return np.clip(model.predict(X), y_train.min(), y_train.max())


def fit_select(XL, y, seed=0):
    """Pick the best layer by 3-fold CV on this training set only, then fit on it."""
    n_layers = XL.shape[1]
    if n_layers == 1:
        best = 0
    else:
        scores = np.zeros(n_layers)
        for tr, va in KFold(3, shuffle=True, random_state=seed).split(y):
            for layer in range(n_layers):
                m = hp.ridge().fit(XL[tr, layer], y[tr])
                scores[layer] += r2_score(y[va], predict(m, XL[va, layer], y[tr]), y[tr].mean())
        best = int(np.argmax(scores))
    return best, hp.ridge().fit(XL[:, best], y)


def nested_cv(XL, y, n_splits=5, n_repeats=5):
    """Outer repeated K-fold; layer chosen inside each outer training set."""
    scores, layers = [], []
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=r).split(y)):
            layer, m = fit_select(XL[tr], y[tr], seed=100 * r + k)
            scores.append(r2_score(y[te], predict(m, XL[te, layer], y[tr]), y[tr].mean()))
            layers.append(layer + 1)
    return np.array(scores), layers


def prediction_robustness(XL, y, n_repeats=10):
    """Split children into thirds. Train one model on each of two thirds, and compare
    their predictions on the remaining third. Rotate so each third is the test set once.

    P = mean held-out R^2 of the two models, A = Spearman agreement of their predictions,
    R = mean of max(P, 0) * A. Unlike coefficient similarity, this does not depend on
    what individual feature dimensions mean, so it is fair to embeddings.
    """
    rng = np.random.default_rng(0)
    P, A = [], []
    for rep in range(n_repeats):
        thirds = np.array_split(rng.permutation(len(y)), 3)
        for t in range(3):
            te, a, b = thirds[t], thirds[(t + 1) % 3], thirds[(t + 2) % 3]
            preds, perf = [], []
            for i, tr in enumerate((a, b)):
                layer, m = fit_select(XL[tr], y[tr], seed=rep * 10 + t * 2 + i)
                p = predict(m, XL[te, layer], y[tr])
                preds.append(p)
                perf.append(r2_score(y[te], p, y[tr].mean()))
            P.append(np.mean(perf))
            A.append(spearmanr(preds[0], preds[1]).statistic)
    P, A = np.array(P), np.nan_to_num(np.array(A))
    return {"P": P.mean(), "A": A.mean(), "R": (np.clip(P, 0, None) * A).mean()}


# ------------------------------------------------------------ per-layer curve (descriptive)
# Every layer scored separately. This is for the figure only: picking the best layer
# from this curve would reuse the test children, which the nested results avoid.
layer_rows = []
for target in TARGETS:
    y = meta[target].to_numpy(float)
    ok = ~np.isnan(y)
    for label in ("LaBraM pretrained", "LaBraM random weights"):
        for layer in range(FEATURE_SETS[label].shape[1]):
            r2, sd = hp.cv_r2(FEATURE_SETS[label][ok, layer], y[ok])
            layer_rows.append({"target": target, "features": label,
                               "layer": layer + 1, "r2": r2, "sd": sd})
layers_df = pd.DataFrame(layer_rows)
layers_df.to_csv(os.path.join(out_dir, "layer_results.csv"), index=False)

# ------------------------------------------------------------ headline comparison
results, fold_scores = [], {}
for target in TARGETS:
    y = meta[target].to_numpy(float)
    ok = ~np.isnan(y)
    for label, XL in FEATURE_SETS.items():
        scores, chosen = nested_cv(XL[ok], y[ok])
        fold_scores[(target, label)] = scores
        rob = prediction_robustness(XL[ok], y[ok])
        common = pd.Series(chosen).value_counts()
        results.append({
            "target": target, "features": label, "n": int(ok.sum()),
            "nested_r2": scores.mean(), "r2_sd": scores.std(), "r2_median": np.median(scores),
            "layer_chosen": "" if label == "Standard features"
                            else f"{common.index[0]} ({common.iloc[0]}/{len(chosen)} folds)",
            "robust_P": rob["P"], "prediction_agreement_A": rob["A"], "robustness_R": rob["R"]})
results = pd.DataFrame(results)
results.to_csv(os.path.join(out_dir, "results.csv"), index=False)

# Paired comparison: the same test children in every fold, so differences are fair.
paired = []
for target in TARGETS:
    for a, b in (("LaBraM pretrained", "Standard features"),
                 ("LaBraM pretrained", "LaBraM random weights"),
                 ("LaBraM random weights", "Standard features")):
        d = fold_scores[(target, a)] - fold_scores[(target, b)]
        paired.append({"target": target, "comparison": f"{a} minus {b}",
                       "mean_r2_difference": d.mean(), "median": np.median(d), "sd": d.std(),
                       "folds_better": f"{(d > 0).sum()}/{len(d)}"})
paired = pd.DataFrame(paired)
paired.to_csv(os.path.join(out_dir, "paired_results.csv"), index=False)

pd.set_option("display.width", 200)
print("\nNested cross-validation and prediction robustness")
print(results.round(3).to_string(index=False))
print("\nPaired differences in R² (same test children)")
print(paired.round(3).to_string(index=False))

# ------------------------------------------------------------ figures
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: age R^2 by layer, with the nested standard-feature level as a reference line
fig, ax = plt.subplots(figsize=(7, 4))
age = layers_df.query("target == 'age'")
for label, color in (("LaBraM pretrained", BLUE), ("LaBraM random weights", ORANGE)):
    d = age.query("features == @label")
    ax.plot(d.layer, d.r2, color=color, lw=2, marker="o", ms=5, label=label)
std_age = results.query("target == 'age' and features == 'Standard features'").nested_r2.iloc[0]
ax.axhline(std_age, color=INK, lw=1.2, ls="--", label="Standard features")
ax.set_xlabel("LaBraM layer")
ax.set_ylabel("Cross-validated R², predicting age")
ax.set_xticks(range(1, 13))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="lower left")
ax.set_title(f"Predicting age from resting EEG, by LaBraM layer ({len(files)} HBN children)",
             loc="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "fig1_age_by_layer.png"), dpi=200)

# Figure 2: nested R^2 (with fold-to-fold spread) and prediction robustness, by target
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels = list(FEATURE_SETS)
colors = [INK, BLUE, ORANGE]
x = np.arange(len(TARGETS))
w = 0.26
for ax, metric, err, title in ((axes[0], "nested_r2", "r2_sd", "Nested cross-validated R²"),
                               (axes[1], "robustness_R", None, "Prediction robustness R")):
    for i, label in enumerate(labels):
        sub = results.set_index(["target", "features"])
        vals = [sub.loc[(t, label), metric] for t in TARGETS]
        errs = [sub.loc[(t, label), err] for t in TARGETS] if err else None
        ax.bar(x + (i - 1) * w, vals, w - 0.02, color=colors[i], label=label,
               yerr=errs, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, list(TARGETS.values()))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK)
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "fig2_comparison.png"), dpi=200)
print("\nsaved figures")
