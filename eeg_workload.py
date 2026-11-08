"""
Mental-workload EEG decoder (rest vs. mental arithmetic) on PhysioNet EEGMAT.

Pipeline
  1. Load 36 subjects x 2 recordings (rest = _1, arithmetic = _2), 19 EEG channels.
  2. Clean: band-pass 1-40 Hz, 50 Hz notch, average reference, resample to 128 Hz.
     Use the last 60 s of rest so both classes have 60 s per person.
  3. Cut into 4 s windows with 50% overlap.
  4. Features / models:
       A. log band power (theta, alpha, beta) per channel + logistic regression
       B. Riemannian: covariance -> tangent space -> logistic regression
  5. Honest testing: leave-one-subject-out (LOSO) vs. a leaky window-level split.
  6. Trust checks: calibration (ECE before/after temperature scaling) and an
     artifact test (drop the frontal channels near the eyes).
  7. Explain: head maps of where theta/alpha power changes during the task.

Everything is logged to a local MLflow store (mlflow.db) and written to results/.
"""
import glob
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mne
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import oas
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "eegmat")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

SFREQ = 128
WIN_S, STEP_S, KEEP_S = 4.0, 2.0, 60.0
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
FRONTAL = ["Fp1", "Fp2", "F7", "F8"]
SEED = 42


# ---------------------------------------------------------------- data
def load_recording(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    eeg = [c for c in raw.ch_names if c.startswith("EEG ") and "A2-A1" not in c]
    raw.pick(eeg)
    raw.rename_channels({c: c.replace("EEG ", "").strip() for c in eeg})
    raw.rename_channels({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"})
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.notch_filter(50.0)
    raw.filter(1.0, 40.0)
    raw.set_eeg_reference("average")
    raw.resample(SFREQ)
    x = raw.get_data()
    n_keep = int(KEEP_S * SFREQ)
    x = x[:, -n_keep:]                      # last 60 s (rest is 180 s, task is 60 s)
    return x, raw.ch_names, raw.info


def windows(x):
    w, s = int(WIN_S * SFREQ), int(STEP_S * SFREQ)
    return np.stack([x[:, i:i + w] for i in range(0, x.shape[1] - w + 1, s)])


def load_all():
    X, y, g = [], [], []
    files = sorted(glob.glob(os.path.join(DATA, "Subject*_1.edf")))
    ch_names, info = None, None
    for f1 in files:
        sid = int(re.search(r"Subject(\d+)_1", f1).group(1))
        for label, f in [(0, f1), (1, f1.replace("_1.edf", "_2.edf"))]:
            x, ch_names, info = load_recording(f)
            ep = windows(x)
            X.append(ep); y += [label] * len(ep); g += [sid] * len(ep)
    return np.concatenate(X), np.array(y), np.array(g), ch_names, info


def band_power(X):
    f, p = welch(X, fs=SFREQ, nperseg=SFREQ * 2, axis=-1)
    feats = [np.log(p[..., (f >= lo) & (f < hi)].mean(-1)) for lo, hi in BANDS.values()]
    return np.concatenate(feats, axis=1)          # (n, 3 * n_channels)


# ---------------------------------------------------------------- models
def model_bandpower():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def _mfun(C, fn):
    w, V = eigh(C)
    return (V * fn(w)) @ V.T


class Covariances(BaseEstimator, TransformerMixin):
    """OAS-shrunk spatial covariance of each window (channels x channels)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.stack([oas(x.T)[0] for x in X])


class TangentSpace(BaseEstimator, TransformerMixin):
    """Project covariances to the tangent space at the (log-Euclidean) mean of the TRAINING covariances."""
    def fit(self, C, y=None):
        ref = _mfun(np.mean([_mfun(c, np.log) for c in C], 0), np.exp)
        self.isqrt_ = _mfun(ref, lambda w: w ** -0.5)
        iu = np.triu_indices(C.shape[1])
        self.iu_, self.w_ = iu, np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
        return self

    def transform(self, C):
        return np.stack([_mfun(self.isqrt_ @ c @ self.isqrt_, np.log)[self.iu_] * self.w_ for c in C])


def model_riemann():
    return make_pipeline(Covariances(), TangentSpace(), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


def subject_zscore(F, g):
    """Normalize features within each person (uses no labels): removes person-to-person offsets."""
    F = F.copy()
    for s in np.unique(g):
        m = g == s
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return F


# ---------------------------------------------------------------- calibration
def ece(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    conf = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def fit_temperature(logit, y):
    def nll(logT):
        p = 1 / (1 + np.exp(-logit / np.exp(logT)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


# ---------------------------------------------------------------- evaluation
def evaluate(make_model, X, y, g, split, calibrate=False):
    prob = np.zeros(len(y)); prob_cal = np.zeros(len(y))
    if split == "loso":
        folds = LeaveOneGroupOut().split(X, y, g)
    else:  # leaky: windows from the same person in both train and test
        folds = StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y)
    for tr, te in folds:
        if calibrate:
            fit_i, cal_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X[tr], y[tr], g[tr]))
            m = make_model().fit(X[tr][fit_i], y[tr][fit_i])
            T = fit_temperature(m.decision_function(X[tr][cal_i]), y[tr][cal_i])
            z = m.decision_function(X[te])
            prob[te] = 1 / (1 + np.exp(-z)); prob_cal[te] = 1 / (1 + np.exp(-z / T))
        else:
            m = make_model().fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    per_subj = {int(s): float((pred[g == s] == y[g == s]).mean()) for s in np.unique(g)}
    res = {"bal_acc": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, prob),
           "ece": ece(prob, y), "per_subject": per_subj, "prob": prob}
    if calibrate:
        res["ece_calibrated"] = ece(prob_cal, y)
        res["bal_acc_calibrated"] = balanced_accuracy_score(y, (prob_cal >= 0.5).astype(int))
        res["prob_cal"] = prob_cal
    return res


def reliability_plot(prob, prob_cal, y, path):
    edges = np.linspace(0.5, 1, 6)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for p, lab, c in [(prob, "before", "#C0573E"), (prob_cal, "after temperature scaling", "#2E8A8A")]:
        conf = np.maximum(p, 1 - p); ok = ((p >= 0.5).astype(int) == y)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 5: xs.append(conf[m].mean()); ys.append(ok[m].mean())
        ax.plot(xs, ys, "o-", color=c, label=f"{lab} (ECE {ece(p, y):.3f})")
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Actual accuracy")
    ax.set_title("Is the confidence honest? (LOSO)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    X, y, g, ch, info = load_all()
    n_subj = len(np.unique(g))
    print(f"windows: {len(y)}  subjects: {n_subj}  channels: {len(ch)}  window shape: {X.shape[1:]}")
    BP = band_power(X)
    BPz = subject_zscore(BP, g)
    keep = [i for i, c in enumerate(ch) if c not in FRONTAL]
    BPz_nofront = subject_zscore(band_power(X[:, keep]), g)

    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("eeg-workload-eegmat")

    rows, store = [], {}
    configs = [
        ("bandpower_leaky",      model_bandpower, BP,  "leaky", False),
        ("bandpower_loso",       model_bandpower, BP,  "loso",  False),
        ("bandpower_subjz_loso", model_bandpower, BPz, "loso",  True),
        ("riemann_leaky",        model_riemann,   X,   "leaky", False),
        ("riemann_loso",         model_riemann,   X,   "loso",  False),
        ("bandpower_subjz_loso_no_frontal", model_bandpower, BPz_nofront, "loso", False),
    ]
    for name, mk, feats, split, cal in configs:
        r = evaluate(mk, feats, y, g, split, calibrate=cal)
        store[name] = r
        row = {"config": name, "split": split, "bal_acc": r["bal_acc"], "auc": r["auc"], "ece": r["ece"]}
        if cal:
            row.update(ece_calibrated=r["ece_calibrated"], bal_acc_calibrated=r["bal_acc_calibrated"])
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"dataset": "PhysioNet EEGMAT v1.0.0", "split": split, "window_s": WIN_S,
                               "step_s": STEP_S, "sfreq": SFREQ, "band": "1-40 Hz", "n_subjects": n_subj,
                               "model": name.split("_")[0]})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, float)})

    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    ps = pd.DataFrame({k: store[k]["per_subject"] for k in store}); ps.index.name = "subject"
    ps.to_csv(os.path.join(OUT, "per_subject_accuracy.csv"))

    # figure 1: leaky vs honest
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["bandpower_leaky", "bandpower_loso", "bandpower_subjz_loso", "riemann_leaky", "riemann_loso"]
    labels = ["Band power\nleaky split", "Band power\nnew people", "Band power\n+ per-person norm.\nnew people",
              "Riemannian\nleaky split", "Riemannian\nnew people"]
    vals = [100 * df.set_index("config").loc[o, "bal_acc"] for o in order]
    cols = ["#C0573E", "#2E8A8A", "#12343B", "#C0573E", "#2E8A8A"]
    b = ax.bar(labels, vals, color=cols)
    ax.bar_label(b, fmt="%.1f"); ax.set_ylim(40, 105); ax.axhline(50, ls="--", c="grey", lw=1)
    ax.set_ylabel("Balanced accuracy (%)"); ax.set_title("Workload decoding: leaky split vs. testing on new people")
    plt.xticks(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "leaky_vs_loso.png"), dpi=150); plt.close(fig)

    # figure 2: per-subject accuracy (honest)
    best = "bandpower_subjz_loso"
    acc = ps[best].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(acc)), 100 * acc.values, color="#2E8A8A"); ax.axhline(50, ls="--", c="grey", lw=1)
    ax.set_xticks(range(len(acc))); ax.set_xticklabels(acc.index, fontsize=7)
    ax.set_xlabel("Held-out person"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy for each new person (LOSO)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_subject.png"), dpi=150); plt.close(fig)

    # figure 3: calibration
    reliability_plot(store[best]["prob"], store[best]["prob_cal"], y, os.path.join(OUT, "calibration.png"))

    # figure 4: head maps (task minus rest, log band power)
    nch = len(ch)
    info_eeg = mne.create_info(ch, SFREQ, "eeg"); info_eeg.set_montage("standard_1020")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for i, (band, ax) in enumerate(zip(BANDS, axes)):
        d = BPz[y == 1, i * nch:(i + 1) * nch].mean(0) - BPz[y == 0, i * nch:(i + 1) * nch].mean(0)
        lim = np.abs(d).max()
        mne.viz.plot_topomap(d, info_eeg, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim))
        ax.set_title(f"{band}: task - rest")
    fig.suptitle("Where brain rhythms change during mental arithmetic (red = more power)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "topomaps.png"), dpi=150); plt.close(fig)

    with mlflow.start_run(run_name="figures"):
        for f in glob.glob(os.path.join(OUT, "*")):
            mlflow.log_artifact(f)
    json.dump({"n_windows": int(len(y)), "n_subjects": int(n_subj), "channels": ch},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
