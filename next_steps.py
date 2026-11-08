"""
Next steps for the EEG workload decoder (PhysioNet EEGMAT, 36 people):

  1. Eye-artifact check
     a. ICA: remove components that track the forehead channels (Fp1/Fp2 used as an
        eye-movement proxy, since EEGMAT has no EOG channel), then re-test.
     b. Posterior-only model: use only central/parietal/occipital/temporal channels,
        far from the eyes.
  2. EEGNet (compact CNN for EEG), tested on new people (6-fold, split by person),
     with per-person Euclidean alignment (uses no labels).
  3. Robot simulation: a robot that slows down when the decoder says "high workload".
     Compare a naive robot (acts on every 0.5 decision) with a confidence-aware robot
     (acts only when the calibrated decoder is sure, otherwise moves cautiously).

All evaluation is subject-independent. Results go to results/next_steps/ and MLflow.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import eeg_workload as ew

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "next_steps")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() - 1))
NEAR_EYES = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"]


# ============================================================ data with optional ICA
def load_all_ica(use_ica):
    X, y, g, n_removed = [], [], [], []
    import glob, re
    files = sorted(glob.glob(os.path.join(ew.DATA, "Subject*_1.edf")))
    ch = None
    for f1 in files:
        sid = int(re.search(r"Subject(\d+)_1", f1).group(1))
        for label, f in [(0, f1), (1, f1.replace("_1.edf", "_2.edf"))]:
            x, ch, info = ew.load_recording(f)
            if use_ica:
                raw = mne.io.RawArray(x, mne.create_info(ch, ew.SFREQ, "eeg"))
                raw.set_montage("standard_1020")
                ica = mne.preprocessing.ICA(n_components=15, method="fastica", random_state=ew.SEED, max_iter=1000)
                ica.fit(raw)
                bad, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], threshold=3.0)
                ica.exclude = bad[:3]
                n_removed.append(len(ica.exclude))
                x = ica.apply(raw.copy()).get_data()
            ep = ew.windows(x)
            X.append(ep); y += [label] * len(ep); g += [sid] * len(ep)
    return np.concatenate(X), np.array(y), np.array(g), ch, n_removed


def eye_checks():
    X, y, g, ch, _ = load_all_ica(False)
    Xi, yi, gi, _, n_removed = load_all_ica(True)
    assert (y == yi).all() and (g == gi).all()
    keep = [i for i, c in enumerate(ch) if c not in NEAR_EYES]
    rows = []
    for name, feats in [
        ("baseline (all 19 channels)", ew.subject_zscore(ew.band_power(X), g)),
        ("after ICA eye-component removal", ew.subject_zscore(ew.band_power(Xi), g)),
        (f"posterior only ({len(keep)} channels, no frontal)", ew.subject_zscore(ew.band_power(X[:, keep]), g)),
        ("ICA + posterior only", ew.subject_zscore(ew.band_power(Xi[:, keep]), g)),
    ]:
        r = ew.evaluate(ew.model_bandpower, feats, y, g, "loso")
        rows.append({"setting": name, "bal_acc": r["bal_acc"], "auc": r["auc"]})
        print("[eye]", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "eye_artifact_check.csv"), index=False)
    ica_info = {"mean_components_removed": float(np.mean(n_removed)),
                "recordings_with_any_removed": int(np.sum(np.array(n_removed) > 0)), "n_recordings": len(n_removed)}
    print("[eye] ICA", ica_info, flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    b = ax.barh(df["setting"][::-1], 100 * df["bal_acc"][::-1], color=["#12343B", "#2E8A8A", "#7FA9A9", "#C0573E"][::-1])
    ax.bar_label(b, fmt="%.1f", padding=3); ax.set_xlim(50, 85); ax.axvline(50, ls="--", c="grey", lw=1)
    ax.set_xlabel("Balanced accuracy on new people (%)"); ax.set_title("Is the decoder reading the brain or the eyes?")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eye_artifact_check.png"), dpi=150); plt.close(fig)
    return X, y, g, ch, df, ica_info


# ============================================================ EEGNet
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018)."""
    def __init__(self, n_ch, n_t, F1=8, D=2, F2=16, k=64, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n_feat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_feat, 2)

    def forward(self, x):
        return self.fc(self.net(x))


def euclidean_align(X, g):
    """Whiten each person's windows by the inverse square root of that person's mean covariance (no labels)."""
    Xa = np.empty_like(X)
    for s in np.unique(g):
        m = g == s
        R = np.mean([x @ x.T / x.shape[1] for x in X[m]], 0)
        R = R + 1e-3 * np.trace(R) / len(R) * np.eye(len(R))   # average reference makes R singular: regularize
        Xa[m] = np.einsum("ij,njt->nit", ew._mfun(R, lambda w: np.clip(w, 1e-12, None) ** -0.5), X[m])
    assert np.isfinite(Xa).all()
    return Xa


def run_eegnet(X, y, g, epochs=40, folds=6):
    torch.manual_seed(ew.SEED); np.random.seed(ew.SEED)
    Xa = euclidean_align(X * 1e6, g).astype(np.float32)[:, None]
    prob = np.zeros(len(y)); logits = np.zeros((len(y), 2)); t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(folds).split(Xa, y, g)):
        model = EEGNet(Xa.shape[2], Xa.shape[3])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        xt, yt = torch.tensor(Xa[tr]), torch.tensor(y[tr])
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 64):
                idx = perm[i:i + 64]
                opt.zero_grad(); loss = nn.functional.cross_entropy(model(xt[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            z = model(torch.tensor(Xa[te])).numpy()
        logits[te] = z; prob[te] = torch.softmax(torch.tensor(z), 1)[:, 1].numpy()
        acc = balanced_accuracy_score(y[te], prob[te] >= 0.5)
        print(f"[eegnet] fold {k + 1}/{folds}  people {sorted(set(g[te]))}  bal_acc {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    res = {"bal_acc": balanced_accuracy_score(y, prob >= 0.5), "auc": roc_auc_score(y, prob), "ece": ew.ece(prob, y)}
    print("[eegnet]", res, flush=True)
    # same folds for the band-power model, for a fair comparison
    BPz = ew.subject_zscore(ew.band_power(X), g); pb = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(BPz, y, g):
        pb[te] = ew.model_bandpower().fit(BPz[tr], y[tr]).predict_proba(BPz[te])[:, 1]
    res_bp = {"bal_acc": balanced_accuracy_score(y, pb >= 0.5), "auc": roc_auc_score(y, pb), "ece": ew.ece(pb, y)}
    print("[eegnet] band-power on the same folds", res_bp, flush=True)
    return res, res_bp


# ============================================================ robot simulation
SPEED = {"normal": 1.0, "cautious": 0.7, "slow": 0.4}


def robot_policies(p):
    naive = np.where(p >= 0.5, SPEED["slow"], SPEED["normal"])
    aware = np.select([p >= 0.8, p <= 0.2], [SPEED["slow"], SPEED["normal"]], SPEED["cautious"])
    # smoothed confidence-aware robot: average the last 3 decoder outputs (6 s) before acting
    ps = pd.Series(p).rolling(3, min_periods=1).mean().values
    smooth = np.select([ps >= 0.8, ps <= 0.2], [SPEED["slow"], SPEED["normal"]], SPEED["cautious"])
    return {"naive (acts on every decision)": naive, "confidence-aware": aware, "confidence-aware + smoothing": smooth}


def simulate_robot(X, y, g):
    BPz = ew.subject_zscore(ew.band_power(X), g)
    r = ew.evaluate(ew.model_bandpower, BPz, y, g, "loso", calibrate=True)
    p_all = r["prob_cal"]
    rows = []
    sessions = {}
    for s in np.unique(g):          # each session: 60 s rest, then 60 s arithmetic (windows are in time order)
        m = g == s
        sessions[s] = (p_all[m], y[m])
    for pol in robot_policies(np.array([0.5])).keys():
        unsafe, unneeded, switches, minutes, speed_task, speed_rest = [], [], [], 0, [], []
        for s, (p, yy) in sessions.items():
            v = robot_policies(p)[pol]
            unsafe.append(np.mean(v[yy == 1] == SPEED["normal"]))       # full speed while the person is overloaded
            unneeded.append(np.mean(v[yy == 0] == SPEED["slow"]))       # slowed down for no reason
            switches.append(np.sum(np.diff(v) != 0)); minutes += len(v) * ew.STEP_S / 60
            speed_task.append(v[yy == 1].mean()); speed_rest.append(v[yy == 0].mean())
        rows.append({"robot": pol,
                     "full_speed_while_overloaded_%": 100 * np.mean(unsafe),
                     "slowed_down_for_no_reason_%": 100 * np.mean(unneeded),
                     "speed_changes_per_min": np.sum(switches) / minutes,
                     "mean_speed_rest": np.mean(speed_rest), "mean_speed_task": np.mean(speed_task)})
        print("[robot]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "robot_simulation.csv"), index=False)

    # timeline figure for one typical person (median accuracy)
    acc = {s: np.mean((p >= 0.5) == yy) for s, (p, yy) in sessions.items()}
    s_med = sorted(acc, key=acc.get)[len(acc) // 2]
    p, yy = sessions[s_med]; t = np.arange(len(p)) * ew.STEP_S
    pol = robot_policies(p)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].axvspan(t[yy == 1][0], t[-1] + ew.STEP_S, color="#E8A33D", alpha=0.15, label="mental arithmetic")
    axes[0].plot(t, p, "o-", color="#12343B", ms=3, label="decoder: P(high workload)")
    axes[0].axhline(0.8, ls=":", c="#C0573E"); axes[0].axhline(0.2, ls=":", c="#2E8A8A"); axes[0].axhline(0.5, ls="--", c="grey", lw=0.8)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Probability"); axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(f"Robot speed follows the decoder (person {s_med}, typical accuracy {100 * acc[s_med]:.0f}%)")
    axes[1].axvspan(t[yy == 1][0], t[-1] + ew.STEP_S, color="#E8A33D", alpha=0.15)
    axes[1].step(t, pol["naive (acts on every decision)"], where="post", color="#C0573E", label="naive robot")
    axes[1].step(t, pol["confidence-aware + smoothing"], where="post", color="#2E8A8A", lw=2, label="confidence-aware robot")
    axes[1].set_yticks([0.4, 0.7, 1.0]); axes[1].set_yticklabels(["slow", "cautious", "normal"])
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Robot speed"); axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "robot_simulation.png"), dpi=150); plt.close(fig)
    return df


# ============================================================ main
def main():
    import sys
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("eeg-workload-eegmat")
    if "--eegnet-only" in sys.argv:
        X, y, g, ch, _ = ew.load_all()
        return eegnet_and_log(X, y, g)

    X, y, g, ch, eye_df, ica_info = eye_checks()
    with mlflow.start_run(run_name="eye_artifact_check"):
        mlflow.log_params({"ica_components": 15, "eog_proxy": "Fp1,Fp2", "max_removed": 3, **ica_info})
        for _, r in eye_df.iterrows():
            key = r["setting"].split(" (")[0].replace(" ", "_").replace("+", "plus")
            mlflow.log_metric("bal_acc_" + key, float(r["bal_acc"]))
        mlflow.log_artifact(os.path.join(OUT, "eye_artifact_check.csv")); mlflow.log_artifact(os.path.join(OUT, "eye_artifact_check.png"))

    robot = simulate_robot(X, y, g)
    with mlflow.start_run(run_name="robot_simulation"):
        mlflow.log_params({"slow_if_p_at_least": 0.8, "normal_if_p_at_most": 0.2, "smoothing_windows": 3, "decoder": "bandpower_subjz_loso_calibrated"})
        for _, r in robot.iterrows():
            tag = r["robot"].split(" (")[0].replace(" ", "_").replace("+", "plus")
            mlflow.log_metrics({f"{tag}_unsafe_pct": float(r["full_speed_while_overloaded_%"]),
                                f"{tag}_unneeded_pct": float(r["slowed_down_for_no_reason_%"]),
                                f"{tag}_changes_per_min": float(r["speed_changes_per_min"])})
        mlflow.log_artifact(os.path.join(OUT, "robot_simulation.csv")); mlflow.log_artifact(os.path.join(OUT, "robot_simulation.png"))
    eegnet_and_log(X, y, g)


def eegnet_and_log(X, y, g):
    res, res_bp = run_eegnet(X, y, g)
    pd.DataFrame([{"model": "EEGNet-8,2 + Euclidean alignment", **res}, {"model": "band power + per-person norm.", **res_bp}]) \
        .to_csv(os.path.join(OUT, "eegnet_vs_bandpower.csv"), index=False)
    with mlflow.start_run(run_name="eegnet_groupkfold6"):
        mlflow.log_params({"model": "EEGNet-8,2", "alignment": "Euclidean (per person)", "epochs": 40, "folds": "GroupKFold(6) by person", "lr": 1e-3})
        mlflow.log_metrics({k: float(v) for k, v in res.items()})
        mlflow.log_metrics({"bandpower_same_folds_" + k: float(v) for k, v in res_bp.items()})
    print("done")


if __name__ == "__main__":
    main()
