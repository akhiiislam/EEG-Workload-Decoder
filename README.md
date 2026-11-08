# EEG Mental-Workload Decoder (rest vs. mental arithmetic)

**Data:** PhysioNet *EEG During Mental Arithmetic Tasks* (EEGMAT v1.0.0), 36 people, 19 EEG channels.
https://physionet.org/content/eegmat/1.0.0/ (ODC-By 1.0). The data is not stored in this repository.

**Setup and run:**
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # on Linux/macOS: .venv/bin/pip
.venv/Scripts/python download_data.py            # about 176 MB from PhysioNet
.venv/Scripts/python eeg_workload.py             # main pipeline
.venv/Scripts/python next_steps.py               # eye-artifact check, robot simulation, EEGNet
```

- Results are written to `results/` and logged to MLflow (`mlflow.db`).
- `eeg_workload_kaggle.py` is the same pipeline for a Kaggle notebook. Attach the dataset `windwerfer/physionet-eegmat-mental-arithmetic`.
To view the runs: `.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db`

## Pipeline
1. Band-pass filter 1–40 Hz, 50 Hz notch, average reference, resample to 128 Hz. Use the last 60 s of rest, so both classes have 60 s per person.
2. Cut into 4 s windows with 50% overlap (2,088 windows).
3. Two models:
   - log band power (theta, alpha, beta) with logistic regression;
   - Riemannian (covariance → tangent space → logistic regression).
4. Testing:
   - **leave-one-subject-out** (LOSO, honest);
   - a **leaky** window-level 10-fold split, run for comparison.
5. Trust checks:
   - calibration (ECE, temperature scaling fitted on held-out *training* people);
   - frontal-channel removal test.

## Results (balanced accuracy)
| Setting | Leaky split | New people (LOSO) |
|---|---|---|
| Band power | 78.5% | 65.7% |
| Band power + per-person normalization | — | **75.2%** (AUC 0.835) |
| Riemannian | **94.0%** | 65.1% |

- **Leakage:** the Riemannian model drops from 94.0% to 65.1% once it is tested on new people, so the leaky number is mostly a person-recognition effect.
- **Calibration (best LOSO model):** ECE fell from 0.060 to 0.024 after temperature scaling.
- **Artifact check:** removing Fp1, Fp2, F7 and F8 lowers LOSO accuracy from 75.2% to 69.8%.
  - Part of the signal is frontal: theta rises at the forehead during the task.
  - This could be genuine frontal-midline theta or eye movement, and it needs an EOG/ICA check.
- **Physiology:** alpha power drops over the whole head during arithmetic, the classic sign of mental effort.

## Follow-up experiments (`next_steps.py`, results in `results/next_steps/`)
**1. Eye-artifact check** (band power with per-person normalization, LOSO).

| Setting | Balanced accuracy |
|---|---|
| All 19 channels | 76.1% |
| After ICA eye-component removal (Fp1/Fp2 as EOG proxy) | 74.2% |
| Posterior only (12 channels, no frontal) | 69.2% |
| ICA + posterior only | 70.5% |

- ICA flagged eye components in 17 of 72 recordings.
- Removing them costs only about 2 points, and the model stays near 70% with no frontal sensors.
- Conclusion: the decoder mainly reads brain activity. The frontal channels add a little, which may be genuine frontal theta.
- Note: these numbers fit on all training people, so they are about 1 point higher than the 75.2% in the main table, which holds out 20% of the training people for calibration.

**2. EEGNet-8,2 with per-person Euclidean alignment** (GroupKFold(6) by person, 40 epochs, CPU).

| Model (same folds) | Balanced accuracy | AUC | ECE |
|---|---|---|---|
| EEGNet | 70.3% | 0.771 | 0.196 |
| Band power + per-person norm. | **75.5%** | **0.833** | **0.060** |

With 36 people and 1 minute per condition, the simple model is more accurate and much better calibrated. EEGNet is overconfident, which shows why calibration matters before a robot acts on it.

**3. Robot simulation.**
- **Decoder:** calibrated LOSO probabilities from the band-power model.
- **Naive robot:** slows down whenever p ≥ 0.5.
- **Confidence-aware robot:** slow if p ≥ 0.8, normal if p ≤ 0.2, cautious otherwise. It averages the last 3 windows (6 s) before acting.

| Robot | Full speed while overloaded | Slowed down for no reason | Speed changes / min |
|---|---|---|---|
| Naive | 24.8% | 24.8% | 7.4 |
| Confidence-aware | 6.1% | 6.1% | 11.4 |
| Confidence-aware + smoothing | **1.7%** | **1.3%** | **4.8** |

The equal rates in the first two rows are real, not a bug: per-person normalization makes the rest and task score distributions nearly mirror images. They differ at other thresholds, for example 178 vs 187 at 0.6.

## Remaining ideas
- Riemannian re-centering per person.
- Calibrating EEGNet with temperature scaling.
- Collecting EEG in a real human-robot task.

## Kaggle run
- **Notebook:** https://www.kaggle.com/code/akhiislam/notebook9448d24e9c (private; Version 2 ran successfully in 190 s).
- **Input:** `windwerfer/physionet-eegmat-mental-arithmetic`, read-only.
- **Result:** identical metrics to the local run. They are logged in MLflow as run `kaggle_run`.
- **Riemannian step:** implemented with numpy/scipy, so no extra installs are needed: OAS covariance, then tangent space at the log-Euclidean mean of the training data.
