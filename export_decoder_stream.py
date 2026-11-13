"""Export the calibrated leave-one-subject-out decoder outputs as a time-ordered stream.

Each row is one 4 s window (2 s step) of one person, in recording order: 60 s rest, then 60 s
mental arithmetic. The probability for a person comes from a model that never saw that person
(LOSO), calibrated with temperature scaling fitted on held-out training people.

Output: ros2_ws/src/eeg_workload_robot/data/decoder_stream.csv
Columns: subject, window, t_s, label (0 = rest, 1 = arithmetic), p_high_workload
"""
import os

import numpy as np
import pandas as pd

import eeg_workload as ew

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ros2_ws", "src", "eeg_workload_robot", "data", "decoder_stream.csv")


def main():
    X, y, g, _, _ = ew.load_all()
    BPz = ew.subject_zscore(ew.band_power(X), g)
    r = ew.evaluate(ew.model_bandpower, BPz, y, g, "loso", calibrate=True)
    p = r["prob_cal"]
    rows = []
    for s in np.unique(g):
        idx = np.where(g == s)[0]  # windows are stored in time order
        for k, i in enumerate(idx):
            rows.append({"subject": int(s), "window": k, "t_s": k * ew.STEP_S,
                         "label": int(y[i]), "p_high_workload": round(float(p[i]), 5)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows), "rows; calibrated balanced accuracy",
          round(r["bal_acc_calibrated"], 4), "ECE", round(r["ece_calibrated"], 4))


if __name__ == "__main__":
    main()
