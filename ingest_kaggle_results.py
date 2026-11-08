"""Log a downloaded Kaggle run (results/ folder) into the local MLflow store.

Usage: .venv\\Scripts\\python.exe ingest_kaggle_results.py <path to downloaded results folder> <kaggle notebook url>
"""
import glob
import json
import os
import sys

import mlflow
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
res_dir, url = sys.argv[1], sys.argv[2]

df = pd.read_csv(os.path.join(res_dir, "results.csv"))
summary = json.load(open(os.path.join(res_dir, "summary.json")))

mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
mlflow.set_experiment("eeg-workload-eegmat")
with mlflow.start_run(run_name="kaggle_run") as parent:
    mlflow.set_tags({"source": "kaggle", "kaggle_notebook": url})
    mlflow.log_params({**summary["params"], "n_subjects": summary["n_subjects"], "n_windows": summary["n_windows"]})
    for _, r in df.iterrows():
        with mlflow.start_run(run_name="kaggle_" + r["config"], nested=True):
            mlflow.log_param("split", r["split"])
            mlflow.log_metrics({k: float(r[k]) for k in df.columns if k not in ("config", "split") and pd.notna(r[k])})
    for f in glob.glob(os.path.join(res_dir, "*")):
        mlflow.log_artifact(f)
print(df.round(4).to_string(index=False))
print("logged run", parent.info.run_id)
