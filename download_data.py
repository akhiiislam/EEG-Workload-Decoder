"""Download PhysioNet EEGMAT v1.0.0 (EEG During Mental Arithmetic Tasks, ODC-By 1.0) into data/eegmat/."""
import os
import urllib.request

BASE = "https://physionet.org/files/eegmat/1.0.0/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eegmat")
os.makedirs(OUT, exist_ok=True)
files = ["subject-info.csv"] + [f"Subject{i:02d}_{k}.edf" for i in range(36) for k in (1, 2)]
for f in files:
    dst = os.path.join(OUT, f)
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        urllib.request.urlretrieve(BASE + f, dst)   # follows PhysioNet's redirect to its S3 mirror
        print("downloaded", f)
print("done:", len(files), "files in", OUT)
