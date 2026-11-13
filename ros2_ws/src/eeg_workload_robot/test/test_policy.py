"""Check that the ROS policies reproduce the offline robot simulation (results/next_steps/robot_simulation.csv).

Runs with plain Python, no ROS needed:  python -m pytest test/test_policy.py
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eeg_workload_robot.policy import SessionStats, SpeedPolicy  # noqa: E402

EXPECTED = {  # from the offline simulation in the EEG project
    "naive": (24.81, 24.81, 7.41),
    "aware": (6.13, 6.13, 11.41),
    "smooth": (1.72, 1.34, 4.80),
}


def load_sessions():
    sess = defaultdict(list)
    with open(os.path.join(os.path.dirname(HERE), "data", "decoder_stream.csv"), newline="") as f:
        for r in csv.DictReader(f):
            sess[int(r["subject"])].append((int(r["window"]), int(r["label"]), float(r["p_high_workload"])))
    return {s: sorted(v) for s, v in sess.items()}


def run(policy_name):
    unsafe, unneeded, switches, windows = [], [], 0, 0
    for rows in load_sessions().values():
        pol, st = SpeedPolicy(policy_name), SessionStats()
        for _, label, p in rows:
            mode, _ = pol.step(p)
            st.update(mode, label)
        unsafe.append(100 * st.unsafe / st.n_task)
        unneeded.append(100 * st.unneeded / st.n_rest)
        switches += st.switches
        windows += st.n
    n = len(unsafe)
    return sum(unsafe) / n, sum(unneeded) / n, switches / (windows * 2.0 / 60.0)


def test_policy_modes():
    assert SpeedPolicy("naive").step(0.6)[0] == "slow"
    assert SpeedPolicy("aware").step(0.6)[0] == "cautious"
    assert SpeedPolicy("aware").step(0.1)[0] == "normal"
    s = SpeedPolicy("smooth")
    assert [s.step(p)[0] for p in (0.9, 0.9, 0.3)] == ["slow", "slow", "cautious"]


def test_matches_offline_simulation():
    for name, (u, n, sw) in EXPECTED.items():
        got = run(name)
        assert abs(got[0] - u) < 0.05 and abs(got[1] - n) < 0.05 and abs(got[2] - sw) < 0.05, (name, got)


if __name__ == "__main__":
    for name in EXPECTED:
        print(name, [round(v, 2) for v in run(name)])
