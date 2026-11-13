"""Robot speed policies driven by the EEG workload decoder (pure Python, no ROS needed).

The decoder outputs p = P(high workload) every 2 s. A policy turns p into a robot speed mode.
These are the same rules used in next_steps.py of the EEG project, so results can be compared.
"""
from collections import deque

SPEED = {"normal": 1.0, "cautious": 0.7, "slow": 0.4}
POLICIES = ("naive", "aware", "smooth")


class SpeedPolicy:
    """naive  : slow if p >= 0.5, otherwise normal (acts on every decision)
    aware  : slow if p >= 0.8, normal if p <= 0.2, otherwise cautious
    smooth : like 'aware', but on the mean of the last `window` outputs (default 3 = 6 s)
    """

    def __init__(self, name="smooth", slow_at=0.8, normal_at=0.2, window=3):
        if name not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {name!r}")
        self.name, self.slow_at, self.normal_at = name, slow_at, normal_at
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def step(self, p):
        """Return (mode, speed_scale) for a new decoder probability p."""
        self.history.append(float(p))
        if self.name == "naive":
            mode = "slow" if p >= 0.5 else "normal"
        else:
            q = sum(self.history) / len(self.history) if self.name == "smooth" else p
            mode = "slow" if q >= self.slow_at else "normal" if q <= self.normal_at else "cautious"
        return mode, SPEED[mode]


class SessionStats:
    """Counts the same safety measures as the offline simulation."""

    def __init__(self):
        self.n_task = self.n_rest = self.unsafe = self.unneeded = self.switches = self.n = 0
        self.last_mode = None

    def update(self, mode, label):
        self.n += 1
        if self.last_mode is not None and mode != self.last_mode:
            self.switches += 1
        self.last_mode = mode
        if label == 1:
            self.n_task += 1
            self.unsafe += mode == "normal"      # full speed while the person is overloaded
        elif label == 0:
            self.n_rest += 1
            self.unneeded += mode == "slow"      # slowed down for no reason

    def summary(self, step_s=2.0):
        pct = lambda a, b: 100.0 * a / b if b else float("nan")
        minutes = self.n * step_s / 60.0
        return {"windows": self.n,
                "full_speed_while_overloaded_%": round(pct(self.unsafe, self.n_task), 1),
                "slowed_down_for_no_reason_%": round(pct(self.unneeded, self.n_rest), 1),
                "speed_changes_per_min": round(self.switches / minutes, 2) if minutes else float("nan")}
