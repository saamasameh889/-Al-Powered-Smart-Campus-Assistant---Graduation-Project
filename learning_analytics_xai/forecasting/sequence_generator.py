"""
sequence_generator.py — Synthetic GPA Trajectory Generator
═══════════════════════════════════════════════════════════════════════════════
Generates per-semester GPA sequences from a student snapshot (students_summary.csv).

Each student in the snapshot has a single aggregated GPA measurement.
We simulate their full 8-semester academic trajectory using trajectory-pattern
logic calibrated against the risk labels and performance metrics in the data.

Multiple stochastic augmentations per student create a large training corpus
for the LSTM without requiring real longitudinal records.

Public API
----------
    from forecasting.sequence_generator import generate_sequences, STATIC_DIM, TEMPORAL_DIM

    data = generate_sequences(n_augmentations=5, history_len=4, horizon=3)
    # data['static']   → np.ndarray (N, STATIC_DIM)
    # data['temporal'] → np.ndarray (N, history_len, TEMPORAL_DIM)
    # data['target']   → np.ndarray (N, horizon)   — normalised GPA (÷4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Programme constants ─────────────────────────────────────────────────────────
_PROG_DIFFICULTY: dict[str, float] = {
    "CSAI": 0.55, "DSAI": 0.62, "SWE":  0.40,
    "MECH": 0.73, "EEE":  0.80, "CIV":  0.75,
    "MATH": 0.82, "PHYS": 0.80, "CHEM": 0.71,
    "BUS":  0.18, "FIN":  0.22,
}
_PROG_ENC: dict[str, float] = {
    prog: i / 10.0
    for i, prog in enumerate(
        ["CSAI", "DSAI", "SWE", "MECH", "EEE", "CIV", "MATH", "PHYS", "CHEM", "BUS", "FIN"]
    )
}
_SCHOOL_ENC: dict[str, float] = {
    "CS&AI": 0.0, "ENGR": 0.333, "SCI": 0.667, "BUS": 1.0,
}

# Typical per-semester credit load across 8 semesters (increasing then stable)
_SEM_LOADS = [12, 15, 18, 18, 21, 21, 21, 18]

STATIC_DIM   = 7    # programme_enc, school_enc, attendance, final_score, failed_ratio, difficulty, total_credits_norm
TEMPORAL_DIM = 5    # gpa_norm, load_norm, risk_flag, gpa_delta (momentum), cumulative_credits_norm
_MAX_CREDITS = 160.0  # max credits for an 8-semester degree (~20 credits/sem)

_DATA_PATH = Path(__file__).parent.parent / "data" / "students_summary.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  Trajectory simulation
# ══════════════════════════════════════════════════════════════════════════════

def _trajectory_params(row: pd.Series) -> tuple[float, float, str]:
    """
    Return (trend, noise_std, pattern_name) for a student row.

    trend      : expected GPA change per semester (can be negative)
    noise_std  : semester-to-semester GPA volatility
    pattern    : human-readable name (for debugging)
    """
    gpa     = float(row["cumulative_gpa"])
    risk    = str(row["risk_level"])
    failed  = float(row.get("failed_courses", 0))
    att     = float(row.get("avg_attendance", 80))

    if risk == "High Risk" and failed >= 4:
        return -0.18, 0.32, "declining"
    if risk == "High Risk":
        return -0.07, 0.28, "at_risk_volatile"
    if risk == "Medium Risk" and att < 72:
        return -0.03, 0.25, "low_attendance_drift"
    if risk == "Medium Risk":
        return +0.06, 0.20, "gradual_improver"
    if risk == "Low Risk" and gpa >= 3.6:
        return +0.01, 0.12, "high_achiever_stable"
    if risk == "Low Risk" and gpa < 2.7:
        return +0.10, 0.18, "late_bloomer"
    # default: low-risk, average GPA
    return +0.03, 0.16, "steady_performer"


def _simulate_trajectory(
    row: pd.Series,
    n_semesters: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """
    Simulate `n_semesters` of GPA + credit-load for one student.

    Returns
    -------
    gpas  : list of float (n_semesters,)
    loads : list of float (n_semesters,) — normalised credit load (÷24)
    """
    base  = float(row["cumulative_gpa"])
    trend, noise_std, _ = _trajectory_params(row)
    diff  = _PROG_DIFFICULTY.get(str(row.get("programme", "CSAI")), 0.5)

    # Jitter the starting GPA slightly so augmentations differ
    gpa = float(np.clip(base + rng.normal(0, 0.18), 0.5, 4.0))

    gpas, loads = [], []
    for t in range(n_semesters):
        # Core GPA evolution: trend + stochastic noise
        gpa += trend + rng.normal(0, noise_std)

        # Every-other-semester difficulty spike (harder mid-degree courses)
        if t % 2 == 1:
            gpa -= diff * rng.uniform(0.0, 0.07)

        # Occasional risk events (failed batch, personal issues)
        if rng.random() < 0.10:
            gpa -= rng.uniform(0.20, 0.45)

        # Occasional recovery term
        if rng.random() < 0.08:
            gpa += rng.uniform(0.15, 0.35)

        gpa = float(np.clip(gpa, 0.5, 4.0))
        gpas.append(gpa)

        # Credit load: ramp up then plateau, add small noise
        base_load = _SEM_LOADS[min(t, len(_SEM_LOADS) - 1)]
        loads.append(float(np.clip(base_load + rng.integers(-3, 4), 9, 24)))

    return gpas, loads


# ══════════════════════════════════════════════════════════════════════════════
#  Window extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_windows(
    gpas: list[float],
    loads: list[float],
    static_x: list[float],
    history_len: int,
    horizon: int,
) -> tuple[list, list, list]:
    """
    Slide a (history_len + horizon) window over the trajectory.
    Returns (static_list, temporal_list, target_list) for all valid positions.

    Credit accumulation is tracked across the full trajectory so that each
    window knows the running total of credits the student has earned —
    this directly encodes GPA inertia (more credits = harder to shift GPA).
    """
    statics, temporals, targets = [], [], []
    n = len(gpas)
    window = history_len + horizon

    for start in range(n - window + 1):
        hist_gpas   = gpas [start : start + history_len]
        hist_loads  = loads[start : start + history_len]
        fut_gpas    = gpas [start + history_len : start + window]

        # Credits completed before this window (from earlier semesters in trajectory)
        credits_before_window = sum(loads[:start])

        temporal = []
        running_credits = credits_before_window
        for i, (g, lo) in enumerate(zip(hist_gpas, hist_loads)):
            running_credits += lo
            gpa_norm  = g  / 4.0
            load_norm = lo / 24.0
            risk_flag = (
                1.0 if g < 2.0
                else 0.5 if g < 2.5
                else 0.25 if g < 3.0
                else 0.0
            )
            gpa_delta = (hist_gpas[i] - hist_gpas[i - 1]) / 4.0 if i > 0 else 0.0
            # Cumulative credits encodes GPA inertia: high value → hard to shift GPA
            cumulative_credits_norm = running_credits / _MAX_CREDITS
            temporal.append([gpa_norm, load_norm, risk_flag, gpa_delta, cumulative_credits_norm])

        # Total credits at end of history = how inertial GPA is going into forecast
        total_credits_norm = running_credits / _MAX_CREDITS
        static_extended = list(static_x) + [total_credits_norm]

        last_hist_gpa = hist_gpas[-1]
        target = [(g - last_hist_gpa) / 4.0 for g in fut_gpas]  # GPA change from last observed

        statics.append(static_extended)
        temporals.append(temporal)
        targets.append(target)

    return statics, temporals, targets


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_sequences(
    n_augmentations: int = 5,
    history_len: int = 4,
    horizon: int = 3,
    n_semesters: int = 8,
    data_path: Path | str | None = None,
    seed: int = 42,
) -> dict:
    """
    Generate LSTM training sequences from students_summary.csv.

    Each student is augmented `n_augmentations` times with different random
    seeds, producing diverse GPA trajectories that share the same underlying
    risk and performance profile.  All valid (history, future) windows are
    extracted from each trajectory.

    Parameters
    ----------
    n_augmentations : stochastic variants per student
    history_len     : T — number of past semesters observed
    horizon         : H — number of future semesters to predict
    n_semesters     : total length of each simulated trajectory
    data_path       : override default students_summary.csv location
    seed            : master RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        'static'      : np.ndarray (N, STATIC_DIM)
        'temporal'    : np.ndarray (N, history_len, TEMPORAL_DIM)
        'target'      : np.ndarray (N, horizon)      — GPA/4, values in [0,1]
        'ids'         : list[str]                     — source student_id + aug
        'static_dim'  : int
        'temporal_dim': int
        'history_len' : int
        'horizon'     : int
    """
    path = Path(data_path) if data_path else _DATA_PATH
    df   = pd.read_csv(path)
    rng  = np.random.default_rng(seed)

    all_static, all_temporal, all_target, all_ids = [], [], [], []

    for idx, row in df.iterrows():
        prog_enc  = _PROG_ENC.get(str(row.get("programme", "CSAI")), 0.0)
        school_enc = _SCHOOL_ENC.get(str(row.get("school", "CS&AI")), 0.0)
        attendance = float(row.get("avg_attendance", 80)) / 100.0
        final_sc   = float(row.get("avg_final",    70)) / 100.0
        failed_r   = float(row.get("failed_ratio",  0))
        difficulty = _PROG_DIFFICULTY.get(str(row.get("programme", "CSAI")), 0.5)

        static_x = [prog_enc, school_enc, attendance, final_sc, failed_r, difficulty]

        for aug in range(n_augmentations):
            child_seed = int(rng.integers(0, 2**31))
            child_rng  = np.random.default_rng(child_seed)

            gpas, loads = _simulate_trajectory(row, n_semesters, child_rng)
            s_list, t_list, tgt_list = _extract_windows(
                gpas, loads, static_x, history_len, horizon
            )
            for win_i, (s, t, tgt) in enumerate(zip(s_list, t_list, tgt_list)):
                all_static.append(s)
                all_temporal.append(t)
                all_target.append(tgt)
                sid = row.get("student_id", f"S{idx}")
                all_ids.append(f"{sid}_a{aug}_w{win_i}")

    return {
        "static":       np.array(all_static,   dtype=np.float32),
        "temporal":     np.array(all_temporal,  dtype=np.float32),
        "target":       np.array(all_target,    dtype=np.float32),
        "ids":          all_ids,
        "static_dim":   STATIC_DIM,
        "temporal_dim": TEMPORAL_DIM,
        "history_len":  history_len,
        "horizon":      horizon,
    }
