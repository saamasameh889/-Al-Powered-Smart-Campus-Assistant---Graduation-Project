"""
Phase 4 — Feature Engineering Pipeline
Creates meaningful features from the student summary data for model training.

Basic features (from students_summary.csv):
  avg_attendance, avg_assignments, avg_quizzes, avg_labs,
  avg_midterm, avg_final, semester, credits_registered, credits_passed, failed_courses

Advanced engineered features:
  attendance_risk_score      — penalises attendance below 75%
  course_difficulty_index    — ratio of hard programmes in course list
  failed_course_ratio        — failed_courses / total courses taken
  performance_trend          — avg_final - avg_midterm (momentum direction)
  credit_load_score          — credits_registered normalised to programme
  academic_consistency_score — 1 - std of all score components
  study_efficiency_score     — outcome per study-hour proxy
  semester_momentum_score    — late-semester vs early-semester performance

Outputs:
  ../data/features_train.csv  — features for model training
  ../data/feature_names.json  — list of feature column names
  ../data/label_encoders.json — encoding maps for categorical features
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

SUM_PATH = DATA / "students_summary.csv"
RAW_PATH = DATA / "synthetic_students.csv"

# Programme difficulty index (fraction of hard courses, pre-computed)
PROG_DIFFICULTY = {
    "CSAI": 0.55, "DSAI": 0.62, "SWE":  0.40,
    "MECH": 0.73, "EEE":  0.80, "CIV":  0.75,
    "MATH": 0.82, "PHYS": 0.80, "CHEM": 0.71,
    "BUS":  0.18, "FIN":  0.22,
}

# ── TRAINING-PIPELINE CONSTANTS (do NOT use for live analytics) ──────────────
# These constants are used ONLY during offline model training to generate
# curriculum-aware features from synthetic student data (students_summary.csv).
#
# For live analytics with real student data, all degree requirements MUST be
# loaded from degree_requirements.json via CurriculumEngine.get_degree_requirements().
# Using these dicts at runtime would bypass the official handbook data.
#
# Known discrepancy: MECH/EEE/CIV core credits below show 90, but
# degree_requirements.json (authoritative) shows 91.  Do NOT fix here —
# fix in the JSON file and reload.  Training data tolerates this minor offset.

PROG_CREDITS = {
    "CSAI": 132, "DSAI": 132, "SWE":  132,
    "MECH": 140, "EEE":  140, "CIV":  140,
    "MATH": 132, "PHYS": 132, "CHEM": 132,
    "BUS":  114, "FIN":  114,
}

# TRAINING ONLY — see note above.
PROG_CORE_CREDITS = {
    "CSAI": 86, "DSAI": 86, "SWE":  86,
    "MECH": 90, "EEE":  90, "CIV":  90,  # handbook says 91; see note above
    "MATH": 80, "PHYS": 80, "CHEM": 80,
    "BUS":  74, "FIN":  74,
}

# Typical semesters for a full degree at Zewail City
_TOTAL_SEMESTERS = 8


def attendance_risk_score(att: pd.Series) -> pd.Series:
    """Score from 0-1 where higher = more risk from poor attendance.
    Threshold at 75%; below that, risk increases nonlinearly."""
    threshold = 75.0
    risk = np.where(
        att < threshold,
        ((threshold - att) / threshold) ** 1.5,
        0.0
    )
    return pd.Series(np.clip(risk, 0, 1), index=att.index)


def failed_course_ratio(failed: pd.Series, total_courses: pd.Series) -> pd.Series:
    """Fraction of courses failed."""
    return (failed / total_courses.clip(lower=1)).clip(0, 1)


def performance_trend(avg_final: pd.Series, avg_midterm: pd.Series) -> pd.Series:
    """Positive = improving toward finals. Range approx -40 to +40."""
    return avg_final - avg_midterm


def credit_completion_ratio(credits_passed: pd.Series, credits_registered: pd.Series) -> pd.Series:
    """Fraction of registered credits successfully passed."""
    return (credits_passed / credits_registered.clip(lower=1)).clip(0, 1)


def academic_consistency_score(df: pd.DataFrame) -> pd.Series:
    """
    Measures consistency across score components.
    High std across components -> low consistency.
    Score: 1 - normalised_std
    """
    score_cols = ["avg_assignments", "avg_quizzes", "avg_labs", "avg_midterm", "avg_final"]
    available = [c for c in score_cols if c in df.columns]
    if not available:
        return pd.Series(0.5, index=df.index)
    row_std = df[available].std(axis=1)
    # normalise: std of ~10-15 is typical; cap at 30
    normalised = (row_std / 30).clip(0, 1)
    return 1 - normalised


def study_efficiency_score(avg_overall: pd.Series, study_hours: pd.Series) -> pd.Series:
    """Outcome per study-hour proxy. Higher = more efficient learner."""
    # study_hours range 5-44; avg_overall range 0-100
    # normalise study_hours to 0-1
    norm_hours = (study_hours - 5) / (44 - 5)
    norm_score = avg_overall / 100
    # efficiency: high output per input
    efficiency = norm_score / (norm_hours.clip(lower=0.05) + 0.1)
    return efficiency.clip(0, 5)   # keep bounded


def semester_momentum_score(df_raw: pd.DataFrame, student_ids: pd.Series) -> pd.Series:
    """
    Compare early semester (1-2) vs late semester (3+) average overall score.
    Positive = improving over time. Computed from raw records.
    """
    early = df_raw[df_raw["semester"].isin([1, 2])].groupby("student_id")["overall_score"].mean()
    late  = df_raw[df_raw["semester"].isin([3, 4, 5, 6, 7, 8])].groupby("student_id")["overall_score"].mean()
    momentum = late - early    # positive if improving
    result = student_ids.map(momentum).fillna(0)
    return result


def compute_curriculum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute curriculum-aware features from aggregate student data.
    These approximate the exact values the CurriculumEngine computes from
    course codes, using only the summary-level data available at training time.

    Adds 8 new columns to a copy of the input DataFrame:
      graduation_progress_ratio       — credits_passed / total_required
      expected_progress_ratio         — semester / total_semesters
      graduation_delay_semesters      — how many semesters behind expected pace
      core_completion_ratio           — credits_passed / core_required (proxy)
      curriculum_readiness_score      — composite readiness metric
      curriculum_alignment_proxy      — penalizes low credit completion
      blocked_progress_ratio          — estimated proportion of degree blocked by failures
      prereq_completion_proxy         — 1 - failed_course_ratio (proxy)
    """
    df = df.copy()

    # Map programme to total required credits (with fallback)
    total_req = df["programme"].map(PROG_CREDITS).fillna(132).astype(float)
    core_req  = df["programme"].map(PROG_CORE_CREDITS).fillna(86).astype(float)

    credits_passed   = df["credits_passed"].clip(lower=0)
    credits_reg      = df["credits_registered"].clip(lower=1)
    failed_courses   = df["failed_courses"].clip(lower=0)
    semester         = df["semester"].clip(lower=1, upper=_TOTAL_SEMESTERS)

    # 1. Graduation progress ratio (actual vs required)
    df["graduation_progress_ratio"] = (credits_passed / total_req).clip(0, 1).round(4)

    # 2. Expected progress ratio (where you SHOULD be based on semester)
    df["expected_progress_ratio"] = (semester / _TOTAL_SEMESTERS).clip(0, 1).round(4)

    # 3. Graduation delay in semesters (positive = behind schedule)
    delay_ratio = df["expected_progress_ratio"] - df["graduation_progress_ratio"]
    df["graduation_delay_semesters"] = (delay_ratio * _TOTAL_SEMESTERS).round(2)

    # 4. Core course completion ratio (proxy: credits_passed / core_credits_required)
    df["core_completion_ratio"] = (credits_passed / core_req).clip(0, 1).round(4)

    # 5. Prerequisite completion proxy (inverse of failure rate, bounded)
    # Students with high failure rates are likely missing prerequisites
    total_courses  = (credits_reg / 3).clip(lower=1)
    fail_ratio     = (failed_courses / total_courses).clip(0, 1)
    df["prereq_completion_proxy"] = (1.0 - fail_ratio).round(4)

    # 6. Blocked progress ratio (estimated credits blocked by failures)
    # Conservatively: each failed course blocks ~6 credits (direct + one-level dependent)
    avg_blocked_per_failure = 6.0
    df["blocked_progress_ratio"] = (
        (failed_courses * avg_blocked_per_failure) / total_req
    ).clip(0, 0.5).round(4)

    # 7. Curriculum alignment proxy
    # Higher credit completion with low failures → better alignment
    credit_completion = (credits_passed / credits_reg).clip(0, 1)
    df["curriculum_alignment_proxy"] = (
        credit_completion * df["prereq_completion_proxy"]
    ).round(4)

    # 8. Curriculum readiness score (composite, 0–1)
    # Weights: graduation progress (40%), core completion (25%),
    #          prereq proxy (20%), pace (15%)
    pace_score = (1.0 - delay_ratio.clip(0, 1))
    df["curriculum_readiness_score"] = (
        0.40 * df["graduation_progress_ratio"]
        + 0.25 * df["core_completion_ratio"]
        + 0.20 * df["prereq_completion_proxy"]
        + 0.15 * pace_score
    ).clip(0, 1).round(4)

    return df


CURRICULUM_FEATURE_COLS = [
    "graduation_progress_ratio",
    "expected_progress_ratio",
    "graduation_delay_semesters",
    "core_completion_ratio",
    "prereq_completion_proxy",
    "blocked_progress_ratio",
    "curriculum_alignment_proxy",
    "curriculum_readiness_score",
]


def encode_programme(prog: pd.Series) -> tuple[pd.Series, dict]:
    """Ordinal encode programme by difficulty index."""
    enc_map = {p: round(d, 2) for p, d in PROG_DIFFICULTY.items()}
    encoded = prog.map(enc_map).fillna(0.5)
    return encoded, enc_map


def encode_school(school: pd.Series) -> tuple[pd.Series, dict]:
    """Ordinal encode school."""
    schools = school.unique().tolist()
    enc_map = {s: i for i, s in enumerate(sorted(schools))}
    encoded = school.map(enc_map).fillna(0)
    return encoded, enc_map


def encode_risk(risk: pd.Series) -> pd.Series:
    """Encode risk level: Low=0, Medium=1, High=2."""
    mapping = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    return risk.map(mapping).fillna(1).astype(int)


def build_features(df_sum: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from student summaries + raw records."""
    df = df_sum.copy()

    # ── Basic features (already in summary) ──────────────────────────────────
    basic_features = [
        "avg_attendance", "avg_assignments", "avg_quizzes", "avg_labs",
        "avg_midterm", "avg_final", "avg_overall",
        "semester", "credits_registered", "credits_passed", "failed_courses",
    ]
    df_feat = df[["student_id"] + basic_features].copy()

    # ── Advanced engineered features ──────────────────────────────────────────

    # 1. Attendance risk score (0=no risk, 1=high risk)
    df_feat["attendance_risk_score"] = attendance_risk_score(df["avg_attendance"])

    # 2. Course difficulty index (0=easy programme, 1=hard programme)
    df_feat["course_difficulty_index"] = df["programme"].map(PROG_DIFFICULTY).fillna(0.5)

    # 3. Failed course ratio
    total_courses = (df["credits_registered"] / 3).clip(lower=1)   # approx courses from credits
    df_feat["failed_course_ratio"] = failed_course_ratio(df["failed_courses"], total_courses)

    # 4. Performance trend (final - midterm)
    df_feat["performance_trend"] = performance_trend(df["avg_final"], df["avg_midterm"])

    # 5. Credit completion ratio
    df_feat["credit_completion_ratio"] = credit_completion_ratio(
        df["credits_passed"], df["credits_registered"]
    )

    # 6. Academic consistency score (1=very consistent, 0=erratic)
    df_feat["academic_consistency"] = academic_consistency_score(df)

    # 7. Study efficiency (outcome per effort)
    if "study_hours" in df.columns:
        df_feat["study_efficiency"] = study_efficiency_score(df["avg_overall"], df["study_hours"])
    else:
        df_feat["study_efficiency"] = (df["avg_overall"] / 100).clip(0, 1)

    # 8. Semester momentum score
    df_feat["semester_momentum"] = semester_momentum_score(df_raw, df["student_id"])

    # 9. Programme difficulty index (numeric, same as course_difficulty_index but kept named separately)
    df_feat["programme_encoded"] = df["programme"].map(PROG_DIFFICULTY).fillna(0.5)

    # 10. School encoded
    school_enc, school_map = encode_school(df["school"])
    df_feat["school_encoded"] = school_enc

    # ── Curriculum-aware features (Phase 4b — new) ────────────────────────────
    if "programme" in df.columns:
        df_curr = compute_curriculum_features(df)
        for col in CURRICULUM_FEATURE_COLS:
            if col in df_curr.columns:
                df_feat[col] = df_curr[col].values

    # ── Targets ───────────────────────────────────────────────────────────────
    df_feat["target_gpa"]  = df["cumulative_gpa"].round(3)
    df_feat["target_risk"] = encode_risk(df["risk_level"])

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    df_feat = df_feat.dropna(subset=["target_gpa", "target_risk"])

    return df_feat, school_map


def main():
    print("=== Phase 4: Feature Engineering ===")
    df_sum = pd.read_csv(SUM_PATH)
    df_raw = pd.read_csv(RAW_PATH)
    print(f"  Loaded {len(df_sum):,} students, {len(df_raw):,} course records")

    df_feat, school_map = build_features(df_sum, df_raw)

    feature_cols = [c for c in df_feat.columns
                    if c not in ("student_id", "target_gpa", "target_risk")]

    print(f"  Feature matrix shape: {df_feat.shape}")
    print(f"  Features ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"    {c:35s}: mean={df_feat[c].mean():.3f}  std={df_feat[c].std():.3f}")

    print(f"\n  Target GPA: mean={df_feat['target_gpa'].mean():.3f}  "
          f"std={df_feat['target_gpa'].std():.3f}")
    print(f"  Target Risk distribution:")
    print(f"    {df_feat['target_risk'].value_counts().sort_index().to_dict()}")

    # Save feature matrix
    feat_path = DATA / "features_train.csv"
    df_feat.to_csv(feat_path, index=False)
    print(f"\n  Saved features -> {feat_path}")

    # Save metadata
    meta = {
        "feature_columns": feature_cols,
        "target_gpa_col":  "target_gpa",
        "target_risk_col": "target_risk",
        "risk_labels":     {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"},
        "programme_difficulty_map": PROG_DIFFICULTY,
        "school_encoding_map":      school_map,
        "n_samples":       int(len(df_feat)),
    }
    (DATA / "feature_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved metadata -> {DATA / 'feature_names.json'}")

    print("\n=== Phase 4 Complete ===")


if __name__ == "__main__":
    main()
