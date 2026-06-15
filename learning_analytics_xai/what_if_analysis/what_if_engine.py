"""
Phase 8 — What-If Analysis / Scenario Simulation Engine

Student changes variables (attendance, midterm, assignments, etc.)
The model instantly recalculates outcomes.

Usage:
  engine = WhatIfEngine()
  result = engine.simulate(base_features, changes)
  # result.delta_gpa, result.new_gpa, result.new_risk, ...
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HERE   = Path(__file__).parent
MODELS = HERE.parent / "models"
DATA   = HERE.parent / "data"

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#27AE60",  1: "#F39C12",     2: "#E74C3C"}


@dataclass
class ScenarioResult:
    original_gpa:   float
    original_risk:  int
    original_label: str
    new_gpa:        float
    new_risk:       int
    new_label:      str
    delta_gpa:      float
    gpa_direction:  str     # "improved", "worsened", "unchanged"
    risk_direction: str     # "improved", "worsened", "unchanged"
    changes_applied: dict   # {feature: (old_val, new_val)}
    message:        str     # human-readable summary


class WhatIfEngine:
    """Loads models once and supports rapid what-if simulations."""

    def __init__(self):
        self._gpa_model  = None
        self._risk_model = None
        self._scaler     = None
        self._feat_cols: list[str] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        with open(MODELS / "gpa_model_xgb.pkl",  "rb") as f:
            self._gpa_model  = pickle.load(f)
        with open(MODELS / "risk_model_xgb.pkl", "rb") as f:
            self._risk_model = pickle.load(f)
        with open(MODELS / "scaler.pkl",          "rb") as f:
            self._scaler = pickle.load(f)
        meta = json.loads((DATA / "feature_names.json").read_text(encoding="utf-8"))
        self._feat_cols = meta["feature_columns"]
        self._loaded = True

    def _predict(self, feature_dict: dict) -> tuple[float, int, list[float]]:
        """Return (gpa, risk_class, risk_proba_list)."""
        x = np.array(
            [feature_dict.get(c, 0.0) for c in self._feat_cols],
            dtype=np.float32
        ).reshape(1, -1)
        x_sc = self._scaler.transform(x)
        gpa  = float(np.clip(self._gpa_model.predict(x_sc)[0], 0.0, 4.0))
        prob = self._risk_model.predict_proba(x_sc)[0].tolist()
        risk = int(np.argmax(prob))
        return round(gpa, 3), risk, [round(p, 4) for p in prob]

    def simulate(
        self,
        base_features: dict,
        changes: dict,
    ) -> ScenarioResult:
        """
        Simulate a scenario.

        Parameters
        ----------
        base_features : current feature dictionary (all feature cols)
        changes       : {feature_name: new_value}  e.g. {"avg_attendance": 85}

        Returns
        -------
        ScenarioResult with comparison of original vs projected outcome
        """
        self._load()

        # Recompute derived features if raw inputs change
        modified = dict(base_features)
        changes_applied: dict = {}

        for feat, new_val in changes.items():
            if feat in modified:
                changes_applied[feat] = (modified[feat], new_val)
                modified[feat] = new_val

        # If core score components change, recalculate engineered features
        score_cols = ["avg_attendance", "avg_assignments", "avg_quizzes",
                      "avg_labs", "avg_midterm", "avg_final"]
        if any(c in changes for c in score_cols):
            modified = self._recompute_derived(modified)

        orig_gpa, orig_risk, _  = self._predict(base_features)
        new_gpa,  new_risk, _   = self._predict(modified)

        delta = round(new_gpa - orig_gpa, 3)
        gpa_dir  = "improved" if delta > 0.05 else ("worsened" if delta < -0.05 else "unchanged")
        risk_dir = (
            "improved"  if new_risk < orig_risk else
            "worsened"  if new_risk > orig_risk else
            "unchanged"
        )

        msg = self._build_message(orig_gpa, new_gpa, delta, orig_risk, new_risk, changes_applied)

        return ScenarioResult(
            original_gpa   = orig_gpa,
            original_risk  = orig_risk,
            original_label = RISK_LABELS[orig_risk],
            new_gpa        = new_gpa,
            new_risk       = new_risk,
            new_label      = RISK_LABELS[new_risk],
            delta_gpa      = delta,
            gpa_direction  = gpa_dir,
            risk_direction = risk_dir,
            changes_applied = changes_applied,
            message        = msg,
        )

    def _recompute_derived(self, feat: dict) -> dict:
        """Recompute engineered features from updated raw scores."""
        att     = feat.get("avg_attendance", 80)
        assigns = feat.get("avg_assignments", 70)
        quizzes = feat.get("avg_quizzes",    70)
        labs    = feat.get("avg_labs",       70)
        midterm = feat.get("avg_midterm",    70)
        final_e = feat.get("avg_final",      70)

        # overall (weighted average)
        overall = 0.10*quizzes + 0.15*assigns + 0.15*labs + 0.30*midterm + 0.30*final_e
        feat["avg_overall"] = round(float(np.clip(overall, 0, 100)), 1)

        # attendance risk score
        threshold = 75.0
        att_risk = ((threshold - att) / threshold) ** 1.5 if att < threshold else 0.0
        feat["attendance_risk_score"] = round(float(np.clip(att_risk, 0, 1)), 4)

        # performance trend
        feat["performance_trend"] = round(final_e - midterm, 2)

        # academic consistency
        scores = [assigns, quizzes, labs, midterm, final_e]
        row_std = float(np.std(scores))
        feat["academic_consistency"] = round(float(np.clip(1 - row_std / 30, 0, 1)), 3)

        # study efficiency (proxy update)
        study_h = feat.get("study_hours", feat.get("avg_overall", 70) / 5)
        norm_h  = (study_h - 5) / (44 - 5)
        eff = (overall / 100) / (max(norm_h, 0.05) + 0.1)
        feat["study_efficiency"] = round(float(np.clip(eff, 0, 5)), 3)

        return feat

    def _build_message(
        self, orig_gpa: float, new_gpa: float, delta: float,
        orig_risk: int, new_risk: int,
        changes: dict
    ) -> str:
        change_strs = []
        DISPLAY = {
            "avg_attendance":   "Attendance",
            "avg_midterm":      "Midterm Score",
            "avg_final":        "Final Exam",
            "avg_assignments":  "Assignments",
            "avg_quizzes":      "Quizzes",
            "avg_labs":         "Labs",
        }
        for feat, (old, new) in changes.items():
            label = DISPLAY.get(feat, feat)
            change_strs.append(f"{label}: {old:.0f} -> {new:.0f}")

        changes_text = " | ".join(change_strs)
        delta_sign   = "+" if delta >= 0 else ""

        msg = (
            f"Scenario: {changes_text}\n"
            f"GPA: {orig_gpa:.2f} -> {new_gpa:.2f} ({delta_sign}{delta:.3f})\n"
            f"Risk: {RISK_LABELS[orig_risk]} -> {RISK_LABELS[new_risk]}"
        )
        return msg

    def multi_scenario(
        self,
        base_features: dict,
        scenarios: list[dict],
        labels: Optional[list[str]] = None,
    ) -> list[ScenarioResult]:
        """Run multiple what-if scenarios and return results list."""
        self._load()
        results = []
        for i, changes in enumerate(scenarios):
            r = self.simulate(base_features, changes)
            results.append(r)
        return results


# Singleton
_whatif: Optional[WhatIfEngine] = None


def get_whatif_engine() -> WhatIfEngine:
    global _whatif
    if _whatif is None:
        _whatif = WhatIfEngine()
    return _whatif


def simulate_curriculum_scenario(
    base_features:  dict,
    programme:      str,
    passed_codes:   list[str],
    failed_codes:   list[str],
    credits_passed: float,
    semester:       int,
    course_code:    str,
    outcome:        str,
    predicted_gpa:  float,
) -> dict:
    """
    Thin wrapper that calls CurriculumEngine.simulate_course_scenario() and
    returns a plain dict (avoids importing the dataclass across modules).

    outcome: "pass" | "fail" | "retake" | "postpone"
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from curriculum_intelligence.curriculum_engine import get_engine as _get_curr

    eng = _get_curr()
    result = eng.simulate_course_scenario(
        programme     = programme,
        passed_codes  = passed_codes,
        failed_codes  = failed_codes,
        credits_passed= credits_passed,
        semester      = semester,
        course_code   = course_code,
        outcome       = outcome,
        predicted_gpa = predicted_gpa,
    )
    return {
        "scenario_name":        result.scenario_name,
        "course_code":          result.course_code,
        "course_title":         result.course_title,
        "outcome":              result.outcome,
        "delta_credits":        result.delta_credits,
        "new_blocked":          result.new_blocked,
        "unblocked":            result.unblocked,
        "new_graduation_delay": result.new_graduation_delay,
        "new_graduation_ok":    result.new_graduation_ok,
        "curriculum_message":   result.curriculum_message,
    }


def get_curriculum_preset_scenarios(
    programme:     str,
    passed_codes:  list[str],
    failed_codes:  list[str],
    credits_passed: float,
    semester:      int,
    predicted_gpa: float,
) -> list[dict]:
    """
    Return a list of pre-built curriculum what-if scenarios for the student.

    Each scenario simulates passing, failing, retaking, or postponing a course
    that is relevant to the student's situation (failed courses, high-impact
    prerequisites, graduation pace).

    Returns a list of scenario dicts as returned by simulate_curriculum_scenario().
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from curriculum_intelligence.curriculum_engine import get_engine as _get_curr

    eng = _get_curr()
    scenarios = []

    # Scenario group 1: Retake each failed course
    for code in failed_codes[:3]:
        result = eng.simulate_course_scenario(
            programme=programme, passed_codes=passed_codes,
            failed_codes=failed_codes, credits_passed=credits_passed,
            semester=semester, course_code=code,
            outcome="retake", predicted_gpa=predicted_gpa,
        )
        scenarios.append({
            "label":    f"Retake {code}",
            "group":    "Retake Failed",
            "scenario": _dataclass_to_dict(result),
        })

    # Scenario group 2: Fail a passed course (stress test)
    for code in passed_codes[:2]:
        deps = eng.get_dependents(code)
        if deps:  # only show high-impact courses
            result = eng.simulate_course_scenario(
                programme=programme, passed_codes=passed_codes,
                failed_codes=failed_codes, credits_passed=credits_passed,
                semester=semester, course_code=code,
                outcome="fail", predicted_gpa=predicted_gpa,
            )
            scenarios.append({
                "label":    f"Fail {code}",
                "group":    "Risk: If You Fail",
                "scenario": _dataclass_to_dict(result),
            })

    # Scenario group 3: Pass the highest-impact prerequisite
    prog_codes = eng.get_programme_courses(programme)
    high_impact = sorted(
        [(c, len(eng.get_all_dependents(c))) for c in prog_codes],
        key=lambda x: x[1], reverse=True,
    )
    for code, n_deps in high_impact[:2]:
        if code not in passed_codes and n_deps > 0:
            result = eng.simulate_course_scenario(
                programme=programme, passed_codes=passed_codes,
                failed_codes=failed_codes, credits_passed=credits_passed,
                semester=semester, course_code=code,
                outcome="pass", predicted_gpa=predicted_gpa,
            )
            scenarios.append({
                "label":    f"Pass {code} (unlocks {n_deps} courses)",
                "group":    "High-Impact Pass",
                "scenario": _dataclass_to_dict(result),
            })

    return scenarios


def _dataclass_to_dict(obj) -> dict:
    """Convert a CurriculumScenarioResult dataclass to a plain dict."""
    return {
        "scenario_name":        obj.scenario_name,
        "course_code":          obj.course_code,
        "course_title":         obj.course_title,
        "outcome":              obj.outcome,
        "delta_credits":        obj.delta_credits,
        "new_blocked":          obj.new_blocked,
        "unblocked":            obj.unblocked,
        "new_graduation_delay": obj.new_graduation_delay,
        "new_graduation_ok":    obj.new_graduation_ok,
        "curriculum_message":   obj.curriculum_message,
    }


if __name__ == "__main__":
    print("=== Phase 8: What-If Analysis Engine — standalone test ===")

    base = {
        "avg_attendance":        70.0,
        "avg_assignments":       64.0,
        "avg_quizzes":           60.0,
        "avg_labs":              68.0,
        "avg_midterm":           62.0,
        "avg_final":             60.0,
        "avg_overall":           62.8,
        "semester":              4,
        "credits_registered":    60,
        "credits_passed":        48,
        "failed_courses":        3,
        "attendance_risk_score": 0.06,
        "course_difficulty_index": 0.55,
        "failed_course_ratio":   0.15,
        "performance_trend":     -2.0,
        "credit_completion_ratio": 0.80,
        "academic_consistency":  0.65,
        "study_efficiency":      1.2,
        "semester_momentum":     -1.5,
        "programme_encoded":     0.55,
        "school_encoded":        2.0,
        "study_hours":           20.0,
    }

    eng = get_whatif_engine()

    scenarios = [
        {"avg_attendance": 85, "avg_midterm": 75, "avg_final": 73},
        {"avg_attendance": 85},
        {"avg_midterm": 80, "avg_assignments": 80},
        {"avg_attendance": 90, "avg_midterm": 85, "avg_assignments": 85, "avg_final": 82},
    ]
    labels = [
        "Improve attendance + midterm + final",
        "Attendance only",
        "Better midterm + assignments",
        "Full improvement",
    ]

    results = eng.multi_scenario(base, scenarios, labels)

    print(f"\n  Base GPA: {results[0].original_gpa:.3f}  Risk: {results[0].original_label}")
    print()
    for label, r in zip(labels, results):
        sign = "+" if r.delta_gpa >= 0 else ""
        print(f"  [{label}]")
        print(f"    New GPA: {r.new_gpa:.3f}  ({sign}{r.delta_gpa:.3f})  Risk: {r.new_label}  ({r.gpa_direction})")

    print("\n=== Phase 8 Complete ===")
