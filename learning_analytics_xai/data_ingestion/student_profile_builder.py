"""
Student Profile Builder — Learning Analytics XAI

Converts a list[CourseRecord] into a StudentXAIProfile that feeds
every downstream XAI engine (SHAP, GPA forecaster, recommender, what-if).

GROUND TRUTH RULE
-----------------
The student's actual Classroom history is the sole source of truth.
The official study plan (study_plans.json) is used ONLY for:
  - Academic position analysis (how far ahead/behind the standard pace)
  - Gap detection (courses not yet taken that appear in curriculum)
  - Enriching performance context labels

The study plan is NEVER used to:
  - Infer which courses were completed
  - Assume the student followed the official semester sequence
  - Fabricate course history
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .classroom_importer import CourseRecord, _extract_code_from_section

_HERE = Path(__file__).parent
_DATA = _HERE.parent / "data"

# Standard Zewail credit load per semester (used for position estimation only)
_AVG_CREDITS_PER_SEM: int = 15


def _load_prog_total_credits() -> dict[str, int]:
    """
    Load programme total credit requirements from degree_requirements.json.
    Source: Zewail City of Science, Technology and Innovation — Academic Handbook.
    Falls back to hardcoded defaults if the file is missing.
    """
    try:
        raw = (_DATA / "degree_requirements.json").read_text("utf-8")
        data = json.loads(raw)
        return {code: int(info["total_credits"]) for code, info in data.items()}
    except Exception:
        # Fallback — values taken directly from the Zewail City handbook
        return {
            "CSAI": 132, "DSAI": 132, "SWE": 132, "IT":  132,
            "MATH": 132, "PHYS": 132, "CHEM": 132, "BIOL": 132,
            "BUS":  114, "FIN":  114, "MKT": 114,  "ACC": 114,
            "MECH": 140, "CIV":  140, "EEE": 140,  "ENGR": 140,
        }


_PROG_TOTAL_CREDITS: dict[str, int] = _load_prog_total_credits()


# ── Profile dataclass ─────────────────────────────────────────────────────────

@dataclass
class StudentXAIProfile:
    """
    Coursework Analytics profile derived from Google Classroom history + curriculum data.

    DESIGN RULE — Classroom is a Coursework Analytics source only.
    This profile describes observed coursework activity.  It does NOT encode
    official academic outcomes (pass/fail, official GPA, graduation eligibility).
    Official academic standing must come from the registrar.

    FIELD CLASSIFICATION
    --------------------
    Coursework : directly observed from Classroom activity
                 → credits_completed_coursework, completed_codes, in_progress_codes
                 → avg_* performance metrics, performance indices
    Estimated  : engineered proxies not directly observed
                 → study_hours, study_efficiency, academic_position (pace estimate)
                 → semester_performance_scores (weighted coursework % per semester)

    The study plan is used ONLY for gap and pace analysis, never to infer completion.
    """
    # ── Identity ───────────────────────────────────────────────────────────────
    user_email:                      str   = ""
    programme:                       str   = ""

    # ── Coursework credit counts (verified credits only) ──────────────────────
    # Credits from courses whose code and credit value were resolved against
    # course_catalog.json.  Courses with unresolved credits are NOT counted here.
    # These reflect Classroom coursework completion — NOT official academic credits.
    credits_completed_coursework:    int   = 0   # formerly credits_passed
    credits_insufficient_data:       int   = 0   # formerly credits_failed
    credits_in_progress:             int   = 0
    credits_total_attempted:         int   = 0
    credits_unresolved:              int   = 0   # courses with no catalog credit match

    actual_semester_number:          int   = 1   # estimated from credits ÷ avg_per_sem

    # ── Course classification (Classroom activity — NOT academic outcomes) ─────
    completed_codes:                 list[str] = field(default_factory=list)   # completed coursework
    insufficient_data_codes:         list[str] = field(default_factory=list)   # insufficient data
    in_progress_codes:               list[str] = field(default_factory=list)

    # ── Full course records ───────────────────────────────────────────────────
    course_records:                  list[CourseRecord] = field(default_factory=list)
    insufficient_data_courses:       list[CourseRecord] = field(default_factory=list)
    unresolved_courses:              list[CourseRecord] = field(default_factory=list)

    # ── Per-semester coursework performance sequence ───────────────────────────
    # Credit-weighted average coursework score (overall_pct) per semester.
    # These are Classroom performance scores — NOT official GPA values.
    semester_performance_scores:     list[float] = field(default_factory=list)
    semester_labels:                 list[str]   = field(default_factory=list)
    semester_credits:                list[int]   = field(default_factory=list)

    # ── Academic position (ESTIMATED — pace comparison only) ──────────────────
    academic_position:               str   = "Unknown"
    semesters_deviation:             float = 0.0

    # ── Aggregate coursework performance features ─────────────────────────────
    avg_attendance:                  float = 0.0
    avg_assignments:                 float = 0.0
    avg_quizzes:                     float = 0.0
    avg_labs:                        float = 0.0
    avg_midterm:                     float = 0.0
    avg_predicted_final:             float = 0.0   # estimated; informational only
    avg_overall:                     float = 0.0
    insufficient_data_count:         int   = 0     # formerly failed_courses_count

    # ── Coursework Analytics Indices (replace GPA-based metrics) ─────────────
    # Derived from observed Classroom activity only — NOT official academic metrics.
    coursework_performance_index:    float = 0.0   # 0–100, avg coursework score
    engagement_index:                float = 0.0   # 0–100, attendance + assignments + labs
    consistency_index:               float = 0.0   # 0–100, score consistency across courses
    risk_index:                      float = 0.0   # 0–100, fraction of weak/at-risk courses

    # ── Course-level analytics ────────────────────────────────────────────────
    weak_courses:                    list[CourseRecord] = field(default_factory=list)
    risk_courses:                    list[CourseRecord] = field(default_factory=list)
    strong_courses:                  list[CourseRecord] = field(default_factory=list)

    # ── Performance trend (from semester_performance_scores) ──────────────────
    performance_trend:               str   = "stable"

    # ── Study plan gap analysis (reference only) ──────────────────────────────
    curriculum_gaps:                 list[str] = field(default_factory=list)

    # ── Course matching validation report ─────────────────────────────────────
    # Populated by StudentProfileBuilder.build(); used for dashboard validation panel.
    matching_report:                 dict      = field(default_factory=dict)


# ── Builder ───────────────────────────────────────────────────────────────────

class StudentProfileBuilder:
    """
    Converts list[CourseRecord] from ClassroomImporter into StudentXAIProfile.

    Steps:
    1. Match each course name to official code via CurriculumEngine
    2. Fill credits from course catalog
    3. Infer programme from code distribution if not provided
    4. Compute credits_passed / failed / in_progress from actual records
    5. Compute actual_semester_number from credits (never from self-report)
    6. Group records by section label → per-semester GPA sequence
    7. Compute academic position vs standard pace
    8. Aggregate performance features for 30-feature vector
    9. Tag weak / strong / risk courses
    10. Compute GPA trajectory
    11. Identify curriculum gaps (reference only)
    """

    def __init__(self) -> None:
        self._curr_engine = None

    def _curriculum_engine(self):
        if self._curr_engine is None:
            import sys
            sys.path.insert(0, str(_HERE.parent))
            from curriculum_intelligence.curriculum_engine import get_engine
            self._curr_engine = get_engine()
        return self._curr_engine

    def _load_catalog(self) -> dict[str, dict]:
        try:
            raw = (_DATA / "course_catalog.json").read_text("utf-8")
            catalog = json.loads(raw)
            return {c["code"].upper(): c for c in catalog if "code" in c}
        except Exception:
            return {}

    def _load_degree_requirements(self) -> dict[str, dict]:
        try:
            raw = (_DATA / "degree_requirements.json").read_text("utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        records:    list[CourseRecord],
        user_email: str = "",
        programme:  str = "",
    ) -> StudentXAIProfile:
        """
        Build StudentXAIProfile from Classroom course records.

        Parameters
        ----------
        records    : from ClassroomImporter.fetch_all_records()
        user_email : student's Google account email
        programme  : known programme code; if empty, inferred from courses
        """
        eng     = self._curriculum_engine()
        catalog = self._load_catalog()
        deg_req = self._load_degree_requirements()

        # ── 1. Match courses → official codes and resolve credits ────────────────
        # Matching priority (handbook-first):
        #   1. Extract code from section identifier  (e.g. "CSAI 100-LCTR-02" → "CSAI100")
        #   2. Handbook aliases / title matching via CurriculumEngine
        #   3. Fuzzy word-overlap matching via CurriculumEngine
        #
        # Credits MUST come from course_catalog.json; never fall back to an assumed value.
        # Unresolved courses (no catalog match) are tracked separately.
        _raw_course_count  = len(records)
        _section_matched   = 0
        _name_matched      = 0
        _unmatched_names: list[str] = []

        for rec in records:
            # Priority 1 — section string extraction (highest confidence)
            section_code = _extract_code_from_section(rec.course_section)
            if section_code and section_code in catalog:
                cat_entry            = catalog[section_code]
                rec.matched_code     = section_code
                rec.credits          = int(cat_entry.get("credits", 0))
                rec.credits_verified = True
                _section_matched    += 1
                continue

            # Priority 2–4 — CurriculumEngine (exact → alias → fuzzy)
            code = eng.match_course_code(rec.course_name)
            if code:
                cat_entry = catalog.get(code, {})
                if cat_entry:
                    rec.matched_code     = code
                    rec.credits          = int(cat_entry.get("credits", 0))
                    rec.credits_verified = True
                else:
                    rec.matched_code     = code
                    rec.credits          = 0
                    rec.credits_verified = False
                _name_matched += 1
            else:
                _unmatched_names.append(rec.course_name)
            # If still unmatched: matched_code stays "", credits stays 0

        # ── 2. Deduplicate retaken/duplicate courses ───────────────────────────
        # If the same course appears more than once (retaken after failure, or the
        # professor created multiple Classroom sections for the same course), keep
        # only the best attempt: passed > in_progress > failed, then highest score.
        records = _deduplicate_records(records)
        _duplicates_merged = _raw_course_count - len(records)

        # Recount after deduplication so accuracy_pct and still_unmatched are correct
        _section_matched = sum(1 for r in records if r.credits_verified and r.matched_code
                               and _extract_code_from_section(r.course_section) == r.matched_code)
        _name_matched    = sum(1 for r in records if r.credits_verified and r.matched_code
                               and _extract_code_from_section(r.course_section) != r.matched_code)
        _unmatched_names = [r.course_name for r in records if not r.matched_code and r.course_name and r.course_name.strip()]

        # ── 3. Infer programme ────────────────────────────────────────────────
        if not programme:
            programme = _infer_programme(records)

        # ── 4. Categorise records by coursework status ───────────────────────
        # Classroom is a Coursework Analytics source only — these categories describe
        # observed Classroom activity, NOT official academic outcomes.

        # Business rule A — Internship (CSAI399, PHYS399, BUS399 …) is always
        # considered completed when found: it is pass/fail and Classroom rarely
        # carries a numeric score for it.
        _INTERNSHIP_CODES = {"CSAI399", "PHYS399", "BUS399", "MECH399", "EEE399", "CIV399"}
        for r in records:
            if r.matched_code and r.matched_code.upper() in _INTERNSHIP_CODES:
                r.pass_status       = "completed_coursework"
                r.coursework_complete = True
                r.in_progress       = False

        # Business rule B — Graduation Project Part 1 is implicitly passed if
        # Part 2 appears anywhere in the student's Classroom history.
        _GP2_CODES = {"CSAI499", "PHYS499", "BUS499"}
        _GP1_CODES = {"CSAI498", "PHYS498", "BUS498"}
        all_codes_in_records = {r.matched_code.upper() for r in records if r.matched_code}
        if all_codes_in_records & _GP2_CODES:
            for r in records:
                if r.matched_code and r.matched_code.upper() in _GP1_CODES:
                    r.pass_status        = "completed_coursework"
                    r.coursework_complete = True
                    r.in_progress        = False

        completed_cw   = [r for r in records if r.pass_status == "completed_coursework"]
        insufficient   = [r for r in records if r.pass_status == "insufficient_data"]
        in_progress_rs = [r for r in records if r.pass_status == "coursework_in_progress"]
        unresolved     = [r for r in records if not r.matched_code]

        completed_codes         = list(dict.fromkeys(
            r.matched_code for r in completed_cw if r.matched_code
        ))
        insufficient_data_codes = list(dict.fromkeys(
            r.matched_code for r in insufficient if r.matched_code
        ))
        in_progress_codes       = list(dict.fromkeys(
            r.matched_code for r in in_progress_rs if r.matched_code
        ))

        # ── 5. Credit counts (verified credits only) ──────────────────────────
        credits_completed   = sum(r.credits for r in completed_cw   if r.credits_verified)
        credits_insuff      = sum(r.credits for r in insufficient    if r.credits_verified)
        credits_in_prog     = sum(r.credits for r in in_progress_rs if r.credits_verified)
        credits_unresolved  = len(unresolved)
        credits_attempted   = credits_completed + credits_insuff

        # ── 6. Estimated semester number (credit-based, never self-reported) ──
        actual_sem = max(1, credits_completed // _AVG_CREDITS_PER_SEM + 1)

        # ── 7. Per-semester coursework performance sequence ───────────────────
        sem_scores, sem_labels, sem_credits = _compute_semester_sequence(records)

        # ── 8. Academic position ──────────────────────────────────────────────
        position, deviation = _compute_academic_position(
            credits_completed, actual_sem, programme, deg_req
        )

        # ── 9. Aggregate performance features ─────────────────────────────────
        graded = [r for r in records if r.overall_pct > 0]

        def _mean_nonzero(vals: list[float]) -> float:
            nonzero = [v for v in vals if v > 0]
            return round(sum(nonzero) / len(nonzero), 1) if nonzero else 0.0

        avg_att  = _mean_nonzero([r.attendance_pct  for r in graded])
        avg_asgn = _mean_nonzero([r.assignments_avg for r in graded])
        avg_quiz = _mean_nonzero([r.quizzes_avg     for r in graded])
        avg_lab  = _mean_nonzero([r.labs_avg        for r in graded])
        avg_mid  = _mean_nonzero([r.midterm_score   for r in graded])
        avg_predicted_fin = _mean_nonzero([
            r.predicted_final_score / 40.0 * 100.0
            for r in graded if r.predicted_final_score > 0
        ])
        avg_ovr = _mean_nonzero([r.overall_pct for r in graded])

        # ── 10. Coursework Analytics Indices ──────────────────────────────────
        import numpy as np

        # Coursework Performance Index — average observed coursework score
        cpi = round(avg_ovr, 1)

        # Engagement Index — average of attendance, assignments, labs per course
        ei_vals = []
        for r in graded:
            parts = [v for v in [r.attendance_pct, r.assignments_avg, r.labs_avg] if v > 0]
            if parts:
                ei_vals.append(sum(parts) / len(parts))
        engagement_idx = round(_mean_nonzero(ei_vals), 1)

        # Consistency Index — 100 minus scaled std-dev of scores across courses
        if len(graded) >= 2:
            scores = [r.overall_pct for r in graded]
            std_dev = float(np.std(scores))
            consistency_idx = round(max(0.0, min(100.0, 100.0 - std_dev * 1.5)), 1)
        else:
            consistency_idx = 50.0

        # Risk Index — fraction of weak/at-risk courses
        if graded:
            risky = [r for r in graded if r.performance_category in ("at_risk", "weak")]
            risk_idx = round(len(risky) / len(graded) * 100, 1)
        else:
            risk_idx = 0.0

        # ── 11. Course tags (by performance_category) ─────────────────────────
        weak   = [r for r in graded if r.performance_category == "weak"]
        strong = [r for r in graded if r.performance_category in ("excellent", "strong")]
        risk   = [r for r in graded if r.performance_category == "at_risk"]

        # ── 12. Performance trend (from semester coursework scores) ───────────
        trend = _compute_trajectory(sem_scores)

        # ── 13. Curriculum gaps (reference only) ──────────────────────────────
        gaps = _compute_gaps(
            programme, completed_codes, insufficient_data_codes,
            in_progress_codes, credits_completed, eng
        )

        return StudentXAIProfile(
            user_email                   = user_email,
            programme                    = programme,
            # ── Coursework credit counts ──────────────────────────────────────
            credits_completed_coursework = credits_completed,
            credits_insufficient_data    = credits_insuff,
            credits_in_progress          = credits_in_prog,
            credits_total_attempted      = credits_attempted,
            credits_unresolved           = credits_unresolved,
            actual_semester_number       = actual_sem,
            # ── Course codes by status ────────────────────────────────────────
            completed_codes              = completed_codes,
            insufficient_data_codes      = insufficient_data_codes,
            in_progress_codes            = in_progress_codes,
            # ── Full record lists ─────────────────────────────────────────────
            course_records               = records,
            insufficient_data_courses    = insufficient,
            unresolved_courses           = unresolved,
            # ── Semester performance sequences ────────────────────────────────
            semester_performance_scores  = sem_scores,
            semester_labels              = sem_labels,
            semester_credits             = sem_credits,
            # ── Academic position (ESTIMATED) ─────────────────────────────────
            academic_position            = position,
            semesters_deviation          = deviation,
            # ── Performance averages ──────────────────────────────────────────
            avg_attendance               = avg_att,
            avg_assignments              = avg_asgn,
            avg_quizzes                  = avg_quiz,
            avg_labs                     = avg_lab,
            avg_midterm                  = avg_mid,
            avg_predicted_final          = avg_predicted_fin,
            avg_overall                  = avg_ovr,
            insufficient_data_count      = len(insufficient),
            # ── Coursework Analytics Indices ──────────────────────────────────
            coursework_performance_index = cpi,
            engagement_index             = engagement_idx,
            consistency_index            = consistency_idx,
            risk_index                   = risk_idx,
            # ── Course tags ───────────────────────────────────────────────────
            weak_courses                 = weak,
            risk_courses                 = risk,
            strong_courses               = strong,
            performance_trend            = trend,
            curriculum_gaps              = gaps,
            matching_report              = {
                "raw_course_count":    _raw_course_count,
                "total_courses":       len(records),
                "duplicates_merged":   _duplicates_merged,
                "matched_by_section":  _section_matched,
                "matched_by_name":     _name_matched,
                "still_unmatched":     len(_unmatched_names),
                "accuracy_pct":        round(
                    (_section_matched + _name_matched) / max(len(records), 1) * 100, 1
                ),
                "total_verified_credits": sum(
                    r.credits for r in records if r.credits_verified and r.matched_code
                ),
                "unmatched_names":     _unmatched_names,
                "matched_courses": [
                    {
                        "Course Name":   r.course_name,
                        "Code":          r.matched_code,
                        "Credits":       r.credits,
                        "Match Method":  "Section" if (
                            _extract_code_from_section(r.course_section) == r.matched_code
                        ) else "Title/Alias",
                    }
                    for r in records if r.matched_code
                ],
            },
        )

    def profile_to_feature_dict(self, profile: StudentXAIProfile) -> dict:
        """
        Convert StudentXAIProfile into the 30-feature dict consumed by XGBoost + SHAP.

        Degree requirements are loaded from degree_requirements.json via CurriculumEngine —
        never from hardcoded dicts.

        The model input key "avg_final" maps to StudentXAIProfile.avg_predicted_final because
        the final exam is never uploaded to Classroom.  The key name is kept for model
        compatibility; it is explicitly documented as a predicted value.
        """
        import numpy as np
        from feature_engineering.feature_engineer import PROG_DIFFICULTY

        att    = profile.avg_attendance                or 0.0
        asgn   = profile.avg_assignments              or 0.0
        quiz   = profile.avg_quizzes                  or 0.0
        labs   = profile.avg_labs                     or 0.0
        ovr    = profile.avg_overall                  or 0.0
        fin    = profile.avg_predicted_final          or ovr  # final not in Classroom → use overall
        # Midterm not always labelled in Classroom; fall back to overall to avoid
        # passing 0 to the model (which trained on 0 meaning "student failed", not "no data")
        mid    = profile.avg_midterm if profile.avg_midterm > 0 else ovr
        failed = float(profile.insufficient_data_count)
        cr     = float(profile.credits_total_attempted or (profile.actual_semester_number * _AVG_CREDITS_PER_SEM))
        cp     = float(profile.credits_completed_coursework)
        sem    = float(profile.actual_semester_number)
        prog   = profile.programme or "CSAI"

        total_courses = max(cr / 3, 1)

        # Attendance risk
        threshold = 75.0
        att_risk  = ((threshold - att) / threshold) ** 1.5 if att < threshold else 0.0

        # Performance trend (predicted final vs midterm)
        trend = fin - mid

        # Credit completion ratio
        ccr = cp / max(cr, 1)

        # Academic consistency
        available_scores = [v for v in [asgn, quiz, labs, mid, fin] if v > 0]
        row_std = float(np.std(available_scores)) if len(available_scores) >= 2 else 0.0
        consist = float(np.clip(1 - row_std / 30, 0, 1))

        # Study efficiency — study_hours is an ESTIMATED proxy (not observed)
        prog_difficulty = PROG_DIFFICULTY.get(prog, 0.55)
        study_h_proxy   = 15 + prog_difficulty * 20   # estimated: 15–26 h/week by programme difficulty
        norm_h = (study_h_proxy - 5) / (44 - 5)
        eff = (ovr / 100) / (max(norm_h, 0.05) + 0.1)

        # Semester momentum from coursework performance sequence
        perf_scores = profile.semester_performance_scores
        if len(perf_scores) >= 2:
            early    = sum(perf_scores[:max(1, len(perf_scores)//2)]) / max(1, len(perf_scores)//2)
            late     = sum(perf_scores[len(perf_scores)//2:]) / max(1, len(perf_scores) - len(perf_scores)//2)
            momentum = round((late - early) / 100.0, 2)   # normalise to ~GPA scale
        else:
            momentum = round(trend * 0.5, 2)

        school_enc = {"CSAI": 2, "SCI": 3, "BUS": 0, "ENGR": 1}.get(
            _programme_to_school(prog), 2
        )
        prog_enc = PROG_DIFFICULTY.get(prog, 0.55)

        # ── Degree requirements from official data (degree_requirements.json) ─
        # Never use hardcoded PROG_CREDITS / PROG_CORE_CREDITS for live analytics.
        eng = self._curriculum_engine()
        req       = eng.get_degree_requirements(prog)
        total_req = float(req.get("total_credits", 132))
        core_req  = float(req.get("core_credits",  86))

        grad_prog = min(cp / max(total_req, 1), 1.0)
        exp_prog  = min(sem / 8.0, 1.0)
        delay_sem = round((exp_prog - grad_prog) * 8, 2)
        core_comp = min(cp / max(core_req, 1), 1.0)
        fail_r    = failed / max(total_courses, 1)
        prereq_px = float(np.clip(1.0 - fail_r, 0, 1))
        blocked_r = float(np.clip(failed * 6 / max(total_req, 1), 0, 0.5))
        align_px  = float(np.clip(ccr * prereq_px, 0, 1))
        readiness = float(np.clip(
            0.40 * grad_prog + 0.25 * core_comp + 0.20 * prereq_px + 0.15 * (1 - min(delay_sem / 8, 1)),
            0, 1,
        ))

        return {
            # ── Classroom-observed features ───────────────────────────────────
            "avg_attendance":             att,
            "avg_assignments":            asgn,
            "avg_quizzes":                quiz,
            "avg_labs":                   labs,
            "avg_midterm":                mid,
            # PREDICTED: maps from avg_predicted_final; final exam not in Classroom
            "avg_final":                  fin,
            "avg_overall":                round(ovr, 1),
            "semester":                   sem,
            "credits_registered":         cr,
            "credits_passed":             cp,
            "failed_courses":             failed,
            # ── Engineered features ───────────────────────────────────────────
            "attendance_risk_score":      round(att_risk, 4),
            "course_difficulty_index":    prog_enc,
            "failed_course_ratio":        round(fail_r, 4),
            "performance_trend":          round(trend, 2),
            "credit_completion_ratio":    round(ccr, 4),
            "academic_consistency":       round(consist, 3),
            # ESTIMATED: study_hours is a proxy derived from programme difficulty
            "study_efficiency":           round(float(np.clip(eff, 0, 5)), 3),
            "semester_momentum":          momentum,
            "programme_encoded":          prog_enc,
            "school_encoded":             float(school_enc),
            "study_hours":                round(study_h_proxy, 1),  # ESTIMATED proxy
            # ── Curriculum features (from degree_requirements.json) ───────────
            "graduation_progress_ratio":  round(grad_prog, 4),
            "expected_progress_ratio":    round(exp_prog, 4),
            "graduation_delay_semesters": delay_sem,
            "core_completion_ratio":      round(core_comp, 4),
            "prereq_completion_proxy":    round(prereq_px, 4),
            "blocked_progress_ratio":     round(blocked_r, 4),
            "curriculum_alignment_proxy": round(align_px, 4),
            "curriculum_readiness_score": round(readiness, 4),
        }


# ── Utility functions ──────────────────────────────────────────────────────────

def _deduplicate_records(records: list[CourseRecord]) -> list[CourseRecord]:
    """
    Remove duplicate course entries that arise when:
    - A student retook a course (two Classroom entries, same course code)
    - A professor created multiple Classroom sections for the same course

    Grouping key: matched_code (preferred) or normalised course name.
    Best attempt wins: completed_coursework > in_progress > insufficient_data,
    then highest overall_pct.
    """
    def _key(r: CourseRecord) -> str:
        if r.matched_code:
            code = r.matched_code.upper()
            # Merge lab-variant into parent: e.g. CSAI201L → CSAI201
            # so lecture and lab sections of the same course dedup together.
            m = re.match(r'^([A-Za-z]{2,8})(\d{3,4})L$', code)
            if m:
                return m.group(1) + m.group(2)
            return code
        return re.sub(r"[^a-z0-9]", "", r.course_name.lower())

    def _priority(r: CourseRecord) -> tuple:
        if r.coursework_complete:
            return (0, -r.overall_pct)
        if r.in_progress:
            return (1, -r.overall_pct)
        return (2, -r.overall_pct)

    groups: dict[str, list[CourseRecord]] = {}
    for rec in records:
        groups.setdefault(_key(rec), []).append(rec)

    result: list[CourseRecord] = []
    for group in groups.values():
        group.sort(key=_priority)
        result.append(group[0])
    return result


def _infer_programme(records: list[CourseRecord]) -> str:
    """
    Infer programme from the most common matched course code prefix.
    Returns "" if no codes matched.
    """
    prefix_freq: dict[str, int] = {}
    for r in records:
        if r.matched_code:
            prefix = r.matched_code.split()[0] if " " in r.matched_code else r.matched_code[:4]
            prefix_freq[prefix] = prefix_freq.get(prefix, 0) + r.credits
    return max(prefix_freq, key=prefix_freq.get) if prefix_freq else ""


def _compute_semester_sequence(
    records: list[CourseRecord],
) -> tuple[list[float], list[str], list[int]]:
    """
    Group completed-coursework records by Classroom section label and compute
    per-semester credit-weighted average coursework score (overall_pct).

    Returns (scores, labels, credit_loads) sorted chronologically.
    These are Classroom performance scores — NOT official GPA values.
    """
    sem_map: dict[str, list[CourseRecord]] = {}
    for r in records:
        if r.pass_status == "completed_coursework" and r.overall_pct > 0 and r.credits > 0:
            # Use normalised term label (e.g. "Fall 2022") — falls back to course_section
            # only when no term could be derived from section string or creationTime.
            label = r.term_label or r.course_section or "Unknown"
            sem_map.setdefault(label, []).append(r)

    if not sem_map:
        return [], [], []

    _TERM_ORDER = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}

    def _sort_key(label: str) -> tuple:
        # "Fall 2022" → (2022, 2), "Spring 2023" → (2023, 0)
        year_m = re.search(r'(\d{4})', label)
        year   = int(year_m.group(1)) if year_m else 9999
        term_m = re.search(r'(fall|spring|summer|winter)', label.lower())
        order  = _TERM_ORDER.get(term_m.group(1), 9) if term_m else 9
        return (year, order)

    sorted_labels = sorted(sem_map.keys(), key=_sort_key)

    scores  = []
    labels  = []
    credits = []
    for label in sorted_labels:
        recs       = sem_map[label]
        total_pts  = sum(r.overall_pct * r.credits for r in recs)
        total_cred = sum(r.credits for r in recs)
        if total_cred > 0:
            scores.append(round(total_pts / total_cred, 1))
            labels.append(label)
            credits.append(total_cred)

    return scores, labels, credits


def _compute_academic_position(
    credits_passed: int,
    actual_sem:     int,
    programme:      str,
    deg_req:        dict,
) -> tuple[str, float]:
    """
    Compare actual credits_passed vs expected at this academic position.

    Expected credits = (actual_sem - 1) × avg_credits_per_semester.
    Programme total is sourced from degree_requirements.json (Zewail handbook).

    Returns (position_label, semesters_deviation).
    Positive deviation = ahead of standard pace; negative = behind.
    """
    # Programme total credits from handbook; fall back to loaded dict or 132
    total_req = float(
        deg_req.get(programme, {}).get("total_credits")
        or _PROG_TOTAL_CREDITS.get(programme, 132)
    )
    avg_sem_credits = total_req / 8.0   # Zewail standard: 8 semesters per degree

    expected = max(0, (actual_sem - 1) * avg_sem_credits)
    diff     = credits_passed - expected
    dev      = round(diff / avg_sem_credits, 2)

    if dev >= 0.5:
        position = "Ahead of Plan"
    elif dev >= -0.3:
        position = "On Track"
    elif dev >= -1.0:
        position = "Slightly Behind"
    else:
        position = "Significantly Behind"

    return position, dev


def _compute_trajectory(sem_gpas: list[float]) -> str:
    """Classify GPA trajectory from historical per-semester values."""
    if len(sem_gpas) < 2:
        return "stable"
    deltas     = [sem_gpas[i] - sem_gpas[i - 1] for i in range(1, len(sem_gpas))]
    avg_delta  = sum(deltas) / len(deltas)
    volatility = max(deltas) - min(deltas)
    if volatility > 0.8:
        return "volatile"
    if avg_delta >  0.10:
        return "improving"
    if avg_delta < -0.10:
        return "declining"
    return "stable"


def _compute_gaps(
    programme:       str,
    completed_codes: list[str],
    failed_codes:    list[str],
    current_codes:   list[str],
    credits_passed:  int,
    eng,
) -> list[str]:
    """
    Identify programme courses expected by this credit level
    that the student has not started.

    Used for informational gap analysis ONLY.
    Never used to fabricate course history.
    """
    if not programme:
        return []
    try:
        prog_codes = set(eng.get_programme_courses(programme))
        taken = set(completed_codes) | set(failed_codes) | set(current_codes)
        not_started = prog_codes - taken
        # Filter to courses that should plausibly have been reached by now
        # (simple heuristic: first N courses by estimated sequence position)
        expected_n = max(0, credits_passed // 3)
        prog_list  = list(prog_codes)[:expected_n]
        gaps = [c for c in prog_list if c in not_started]
        return gaps[:10]  # cap display to 10
    except Exception:
        return []


def _programme_to_school(programme: str) -> str:
    _MAP = {
        "CSAI": "CSAI", "DSAI": "CSAI", "SWD": "CSAI", "IT": "CSAI",
        "BMS":  "SCI",  "NANO": "SCI",  "PHY": "SCI",
        "BUS":  "BUS",
        "MECH": "ENGR", "CIE": "ENGR",
    }
    return _MAP.get(programme.upper(), "CSAI")
