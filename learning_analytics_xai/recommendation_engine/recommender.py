"""
Phase 7 — Personalized Recommendation Engine

NOT a separate ML model.
Uses: Prediction Results + SHAP Values + Academic Rules

Logic:
  1. Look at which SHAP features are most negative (hurting GPA).
  2. Cross-reference with academic rules (attendance thresholds, GPA bands).
  3. Generate actionable, personalized, prioritised recommendations.

Returns a list of Recommendation objects, ordered by impact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Curriculum recs are injected at call-time from curriculum_engine to avoid
# circular imports and keep this module free of curriculum-specific logic.



RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# Thresholds from Zewail academic regulations
ATT_WARNING   = 75.0   # below → attendance warning
ATT_CRITICAL  = 60.0   # below → failing attendance
ASSIGN_WARN   = 65.0
QUIZ_WARN     = 60.0
MID_WARN      = 65.0
FAIL_WARN     = 2       # ≥ this many failed courses → serious concern
GPA_PROB      = 1.7     # Zewail probation threshold
GPA_GOOD      = 2.5
FAIL_RATIO_W  = 0.10    # >10% failed ratio → warning


@dataclass
class Recommendation:
    priority:    int          # 1=Critical, 2=High, 3=Medium, 4=Low
    category:    str          # Attendance, Exams, Assignments, Labs, Strategy
    icon:        str          # emoji for dashboard display
    title:       str          # short headline
    detail:      str          # actionable explanation
    impact:      str          # "Expected GPA gain: +0.3"
    shap_driver: str          # which feature drove this recommendation


def generate_recommendations(
    feature_dict: dict,
    predicted_gpa: float,
    risk_level: int,
    top_negative: list[dict],
    top_positive: list[dict],
    curriculum_recs: Optional[list[dict]] = None,
) -> list[Recommendation]:
    """
    Parameters
    ----------
    feature_dict  : {feature_name: value} for the student
    predicted_gpa : float
    risk_level    : 0=Low, 1=Medium, 2=High
    top_negative  : top SHAP factors hurting GPA  [{feature, shap_value, feature_value}]
    top_positive  : top SHAP factors helping GPA  [{feature, shap_value, feature_value}]

    Returns
    -------
    list of Recommendation sorted by priority (1=most urgent)
    """
    recs: list[Recommendation] = []

    att     = feature_dict.get("avg_attendance",     80)
    assigns = feature_dict.get("avg_assignments",    70)
    quizzes = feature_dict.get("avg_quizzes",        70)
    labs    = feature_dict.get("avg_labs",           70)
    midterm = feature_dict.get("avg_midterm",        70)
    final_e = feature_dict.get("avg_final",          70)
    failed  = feature_dict.get("failed_courses",      0)
    fail_r  = feature_dict.get("failed_course_ratio", 0)
    trend   = feature_dict.get("performance_trend",   0)
    moment  = feature_dict.get("semester_momentum",   0)
    consist = feature_dict.get("academic_consistency",0.7)
    semester = feature_dict.get("semester", 1)

    # ── SHAP-driven recommendations (top negative factors first) ──────────────
    for neg in top_negative[:5]:
        raw = neg.get("raw_name", "")
        sv  = neg.get("shap_value", 0)
        fv  = neg.get("feature_value", 0)

        if raw == "avg_attendance" or raw == "attendance_risk_score":
            if att < ATT_CRITICAL:
                recs.append(Recommendation(
                    priority=1,
                    category="Attendance",
                    icon="🚨",
                    title="Critical: Attendance Below 60%",
                    detail=(
                        f"Your attendance ({att:.0f}%) is below the critical threshold. "
                        "You risk academic penalties and course failure. "
                        "Contact your academic advisor immediately and create a recovery plan."
                    ),
                    impact="Expected GPA gain: +0.5 to +0.8 if recovered to 80%",
                    shap_driver="Attendance",
                ))
            elif att < ATT_WARNING:
                recs.append(Recommendation(
                    priority=2,
                    category="Attendance",
                    icon="⚠️",
                    title="Improve Your Attendance",
                    detail=(
                        f"Your attendance ({att:.0f}%) is below the recommended 75%. "
                        "Research shows each 10% increase in attendance correlates with "
                        "0.2 GPA improvement. Aim for at least 80% in each course."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.5:.2f} (based on SHAP analysis)",
                    shap_driver="Attendance",
                ))

        elif raw == "avg_midterm":
            if midterm < MID_WARN:
                recs.append(Recommendation(
                    priority=1,
                    category="Exams",
                    icon="📝",
                    title="Focus on Midterm Preparation",
                    detail=(
                        f"Your midterm average ({midterm:.0f}/100) is dragging down your GPA. "
                        "Schedule dedicated weekly study sessions. Form a study group with peers. "
                        "Meet professors during office hours. Practice past exam papers."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.6:.2f}",
                    shap_driver="Midterm Score",
                ))

        elif raw == "avg_final":
            if final_e < 65:
                recs.append(Recommendation(
                    priority=2,
                    category="Exams",
                    icon="🎯",
                    title="Strengthen Final Exam Performance",
                    detail=(
                        f"Final exams carry 30% of your grade and your average ({final_e:.0f}/100) "
                        "needs improvement. Start revision 4 weeks before finals. "
                        "Practice past papers under timed conditions. Reduce anxiety through preparation."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.5:.2f}",
                    shap_driver="Final Exam Score",
                ))

        elif raw == "avg_assignments":
            if assigns < ASSIGN_WARN:
                recs.append(Recommendation(
                    priority=2,
                    category="Assignments",
                    icon="📚",
                    title="Improve Assignment Completion Rate",
                    detail=(
                        f"Your assignment average ({assigns:.0f}/100) is below recommended levels. "
                        "Assignments build foundational knowledge tested in exams. "
                        "Create a weekly schedule, submit on time, and seek help early when stuck."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.4:.2f}",
                    shap_driver="Assignments",
                ))

        elif raw == "avg_quizzes":
            if quizzes < QUIZ_WARN:
                recs.append(Recommendation(
                    priority=3,
                    category="Assignments",
                    icon="✏️",
                    title="Improve Quiz Scores",
                    detail=(
                        f"Your quiz average ({quizzes:.0f}/100) indicates gaps in class engagement. "
                        "Attend all lectures, review notes before each class, "
                        "and use active recall techniques during study sessions."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.3:.2f}",
                    shap_driver="Quizzes",
                ))

        elif raw == "failed_courses" or raw == "failed_course_ratio":
            if failed >= FAIL_WARN:
                recs.append(Recommendation(
                    priority=1,
                    category="Strategy",
                    icon="🆘",
                    title=f"Address {failed} Failed Course(s) Urgently",
                    detail=(
                        f"You have failed {failed} course(s), which is severely impacting your GPA. "
                        "You must retake these courses. Consider reducing your credit load this semester. "
                        "Meet with your academic advisor to create a recovery roadmap."
                    ),
                    impact="Passing failed retakes could raise GPA by +0.4 to +0.9",
                    shap_driver="Failed Courses",
                ))

        elif raw == "avg_overall":
            # Generic performance recommendation
            if fv < 65:
                recs.append(Recommendation(
                    priority=2,
                    category="Strategy",
                    icon="📊",
                    title="Holistic Performance Improvement Needed",
                    detail=(
                        f"Your overall course score ({fv:.0f}/100) is below passing threshold comfort. "
                        "Focus on the highest-weighted assessment components: midterms (30%) and finals (30%). "
                        "Request a meeting with each professor to understand where you're losing marks."
                    ),
                    impact=f"Expected GPA gain: +{abs(sv)*0.4:.2f}",
                    shap_driver="Overall Performance",
                ))

    # ── Rule-based additions (not covered by SHAP) ────────────────────────────

    # Consistency issue
    if consist < 0.60:
        recs.append(Recommendation(
            priority=3,
            category="Strategy",
            icon="🎯",
            title="Build Academic Consistency",
            detail=(
                "Your scores vary significantly across different assessment types. "
                "This suggests you perform well in some areas but poorly in others. "
                "Identify your weakest component type and dedicate focused practice there."
            ),
            impact="Consistent high performance adds +0.2 to +0.4 GPA",
            shap_driver="Academic Consistency",
        ))

    # Positive trend — reinforce
    if trend > 5:
        recs.append(Recommendation(
            priority=4,
            category="Strategy",
            icon="📈",
            title="Maintain Your Positive Momentum",
            detail=(
                f"You are improving (final average {trend:.1f} points above midterm average). "
                "This positive trend is excellent. Maintain your current study habits "
                "and continue building on what's working."
            ),
            impact="Sustaining this trend predicts GPA improvement next semester",
            shap_driver="Performance Trend",
        ))
    elif trend < -5:
        recs.append(Recommendation(
            priority=2,
            category="Exams",
            icon="📉",
            title="Reverse Your Declining Trend",
            detail=(
                f"Your final exam scores are {abs(trend):.1f} points below midterm scores on average. "
                "This suggests exam anxiety or insufficient revision time for finals. "
                "Start exam preparation earlier and build a structured revision plan."
            ),
            impact="Reversing this trend could add +0.3 GPA",
            shap_driver="Performance Trend",
        ))

    # Negative semester momentum
    if moment < -3:
        recs.append(Recommendation(
            priority=3,
            category="Strategy",
            icon="⬇️",
            title="Academic Performance Declining Over Semesters",
            detail=(
                "Your performance is lower in recent semesters than earlier ones. "
                "This may indicate increasing course difficulty or reduced motivation. "
                "Consider speaking with a counsellor and reviewing your study strategies."
            ),
            impact="Addressing root cause can prevent further GPA decline",
            shap_driver="Learning Momentum",
        ))

    # GPA band context
    if predicted_gpa < GPA_PROB:
        recs.append(Recommendation(
            priority=1,
            category="Strategy",
            icon="🚨",
            title="Academic Probation Risk — Act Now",
            detail=(
                f"Your predicted GPA ({predicted_gpa:.2f}) is near or below the Zewail academic "
                f"probation threshold ({GPA_PROB}). You must raise your GPA this semester to avoid "
                "serious academic consequences. Schedule an urgent meeting with your advisor."
            ),
            impact="Raising GPA above 2.0 restores good academic standing",
            shap_driver="Cumulative GPA",
        ))
    elif predicted_gpa >= 3.5:
        recs.append(Recommendation(
            priority=4,
            category="Strategy",
            icon="🌟",
            title="Excellent Performance — Aim for Honours",
            detail=(
                f"Your predicted GPA ({predicted_gpa:.2f}) qualifies for Dean's Honours. "
                "Consider taking advanced electives or research projects to further distinguish yourself. "
                "Your strong academic record is an asset for graduate school applications."
            ),
            impact="Maintaining 3.5+ qualifies for Honours distinction at graduation",
            shap_driver="Cumulative GPA",
        ))

    # Labs
    if labs < 65:
        recs.append(Recommendation(
            priority=3,
            category="Labs",
            icon="🔬",
            title="Improve Lab Performance",
            detail=(
                f"Your lab average ({labs:.0f}/100) is below average. Labs build practical skills "
                "and carry 15% of your grade. Review lab instructions before each session, "
                "collaborate with your lab partner, and ask demonstrators for help."
            ),
            impact="Improving labs to 75+ adds approximately +0.15 GPA",
            shap_driver="Lab Scores",
        ))

    # ── Curriculum-aware recommendations (PRIMARY — always shown first) ────────
    # Curriculum recs reference official Zewail handbooks and regulations.
    # Priority-1 curriculum recs (prerequisite bottlenecks, graduation GPA failures)
    # are inserted BEFORE all other recommendations — they represent the most
    # critical academic obligations a student has per programme requirements.
    curriculum_priority1: list[Recommendation] = []
    curriculum_other:     list[Recommendation] = []

    if curriculum_recs:
        for cr in curriculum_recs:
            rec = Recommendation(
                priority    = cr.get("priority",    3),
                category    = cr.get("category",    "Curriculum"),
                icon        = cr.get("icon",        "📋"),
                title       = cr.get("title",       ""),
                detail      = cr.get("detail",      ""),
                impact      = cr.get("impact",      ""),
                shap_driver = cr.get("shap_driver", "Curriculum"),
            )
            if rec.priority == 1:
                curriculum_priority1.append(rec)
            else:
                curriculum_other.append(rec)

    # Assemble final list: critical curriculum recs → score recs → other curriculum recs
    combined = curriculum_priority1 + recs + curriculum_other

    # Deduplicate by title and sort by priority
    seen_titles: set[str] = set()
    unique_recs = []
    for r in combined:
        if r.title not in seen_titles:
            seen_titles.add(r.title)
            unique_recs.append(r)

    unique_recs.sort(key=lambda r: r.priority)
    return unique_recs[:12]   # max 12 to accommodate curriculum recs


def recs_to_dict(recs: list[Recommendation]) -> list[dict]:
    return [
        {
            "priority":    r.priority,
            "category":    r.category,
            "icon":        r.icon,
            "title":       r.title,
            "detail":      r.detail,
            "impact":      r.impact,
            "shap_driver": r.shap_driver,
        }
        for r in recs
    ]


if __name__ == "__main__":
    print("=== Phase 7: Recommendation Engine — standalone test ===")

    test_feat = {
        "avg_attendance": 68.0,
        "avg_assignments": 62.0,
        "avg_quizzes": 58.0,
        "avg_labs": 64.0,
        "avg_midterm": 61.0,
        "avg_final": 59.0,
        "avg_overall": 61.5,
        "failed_courses": 3,
        "failed_course_ratio": 0.15,
        "performance_trend": -2.5,
        "semester_momentum": -3.5,
        "academic_consistency": 0.58,
    }

    test_neg = [
        {"raw_name": "avg_overall", "shap_value": -0.95, "feature_value": 61.5},
        {"raw_name": "avg_midterm", "shap_value": -0.12, "feature_value": 61.0},
        {"raw_name": "avg_attendance", "shap_value": -0.09, "feature_value": 68.0},
    ]
    test_pos = [
        {"raw_name": "credits_passed", "shap_value": 0.05, "feature_value": 48},
    ]

    recs = generate_recommendations(test_feat, 1.1, 2, test_neg, test_pos)
    print(f"\n  Generated {len(recs)} recommendations:")
    for r in recs:
        print(f"\n  [{r.priority}] {r.icon} {r.title}")
        print(f"       {r.detail[:80]}...")
        print(f"       Impact: {r.impact}")

    print("\n=== Phase 7 Complete ===")
