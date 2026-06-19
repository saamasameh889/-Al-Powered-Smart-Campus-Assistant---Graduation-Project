"""
Phases 9-11 — Learning Analytics & XAI Dashboard Page
Renders inside the existing phase7_streamlit_app.py as a new tab.

Sections:
  1. Conversational Chat    — collects student info naturally
  2. Student Dashboard      — health score, GPA, risk, breakdown
  3. Recommendations        — personalized action plan
  4. What-If Simulator      — interactive scenario sliders
  5. Advanced Analytics     — SHAP, PDP, PCA (for professors)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add parent to sys.path so imports work when called from root
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from xai.explainability import get_engine, FEATURE_LABELS, RISK_LABELS, RISK_COLORS, ExplainabilityEngine
from recommendation_engine.recommender import generate_recommendations
from what_if_analysis.what_if_engine import (
    get_whatif_engine, simulate_curriculum_scenario, get_curriculum_preset_scenarios,
)
from feature_engineering.feature_engineer import PROG_DIFFICULTY
# PROG_CREDITS / PROG_CORE_CREDITS intentionally NOT imported here.
# All degree requirements for live analytics come from CurriculumEngine.
from curriculum_intelligence.curriculum_engine import get_engine as get_curriculum_engine
from data_ingestion.classroom_importer import (
    ClassroomImporter, build_oauth_flow, exchange_code_for_token,
    _GOOGLE_AVAILABLE, get_pass_status,
)
from data_ingestion.student_profile_builder import StudentProfileBuilder, StudentXAIProfile

# ── Colour palette (matches existing purple theme) ────────────────────────────
PURPLE   = "#7C3AED"
PURPLE_L = "#A855F7"
PINK     = "#E879F9"
DARK     = "#0D0720"
CARD_BG  = "rgba(21,12,48,0.85)"

# ── Programmes ────────────────────────────────────────────────────────────────
PROGRAMMES = ["CSAI", "DSAI", "SWE", "MECH", "EEE", "CIV", "MATH", "PHYS", "CHEM", "BUS", "FIN"]
SCHOOLS    = {
    "CSAI":"CS&AI","DSAI":"CS&AI","SWE":"CS&AI",
    "MECH":"ENGR","EEE":"ENGR","CIV":"ENGR",
    "MATH":"SCI","PHYS":"SCI","CHEM":"SCI",
    "BUS":"BUS","FIN":"BUS",
}


# ════════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
.xai-card {
    background: rgba(21,12,48,0.90);
    border: 1px solid rgba(124,58,237,.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.xai-metric-card {
    background: rgba(124,58,237,.12);
    border: 1px solid rgba(124,58,237,.30);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
}
.xai-metric-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F3EFFF;
    margin: 0;
    line-height: 1.1;
}
.xai-metric-lbl {
    font-size: 0.75rem;
    color: #A78BFA;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: 4px;
}
.xai-risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: .05em;
}
.xai-rec-card {
    background: rgba(21,12,48,0.80);
    border-left: 4px solid;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.xai-chat-msg {
    background: rgba(124,58,237,.15);
    border: 1px solid rgba(124,58,237,.25);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #DDD6FE;
}
.xai-chat-ai {
    background: rgba(168,85,247,.10);
    border: 1px solid rgba(168,85,247,.20);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #F3EFFF;
}
.xai-section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #E879F9;
    letter-spacing: -.01em;
    margin: 0 0 12px 0;
}
</style>
"""


# ════════════════════════════════════════════════════════════════════════════════
#  Session state helpers
# ════════════════════════════════════════════════════════════════════════════════

def _new_flow() -> dict:
    return {
        "phase":          "programme",
        "programme":      None,
        "semester":       None,
        "num_courses":    None,
        "courses":        [],
        "wip":            {},
        "failed_courses": None,
        "total_credits":  None,
    }


def _init_state():
    defaults = {
        # ── Classroom import state ──────────────────────────────────────────
        "xai_google_token":      None,   # OAuth access token (session-only)
        "xai_google_email":      "",     # authenticated student email
        "xai_classroom_profile": None,   # StudentXAIProfile from Classroom
        "xai_import_mode":       "manual",  # "classroom" | "manual"
        "xai_import_error":      "",     # last import error message
        # ── Legacy manual-entry flow ────────────────────────────────────────
        "xai_flow":          None,
        "xai_messages":      [],
        "xai_result":        None,
        "xai_recs":          None,
        "xai_analysed":      False,
        "xai_profile":       {},
        "xai_step":          0,
        "xai_curriculum":    {},
        "xai_graduation":    None,
        "xai_passed_codes":  [],
        "xai_failed_codes":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.xai_flow is None:
        st.session_state.xai_flow = _new_flow()


# ── Google OAuth credential helpers ──────────────────────────────────────────

# ── PKCE helpers ─────────────────────────────────────────────────────────────
import hashlib, base64, secrets as _secrets

def _pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE-S256."""
    verifier  = base64.urlsafe_b64encode(_secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# ── Component-type keyword map for feature extraction ─────────────────────────
_COMP_KEYWORDS = {
    "midterm":    ["midterm", "mid term", "mid-term", "mid exam", "midexam"],
    "final":      ["final", "final exam", "end term", "endterm", "end-term"],
    "assignment": ["assignment", "homework", "project", "coursework", "report", "task"],
    "quiz":       ["quiz", "quizzes", "test", "pop quiz"],
    "lab":        ["lab", "laboratory", "practical", "attendance", "lab attendance"],
}


def _profile_to_features(profile: dict) -> dict:
    """
    Convert manual-entry chat profile to the 30-feature dict for XGBoost + SHAP.

    MISSING DATA POLICY
    -------------------
    Values that the student did not provide are set to 0.0 — they are NOT filled
    with invented averages.  Any feature computed from missing inputs is marked in
    the return dict comment as ESTIMATED or DERIVED.

    Degree requirements are loaded from degree_requirements.json via CurriculumEngine,
    never from hardcoded PROG_CREDITS / PROG_CORE_CREDITS dicts.
    """
    # Read exactly what the student entered; use 0.0 when a field is absent.
    # 0.0 signals "not available" — downstream display should show "N/A", not "0%".
    att    = float(profile.get("avg_attendance")   or 0.0)
    asgn   = float(profile.get("avg_assignments")  or 0.0)
    quiz   = float(profile.get("avg_quizzes")      or 0.0)
    labs   = float(profile.get("avg_labs")         or 0.0)
    mid    = float(profile.get("avg_midterm")      or 0.0)
    # avg_final in the manual flow is student-entered (predicted/current), not an official exam score
    fin    = float(profile.get("avg_final")        or 0.0)
    failed = float(profile.get("failed_courses",    0))
    cr     = float(profile.get("credits_registered", 0))
    sem    = float(profile.get("semester",           1))
    prog   = str(profile.get("programme", "CSAI"))

    # overall: weighted from available components; zero if no component data
    avail  = [(quiz, 0.10), (asgn, 0.15), (labs, 0.15), (mid, 0.30), (fin, 0.30)]
    w_sum  = sum(w for v, w in avail if v > 0)
    overall = float(np.clip(
        sum(v * w for v, w in avail if v > 0) / w_sum if w_sum > 0 else 0.0,
        0, 100,
    ))

    total_courses = max(cr / 3, 1)
    # credits_passed is a rough estimate when student did not pass all courses
    cp = max(cr - failed * 3, 0)

    # Attendance risk score
    threshold = 75.0
    att_risk = ((threshold - att) / threshold) ** 1.5 if att > 0 and att < threshold else 0.0

    # Performance trend (predicted final vs midterm — both from student entry)
    trend = fin - mid if fin > 0 and mid > 0 else 0.0

    # Credit completion ratio
    ccr = cp / max(cr, 1) if cr > 0 else 0.0

    # Academic consistency — only include non-zero components
    scores_nonzero = [v for v in [asgn, quiz, labs, mid, fin] if v > 0]
    row_std = float(np.std(scores_nonzero)) if len(scores_nonzero) >= 2 else 0.0
    consist = float(np.clip(1 - row_std / 30, 0, 1))

    # Study efficiency — ESTIMATED proxy; 20h/week is not observed data
    prog_enc   = PROG_DIFFICULTY.get(prog, 0.55)
    study_h_proxy = 15 + prog_enc * 20   # ESTIMATED: 15–26 h/week by programme difficulty
    norm_h = (study_h_proxy - 5) / (44 - 5)
    eff = (overall / 100) / (max(norm_h, 0.05) + 0.1) if overall > 0 else 0.0

    # Semester momentum (ESTIMATED — no per-semester history in manual flow)
    momentum = round(trend * 0.5, 2)

    school_enc = {"CS&AI": 2, "ENGR": 1, "SCI": 3, "BUS": 0}.get(SCHOOLS.get(prog, "CS&AI"), 2)

    # ── Degree requirements from official data (degree_requirements.json) ──────
    curr_eng  = get_curriculum_engine()
    req       = curr_eng.get_degree_requirements(prog)
    total_req = float(req.get("total_credits", 132))
    core_req  = float(req.get("core_credits",  86))

    cp_safe   = max(cp, 0)
    grad_prog = min(cp_safe / max(total_req, 1), 1.0)
    exp_prog  = min(sem / 8.0, 1.0)
    delay_sem = round((exp_prog - grad_prog) * 8, 2)
    core_comp = min(cp_safe / max(core_req, 1), 1.0)
    fail_r    = failed / max(total_courses, 1)
    prereq_px = float(np.clip(1.0 - fail_r, 0, 1))
    blocked_r = float(np.clip(failed * 6 / max(total_req, 1), 0, 0.5))
    align_px  = float(np.clip(ccr * prereq_px, 0, 1))
    readiness = float(np.clip(
        0.40 * grad_prog + 0.25 * core_comp + 0.20 * prereq_px + 0.15 * (1 - min(delay_sem / 8, 1)),
        0, 1,
    ))

    return {
        # ── Student-entered (manual flow) ─────────────────────────────────────
        "avg_attendance":              att,
        "avg_assignments":             asgn,
        "avg_quizzes":                 quiz,
        "avg_labs":                    labs,
        "avg_midterm":                 mid,
        # avg_final here is student-entered current/predicted performance, not official exam
        "avg_final":                   fin,
        "avg_overall":                 round(overall, 1),
        "semester":                    sem,
        "credits_registered":          cr,
        "credits_passed":              cp_safe,
        "failed_courses":              failed,
        # ── Engineered ───────────────────────────────────────────────────────
        "attendance_risk_score":       round(att_risk, 4),
        "course_difficulty_index":     prog_enc,
        "failed_course_ratio":         round(fail_r, 4),
        "performance_trend":           round(trend, 2),
        "credit_completion_ratio":     round(ccr, 4),
        "academic_consistency":        round(consist, 3),
        "study_efficiency":            round(float(np.clip(eff, 0, 5)), 3),  # ESTIMATED
        "semester_momentum":           momentum,                               # ESTIMATED
        "programme_encoded":           prog_enc,
        "school_encoded":              float(school_enc),
        "study_hours":                 round(study_h_proxy, 1),               # ESTIMATED
        # ── Curriculum (from degree_requirements.json) ────────────────────────
        "graduation_progress_ratio":   round(grad_prog, 4),
        "expected_progress_ratio":     round(exp_prog,  4),
        "graduation_delay_semesters":  delay_sem,
        "core_completion_ratio":       round(core_comp, 4),
        "prereq_completion_proxy":     round(prereq_px, 4),
        "blocked_progress_ratio":      round(blocked_r, 4),
        "curriculum_alignment_proxy":  round(align_px,  4),
        "curriculum_readiness_score":  round(readiness, 4),
    }


def _flow_to_profile(flow: dict) -> dict:
    """Convert course-based flow data into a profile dict for _profile_to_features."""
    courses    = flow.get("courses", [])
    buckets    = {k: [] for k in _COMP_KEYWORDS}
    all_pcts:  list[float] = []
    passed_codes: list[str] = []
    failed_codes: list[str] = []

    # Try to match each course name to an official catalog code
    _curr = get_curriculum_engine()

    for course in courses:
        # Score averaging
        for comp in course.get("components", []):
            score = comp.get("score", -1)
            max_m = comp.get("max", 0)
            if score < 0 or max_m <= 0:
                continue
            pct = score / max_m * 100
            all_pcts.append(pct)
            name_l = comp["name"].lower()
            for cat, kws in _COMP_KEYWORDS.items():
                if any(kw in name_l for kw in kws):
                    buckets[cat].append(pct)
                    break

        # Curriculum code matching
        code = _curr.match_course_code(course.get("name", ""))
        if code:
            cur_pct = course.get("current_pct", 0)
            if cur_pct >= 60:
                passed_codes.append(code)
            else:
                failed_codes.append(code)

    overall = sum(all_pcts) / len(all_pcts) if all_pcts else 70.0

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else overall

    return {
        "programme":          flow.get("programme", "CSAI"),
        "semester":           flow.get("semester", 3),
        "avg_attendance":     80.0,
        "avg_assignments":    avg(buckets["assignment"]),
        "avg_quizzes":        avg(buckets["quiz"]),
        "avg_labs":           avg(buckets["lab"]),
        "avg_midterm":        avg(buckets["midterm"]),
        "avg_final":          avg(buckets["final"]),
        "failed_courses":     flow.get("failed_courses", 0),
        "credits_registered": flow.get("total_credits", 45),
        "courses":            courses,
        # Curriculum intelligence data (passed to all downstream modules)
        "passed_course_codes": passed_codes,
        "failed_course_codes": failed_codes,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  Plotly chart helpers
# ════════════════════════════════════════════════════════════════════════════════

def _gauge_chart(value: float, max_val: float, title: str, color: str) -> go.Figure:
    """Circular gauge chart."""
    pct = value / max_val * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f"/{max_val}", "font": {"size": 28, "color": "#F3EFFF"}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 13, "color": "#A78BFA"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#A78BFA",
                     "tickfont": {"color": "#A78BFA", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(21,12,48,0.5)",
            "borderwidth": 1,
            "bordercolor": "rgba(124,58,237,.3)",
            "steps": [
                {"range": [0,    max_val*0.4], "color": "rgba(231,76,60,.15)"},
                {"range": [max_val*0.4, max_val*0.7], "color": "rgba(243,156,18,.15)"},
                {"range": [max_val*0.7, max_val],     "color": "rgba(39,174,96,.15)"},
            ],
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F3EFFF",
    )
    return fig


def _radar_chart(categories: list[str], values: list[float], max_val: float = 100) -> go.Figure:
    """Radar / spider chart of score components."""
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(124,58,237,.25)",
        line=dict(color=PURPLE_L, width=2),
        marker=dict(color=PURPLE_L, size=7),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_val],
                            tickfont=dict(color="#A78BFA", size=9),
                            gridcolor="rgba(124,58,237,.2)"),
            angularaxis=dict(tickfont=dict(color="#DDD6FE", size=10),
                             gridcolor="rgba(124,58,237,.15)"),
            bgcolor="rgba(13,7,32,0.7)",
        ),
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F3EFFF",
    )
    return fig


def _health_bar(score: float) -> go.Figure:
    """Horizontal health bar 0-100."""
    color = "#E74C3C" if score < 40 else ("#F39C12" if score < 65 else "#27AE60")
    fig = go.Figure(go.Bar(
        x=[score], y=["Academic Health"],
        orientation="h",
        marker_color=color,
        text=[f"{score:.0f}/100"],
        textposition="inside",
        textfont=dict(size=14, color="white"),
    ))
    fig.add_trace(go.Bar(
        x=[100 - score], y=["Academic Health"],
        orientation="h",
        marker_color="rgba(124,58,237,.15)",
        showlegend=False,
    ))
    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(t=5, b=5, l=5, r=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,100], visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _trend_chart(features: dict) -> go.Figure:
    """Bar chart of assessment component averages. 'Pred. Final' is an estimate, not official."""
    labels = ["Attendance", "Assignments", "Quizzes", "Labs", "Midterm", "Pred. Final (est.)"]
    keys   = ["avg_attendance", "avg_assignments", "avg_quizzes",
              "avg_labs", "avg_midterm", "avg_final"]
    vals   = [features.get(k, 0) for k in keys]
    colors = ["#27AE60" if v >= 75 else ("#F39C12" if v >= 60 else "#E74C3C") for v in vals]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=[f"{v:.0f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#F3EFFF"),
    ))
    fig.add_hline(y=75, line_dash="dot", line_color="rgba(255,255,255,.4)",
                  annotation_text="Target (75)", annotation_font_color="#A78BFA")
    fig.add_hline(y=60, line_dash="dot", line_color="rgba(231,76,60,.5)",
                  annotation_text="Pass (60)", annotation_font_color="#E74C3C")
    fig.update_layout(
        height=300,
        margin=dict(t=20, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,7,32,0.6)",
        font_color="#F3EFFF",
        yaxis=dict(range=[0, 110], gridcolor="rgba(124,58,237,.15)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def _curriculum_impact_chart(curriculum_shap: dict) -> Optional[go.Figure]:
    """Plotly bar chart of curriculum factor impacts (SHAP-equivalent, in GPA units)."""
    if not curriculum_shap:
        return None
    items = sorted(curriculum_shap.items(), key=lambda x: x[1])
    names  = [n for n, _ in items]
    values = [v for _, v in items]
    colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in values]
    text   = [f"{'+'if v >= 0 else ''}{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        textfont=dict(color="#F3EFFF", size=10),
        hovertemplate="%{y}<br>Impact: %{x:+.3f} GPA<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,.5)", line_width=1.5)
    fig.update_layout(
        height=max(260, len(names) * 38),
        margin=dict(t=10, b=10, l=10, r=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,7,32,0.6)",
        font_color="#F3EFFF",
        xaxis=dict(gridcolor="rgba(124,58,237,.15)", zeroline=False,
                   title=dict(text="Estimated GPA Impact", font=dict(size=10, color="#A78BFA"))),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def _whatif_chart(base_gpa: float, scenario_gpas: list[float], scenario_labels: list[str]) -> go.Figure:
    """Horizontal bar comparison of what-if scenarios."""
    all_vals   = [base_gpa] + scenario_gpas
    all_labels = ["Current"] + scenario_labels
    colors = [PURPLE] + [
        "#27AE60" if g > base_gpa else "#E74C3C"
        for g in scenario_gpas
    ]
    fig = go.Figure(go.Bar(
        x=all_vals, y=all_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in all_vals],
        textposition="outside",
        textfont=dict(color="#F3EFFF", size=11),
    ))
    fig.add_vline(x=2.0, line_dash="dot", line_color="rgba(243,156,18,.7)",
                  annotation_text="Good Standing (2.0)",
                  annotation_font_color="#F39C12", annotation_position="top")
    fig.update_layout(
        height=max(200, 60 * len(all_labels)),
        margin=dict(t=20, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,7,32,0.6)",
        font_color="#F3EFFF",
        xaxis=dict(range=[0, 4.2], gridcolor="rgba(124,58,237,.15)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  Main render function (called from phase7_streamlit_app.py)
# ════════════════════════════════════════════════════════════════════════════════

def render_learning_analytics_page():
    """Entry point — renders the full Learning Analytics & XAI page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="xai-card" style="background:linear-gradient(125deg,#150C30,#2D1065,#4C1D95);
         border-color:rgba(232,121,249,.25);">
        <h2 style="margin:0;font-size:1.6rem;font-weight:800;color:#F3EFFF;">
            Learning Analytics & XAI
        </h2>
        <p style="margin:6px 0 0;color:#A78BFA;font-size:.88rem;">
            Explainable AI Platform — Predict GPA, understand risks,
            get personalized recommendations, simulate your future.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation tabs ─────────────────────────────────────────────────────────
    tab_labels = [
        "AI Analysis Chat",
        "Academic Dashboard",
        "Recommendations",
        "What-If Simulator",
        "Advanced Analytics",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_chat_tab()

    with tabs[1]:
        _render_dashboard_tab()

    with tabs[2]:
        _render_recommendations_tab()

    with tabs[3]:
        _render_whatif_tab()

    with tabs[4]:
        _render_advanced_tab()


# ════════════════════════════════════════════════════════════════════════════════
#  Tab 1: Conversational Chat  —  Course-based data collection
# ════════════════════════════════════════════════════════════════════════════════

import re as _re

def _md(text: str) -> str:
    """Convert **bold** and *italic* markdown to HTML for safe injection into divs."""
    text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = text.replace('\n', '<br>')
    return text


def _chat_question(flow: dict) -> str:
    """Return the question text for the current phase (markdown supported)."""
    phase     = flow["phase"]
    wip       = flow.get("wip", {})
    prog      = flow.get("programme", "your programme")
    sem       = flow.get("semester", "?")
    n_c       = flow.get("num_courses", 1)
    n_done    = len(flow.get("courses", []))
    c_num     = n_done + 1
    c_name    = wip.get("course_name", "the course")
    n_comp    = wip.get("num_comps", 1)
    p_num     = len(wip.get("components", [])) + 1
    comp_name = wip.get("comp_name", "this component")
    comp_max  = wip.get("comp_max", 20.0)

    if phase == "programme":
        return "**Select your major / programme:**"
    if phase == "semester":
        return f"**{prog}** — noted. **Which semester are you currently in?**"
    if phase == "num_courses":
        return f"Semester **{sem}**. **How many courses are you registered in this semester?**"
    if phase == "course_name":
        return (f"Course **{c_num} of {n_c}** — "
                "**What is the course name or code?**  \n"
                "*(e.g. Machine Learning, CSAI 301)*")
    if phase == "course_credits":
        return (f"**How many credit hours is {c_name} worth?**  \n"
                "*(Enter the value from the official course catalog — "
                "self-reported credits are marked as unverified)*")
    if phase == "course_num_comps":
        return (f"**{c_name}** ({wip.get('course_credits', 3)} cr).  \n"
                "**How many assessment components does this course have?**  \n"
                "*(Midterm, Assignment, Quiz, Lab, Final — count each separately)*")
    if phase == "comp_name":
        return (f"Component **{p_num} of {n_comp}** for *{c_name}*:  \n"
                "**What is this component called?**  \n"
                "*(e.g. Midterm, Assignment 1, Lab Attendance, Final Exam)*")
    if phase == "comp_max":
        return f"**Maximum marks for {comp_name}?**"
    if phase == "comp_score":
        return (f"**Your score in {comp_name}** (out of {comp_max:.0f})?  \n"
                "*(Tick the box below if not taken yet)*")
    if phase == "failed":
        return ("**How many courses have you previously failed** across all semesters?  \n"
                "*(Enter 0 if none)*")
    if phase == "total_credits":
        return ("**Total credit hours registered across all semesters combined?**  \n"
                "*(Include current semester)*")
    return ""


def _render_chat_widget(wtype: str, kwargs: dict, step: int):
    """Render the input widget for the current question; returns the answer."""
    key = f"xai_w_{step}"
    if wtype == "select":
        return st.selectbox("Answer", kwargs["options"], key=key,
                            label_visibility="collapsed")
    if wtype == "int":
        return st.number_input("Answer",
            min_value=int(kwargs.get("lo", 0)),
            max_value=int(kwargs.get("hi", 100)),
            value=int(kwargs.get("default", 5)),
            step=1, key=key, label_visibility="collapsed")
    if wtype == "float":
        return st.number_input("Answer",
            min_value=float(kwargs.get("lo", 0)),
            max_value=float(kwargs.get("hi", 200)),
            value=float(kwargs.get("default", 20)),
            step=0.5, key=key, label_visibility="collapsed")
    if wtype == "text":
        return st.text_input("Answer",
            placeholder=kwargs.get("placeholder", ""),
            key=key, label_visibility="collapsed")
    if wtype == "score":
        not_taken = st.checkbox("Not taken yet", key=f"xai_nt_{step}")
        if not_taken:
            st.caption("This component will be excluded from the calculation.")
            return -1.0
        max_m = float(kwargs.get("max", 20))
        return st.number_input("",
            min_value=0.0, max_value=max_m,
            value=round(max_m * 0.7, 1), step=0.5,
            key=key, label_visibility="collapsed")
    return None


def _widget_type(flow: dict) -> tuple[str, dict]:
    """Return (widget_type, kwargs) for the current flow phase."""
    phase    = flow["phase"]
    wip      = flow.get("wip", {})
    comp_max = wip.get("comp_max", 20.0)

    if phase == "programme":     return "select", {"options": PROGRAMMES}
    if phase == "semester":      return "int",    {"lo": 1,   "hi": 9,   "default": 3}
    if phase == "num_courses":   return "int",    {"lo": 1,   "hi": 12,  "default": 5}
    if phase == "course_name":   return "text",   {"placeholder": "e.g. Machine Learning"}
    if phase == "course_credits":return "int",    {"lo": 1,   "hi": 6,   "default": 3}
    if phase == "course_num_comps": return "int", {"lo": 1,   "hi": 12,  "default": 4}
    if phase == "comp_name":     return "text",   {"placeholder": "e.g. Midterm"}
    if phase == "comp_max":      return "float",  {"lo": 1.0, "hi": 200.0, "default": 20.0}
    if phase == "comp_score":    return "score",  {"max": comp_max}
    if phase == "failed":        return "int",    {"lo": 0,   "hi": 30,  "default": 0}
    if phase == "total_credits": return "int",    {"lo": 0,   "hi": 250, "default": 45}
    return "text", {}


def _advance_flow(flow: dict, phase: str, ans, msgs: list) -> bool:
    """Update flow state for the given answer. Returns False if invalid."""
    wip = flow.setdefault("wip", {})

    if phase == "programme":
        flow["programme"] = str(ans)
        msgs.append(("user", str(ans)))
        flow["phase"] = "semester"

    elif phase == "semester":
        flow["semester"] = int(ans)
        msgs.append(("user", f"Semester {ans}"))
        flow["phase"] = "num_courses"

    elif phase == "num_courses":
        n = int(ans)
        flow["num_courses"] = n
        flow["courses"] = []
        flow["wip"] = {}
        msgs.append(("user", f"{n} course{'s' if n != 1 else ''}"))
        msgs.append(("ai", f"Collecting data for **{n} course{'s' if n != 1 else ''}**. Let's begin!"))
        flow["phase"] = "course_name"

    elif phase == "course_name":
        name = str(ans).strip()
        if not name:
            return False
        wip["course_name"] = name
        # Attempt to resolve credits from official catalog immediately
        _ce  = get_curriculum_engine()
        _code = _ce.match_course_code(name)
        if _code:
            _info = _ce.get_course(_code)
            if _info and _info.get("credits"):
                wip["course_credits"]          = int(_info["credits"])
                wip["course_credits_verified"] = True
                wip["course_matched_code"]     = _code
                msgs.append(("user", name))
                msgs.append(("ai",
                    f"Matched **{_code}** ({_info.get('title', name)}) — "
                    f"**{wip['course_credits']} credit hours** from the official catalog. "
                    f"How many assessment components does this course have?"))
                flow["phase"] = "course_num_comps"
                return True
        wip["course_credits_verified"] = False
        msgs.append(("user", name))
        flow["phase"] = "course_credits"

    elif phase == "course_credits":
        wip["course_credits"]          = int(ans)
        wip["course_credits_verified"] = False  # student-reported, not from official catalog
        msgs.append(("user", f"{ans} credit hours (unverified — student-reported)"))
        flow["phase"] = "course_num_comps"

    elif phase == "course_num_comps":
        wip["num_comps"] = int(ans)
        wip["components"] = []
        msgs.append(("user", f"{ans} components"))
        msgs.append(("ai",
            f"**{wip['course_name']}** ({wip['course_credits']} cr) — **{ans} component(s)**. Let's enter each one."))
        flow["phase"] = "comp_name"

    elif phase == "comp_name":
        name = str(ans).strip()
        if not name:
            return False
        wip["comp_name"] = name
        msgs.append(("user", name))
        flow["phase"] = "comp_max"

    elif phase == "comp_max":
        wip["comp_max"] = float(ans)
        msgs.append(("user", f"Max: {ans} marks"))
        flow["phase"] = "comp_score"

    elif phase == "comp_score":
        score = float(ans)
        max_m = wip["comp_max"]
        comp  = {"name": wip.pop("comp_name"), "max": max_m, "score": score}
        wip.pop("comp_max", None)
        wip["components"].append(comp)

        if score < 0:
            msgs.append(("user", "Not taken yet"))
            msgs.append(("ai", f"**{comp['name']}** — not taken yet, will be excluded."))
        else:
            pct = score / max_m * 100
            msgs.append(("user", f"{score:.0f} / {max_m:.0f}"))
            msgs.append(("ai", f"**{comp['name']}**: {score:.0f}/{max_m:.0f} = **{pct:.0f}%**"))

        done_comps = len(wip["components"])
        if done_comps < wip["num_comps"]:
            flow["phase"] = "comp_name"
        else:
            taken     = [c for c in wip["components"] if c["score"] >= 0]
            t_score   = sum(c["score"] for c in taken)
            t_max_tkn = sum(c["max"]   for c in taken)
            t_max_all = sum(c["max"]   for c in wip["components"])
            remaining = t_max_all - t_max_tkn
            cur_pct   = t_score / t_max_tkn * 100 if t_max_tkn > 0 else 0

            course = {
                "name":             wip["course_name"],
                "credits":          wip["course_credits"],
                "components":       wip["components"],
                "current_pct":      round(cur_pct, 1),
                "total_score":      round(t_score, 1),
                "total_max_taken":  round(t_max_tkn, 1),
                "remaining_marks":  round(remaining, 1),
            }
            flow["courses"].append(course)
            flow["wip"] = {}

            summary = (f"**{course['name']}** ({course['credits']} cr) — "
                       f"**{t_score:.0f}/{t_max_tkn:.0f}** = **{cur_pct:.0f}%**")
            if remaining > 0:
                summary += f" | Remaining: {remaining:.0f} marks"
            msgs.append(("ai", summary))

            n_done  = len(flow["courses"])
            n_total = flow["num_courses"]
            if n_done < n_total:
                msgs.append(("ai", f"Moving on to **Course {n_done + 1} of {n_total}**."))
                flow["phase"] = "course_name"
            else:
                msgs.append(("ai", "All courses entered! Two quick questions remain…"))
                flow["phase"] = "failed"

    elif phase == "failed":
        flow["failed_courses"] = int(ans)
        msgs.append(("user", str(ans)))
        msgs.append(("ai", "No previously failed courses." if int(ans) == 0
                     else f"**{ans}** previously failed course(s) — noted."))
        flow["phase"] = "total_credits"

    elif phase == "total_credits":
        flow["total_credits"] = int(ans)
        msgs.append(("user", f"{ans} credits total"))
        msgs.append(("ai",
            "All information collected! "
            "Click **Run Full Analysis** below to get your predicted GPA, "
            "risk level, and personalised recommendations."))
        flow["phase"] = "done"

    return True


def _show_course_summary(flow: dict) -> None:
    courses = flow.get("courses", [])
    if not courses:
        return
    st.markdown("#### Course Summary")
    rows = []
    for c in courses:
        taken = [x for x in c["components"] if x["score"] >= 0]
        t_s   = sum(x["score"] for x in taken)
        t_m   = sum(x["max"]   for x in taken)
        a_m   = sum(x["max"]   for x in c["components"])
        rem   = a_m - t_m
        pct   = t_s / t_m * 100 if t_m > 0 else 0
        verified = c.get("course_credits_verified", False)
        rows.append({
            "Course":          c["name"],
            "Credits":         f"{c['credits']}" + ("" if verified else " ⚠️ unverified"),
            "Score (taken)":   f"{t_s:.0f} / {t_m:.0f}",
            "Current %":       f"{pct:.0f}%",
            "Remaining Marks": f"{rem:.0f}",
            "Components":      len(c["components"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chat_tab():
    """
    Primary data entry tab. Attempts Google Classroom import first.
    Falls back to manual entry if OAuth is not configured or student declines.
    """
    import os
    st.markdown('<p class="xai-section-title">Academic Data — Import or Enter Manually</p>',
                unsafe_allow_html=True)

    # ── OAuth callback detection ──────────────────────────────────────────────
    # Session state is LOST when Google redirects back (new browser session).
    # Solution: embed the code_verifier inside the OAuth state parameter so
    # Google echoes it back in the redirect URL — no session state needed.
    params = st.query_params
    if "code" in params and not st.session_state.get("xai_google_token"):
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
        # Google normalises scope names on return (e.g. "email" → "userinfo.email").
        # Relax the strict scope-equality check so the exchange doesn't fail.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", _detect_redirect_uri())

        # State format: "<random>|<code_verifier>" — we packed the verifier in on sign-in
        raw_state = params.get("state", "")
        if "|" in raw_state:
            code_verifier = raw_state.split("|", 1)[1]
        else:
            code_verifier = ""

        _tok, _err = None, ""
        try:
            _flow = build_oauth_flow(redirect_uri)
            if _flow is None:
                _err = "OAuth not configured — check GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env."
            else:
                # Inject echoed state so oauthlib state-check passes on the recreated Flow
                _flow.oauth2session._state = raw_state
                fetch_kwargs = {"code": params["code"]}
                if code_verifier:
                    fetch_kwargs["code_verifier"] = code_verifier
                _flow.fetch_token(**fetch_kwargs)
                _tok = _flow.credentials.token
        except Exception as _exc:
            _err = str(_exc)

        st.query_params.clear()
        if _tok:
            st.session_state.xai_google_token = _tok
            st.session_state.xai_import_mode  = "classroom"
            st.rerun()
        else:
            st.session_state.xai_import_error = f"OAuth token exchange failed: {_err}"

    # ── Show import error if present ──────────────────────────────────────────
    if st.session_state.get("xai_import_error"):
        st.error(st.session_state.xai_import_error)
        if st.button("Clear error", key="xai_clear_err"):
            st.session_state.xai_import_error = ""
            st.rerun()

    # ── Reset button ──────────────────────────────────────────────────────────
    col_rst, _ = st.columns([1, 5])
    with col_rst:
        if st.button("↺ Reset", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("xai_")]:
                del st.session_state[k]
            st.rerun()

    # ── If already authenticated and profile not yet built → import ──────────
    if st.session_state.get("xai_google_token") and not st.session_state.get("xai_classroom_profile"):
        _run_classroom_import()

    if st.session_state.get("xai_classroom_profile"):
        _render_classroom_import_status()
        return

    # ── Google Classroom sign-in card ─────────────────────────────────────────
    google_configured = _GOOGLE_AVAILABLE and bool(os.getenv("GOOGLE_CLIENT_ID"))

    if google_configured:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(66,133,244,.12),rgba(124,58,237,.10));
             border:1px solid rgba(66,133,244,.35);border-radius:18px;padding:28px 32px;
             margin:12px 0 20px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
                <span style="font-size:1.8rem;">🎓</span>
                <div>
                    <p style="color:#F3EFFF;font-weight:800;font-size:1.15rem;margin:0;">
                        Connect Google Classroom
                    </p>
                    <p style="color:#93C5FD;font-size:.82rem;margin:4px 0 0;">
                        Zewail City University of Science and Technology
                    </p>
                </div>
            </div>
            <p style="color:#CBD5E1;font-size:.88rem;line-height:1.6;margin:0;">
                Sign in with your <strong style="color:#F3EFFF;">@zewailcity.edu.eg</strong> account.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Sign in with Google Classroom →", type="primary",
                     use_container_width=True, key="gc_signin_btn"):
            redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", _detect_redirect_uri())
            flow = build_oauth_flow(redirect_uri)
            if flow:
                code_verifier, code_challenge = _pkce_pair()
                packed_state = f"{_secrets.token_urlsafe(16)}|{code_verifier}"
                auth_url, _ = flow.authorization_url(
                    prompt="select_account",
                    access_type="online",
                    state=packed_state,
                    code_challenge=code_challenge,
                    code_challenge_method="S256",
                )
                st.code(auth_url, language=None)
                st.link_button("Open Google Sign-in →", auth_url)
            else:
                st.error("Failed to build OAuth flow. Check GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env.")

    else:
        st.warning("Google Classroom integration is not configured. "
                   "Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env.")


def _detect_redirect_uri() -> str:
    """Detect the current Streamlit app URL for OAuth redirect."""
    import os
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    return f"http://localhost:{port}"


def _run_classroom_import() -> None:
    """Fetch Classroom data and build StudentXAIProfile. Called once after OAuth."""
    token = st.session_state.get("xai_google_token", "")
    if not token:
        return

    with st.spinner("Importing your academic history from Google Classroom…"):
        try:
            importer = ClassroomImporter(token)
            email    = importer.get_user_email()
            records  = importer.fetch_all_records()

            builder = StudentProfileBuilder()
            profile = builder.build(records, user_email=email)

            st.session_state.xai_google_email      = email
            st.session_state.xai_classroom_profile = profile
            st.session_state.xai_import_error      = ""
        except Exception as exc:
            st.session_state.xai_import_error = f"Import failed: {exc}"
            st.session_state.xai_google_token = None


def _render_classroom_import_status() -> None:
    """Import summary focused on course-level performance analytics."""
    p: StudentXAIProfile = st.session_state.xai_classroom_profile
    email = st.session_state.get("xai_google_email", "")

    graded = [r for r in p.course_records if r.overall_pct > 0]
    total_courses = len(p.course_records)
    graded_count  = len(graded)

    # Performance category counts — all courses so donut shows full 50
    cat_counts = {}
    for r in p.course_records:
        cat_counts[r.performance_category] = cat_counts.get(r.performance_category, 0) + 1

    traj_map = {"improving": ("📈", "#6EE7B7"), "stable": ("➡️", "#93C5FD"),
                "declining": ("📉", "#FCA5A5"), "volatile": ("⚡", "#FDE68A")}
    traj_icon, traj_clr = traj_map.get(p.performance_trend, ("➡️", "#93C5FD"))

    # ── Import banner ─────────────────────────────────────────────────────────
    # Count unique academic terms (not unique section strings)
    term_set    = set(
        r.term_label or r.course_section
        for r in p.course_records
        if (r.term_label or r.course_section)
    )
    n_terms = len(term_set)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(13,7,32,.97),rgba(30,15,60,.97));'
        f'border:1px solid rgba(99,102,241,.35);border-radius:16px;padding:14px 20px;'
        f'margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;'
        f'flex-wrap:wrap;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:1.2rem;">📊</span>'
        f'<span style="color:#A5B4FC;font-weight:800;font-size:1rem;">Learning Analytics Ready</span>'
        f'<span style="color:#94A3B8;font-size:.80rem;">&nbsp;—&nbsp;{email}</span>'
        f'</div>'
        f'<span style="color:#CBD5E1;font-size:.82rem;">'
        f'{total_courses} courses &nbsp;·&nbsp; {n_terms} academic terms &nbsp;·&nbsp; {graded_count} graded'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Four Coursework Analytics Indices ────────────────────────────────────
    st.markdown(
        '<p style="color:#A5B4FC;font-size:.78rem;font-weight:700;'
        'text-transform:uppercase;letter-spacing:.08em;margin:10px 0 6px;">'
        '📐 Coursework Analytics Indices</p>',
        unsafe_allow_html=True,
    )
    i1, i2, i3, i4 = st.columns(4)
    i1.metric(
        "Performance Index",
        f"{p.coursework_performance_index:.1f}/100" if p.coursework_performance_index > 0 else "—",
        help="Credit-weighted average coursework score across all graded courses.",
    )
    i2.metric(
        "Engagement Index",
        f"{p.engagement_index:.1f}/100" if p.engagement_index > 0 else "—",
        help="Average of attendance, assignment, and lab scores — measures active participation.",
    )
    i3.metric(
        "Consistency Index",
        f"{p.consistency_index:.1f}/100" if p.consistency_index > 0 else "—",
        help="How consistent performance is across courses. Higher = more stable.",
    )
    i4.metric(
        "Risk Index",
        f"{p.risk_index:.1f}%" if p.risk_index > 0 else "0%",
        help="Percentage of graded courses in weak or at-risk performance categories.",
    )

    # ── Performance distribution — Plotly donut chart ────────────────────────
    _CAT_CFG = [
        ("excellent",   "#22C55E", "Excellent ≥85%"),
        ("strong",      "#3B82F6", "Strong ≥70%"),
        ("average",     "#A78BFA", "Average ≥55%"),
        ("weak",        "#F59E0B", "Weak ≥40%"),
        ("at_risk",     "#EF4444", "At Risk <40%"),
        ("in_progress", "#64748B", "In Progress"),
        ("no_data",     "#2D3748", "No Data"),
    ]
    _donut_labels = []
    _donut_values = []
    _donut_colors = []
    for cat, clr, lbl in _CAT_CFG:
        n = cat_counts.get(cat, 0)
        if n > 0:
            _donut_labels.append(lbl)
            _donut_values.append(n)
            _donut_colors.append(clr)

    if _donut_values:
        _total_shown = sum(_donut_values)
        _fig_donut = go.Figure(go.Pie(
            labels=_donut_labels,
            values=_donut_values,
            hole=0.62,
            marker=dict(colors=_donut_colors,
                        line=dict(color="rgba(13,7,32,1)", width=3)),
            textinfo="percent",
            textfont=dict(size=11, color="#F3EFFF"),
            hovertemplate="<b>%{label}</b><br>%{value} courses (%{percent})<extra></extra>",
            sort=False,
        ))
        _fig_donut.add_annotation(
            text=f"<b style='font-size:22px'>{_total_shown}</b><br><span style='font-size:11px'>Courses</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#F3EFFF", size=14),
        )
        _fig_donut.update_layout(
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                font=dict(color="#CBD5E1", size=11),
                bgcolor="rgba(0,0,0,0)",
                orientation="v",
                x=1.02, y=0.5,
                xanchor="left",
                yanchor="middle",
            ),
            showlegend=True,
        )
        st.markdown(
            '<p style="color:#A5B4FC;font-size:.78rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:.08em;margin:14px 0 4px;">'
            '🎯 Course Performance Distribution</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ── Performance trend pill (skip "volatile" — too few semesters to be meaningful) ──
    if p.performance_trend and p.performance_trend != "volatile":
        _traj_map = {
            "improving": ("📈", "#6EE7B7", "Improving"),
            "stable":    ("➡️",  "#93C5FD", "Stable"),
            "declining": ("📉", "#FCA5A5", "Declining"),
        }
        t_icon, t_clr, t_label = _traj_map.get(p.performance_trend, ("➡️", "#93C5FD", p.performance_trend.capitalize()))
        st.markdown(
            f'<div style="margin:4px 0 4px;font-size:.84rem;color:#CBD5E1;">'
            f'Performance trend across semesters: '
            f'<span style="color:{t_clr};font-weight:700;">{t_icon} {t_label}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:#94A3B8;">{p.programme or "—"}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Credit Audit Report ───────────────────────────────────────────────────
    mr = getattr(p, "matching_report", {})
    if mr:
        total_c     = mr.get("total_courses", 0)
        raw_c       = mr.get("raw_course_count", total_c)
        by_sec      = mr.get("matched_by_section", 0)
        by_name     = mr.get("matched_by_name", 0)
        unmatched   = mr.get("still_unmatched", 0)
        duplicates  = mr.get("duplicates_merged", 0)
        total_cr    = mr.get("total_verified_credits", 0)
        accuracy    = mr.get("accuracy_pct", 0.0)
        with st.expander(
            f"📋 Credit Audit Report — {total_cr} verified credits · {accuracy:.0f}% matched ({unmatched} unresolved)",
            expanded=(unmatched > 0),
        ):
            ca1, ca2, ca3, ca4, ca5 = st.columns(5)
            ca1.metric("Classroom Entries",  raw_c,
                       help="Total course entries imported from Google Classroom")
            ca2.metric("After Dedup",        total_c,
                       help=f"{duplicates} duplicate/retake/lab entries merged")
            ca3.metric("Verified Credits",   total_cr,
                       help="Sum of catalog credits for all matched courses")
            ca4.metric("Matched",            by_sec + by_name,
                       help=f"By section code: {by_sec}  ·  By title/alias: {by_name}")
            ca5.metric("Unresolved",         unmatched,
                       delta=f"-{unmatched}" if unmatched else None,
                       delta_color="inverse")

            if duplicates > 0:
                st.caption(
                    f"ℹ️ {duplicates} Classroom entr{'ies' if duplicates != 1 else 'y'} merged "
                    f"(retakes kept best attempt · lecture+lab deduplicated to one course entry)."
                )

            matched_list = mr.get("matched_courses", [])
            if matched_list:
                import pandas as pd
                st.markdown("**Matched Courses (handbook-verified)**")
                st.dataframe(
                    pd.DataFrame(matched_list),
                    use_container_width=True,
                    hide_index=True,
                )

            if mr.get("unmatched_names"):
                st.markdown("**Unresolved courses (not in handbook):**")
                for name in mr["unmatched_names"]:
                    if name and name.strip():
                        st.markdown(f"- {name}")

    if p.credits_unresolved > 0 and not mr:
        st.caption(
            f"ℹ️ {p.credits_unresolved} course(s) could not be matched to the official catalog "
            f"and are excluded from curriculum grouping."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NOTE: No pass/fail counts, no credit totals, no GPA, no transcript status.
    # This panel describes HOW the student is performing, not whether they passed.
    # ──────────────────────────────────────────────────────────────────────────

    if not st.session_state.xai_analysed:
        if st.button("🔍  Run Full XAI Analysis", type="primary", use_container_width=True,
                     key="run_xai_btn"):
            _run_analysis_from_classroom()
            st.rerun()
    else:
        st.markdown("""
        <div style="background:rgba(16,185,129,.10);border:1px solid rgba(16,185,129,.30);
             border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.2rem;">✅</span>
            <span style="color:#6EE7B7;font-weight:600;font-size:.9rem;">
                Analysis complete — switch to <strong>Academic Dashboard</strong> to view results.
            </span>
        </div>
        """, unsafe_allow_html=True)


def _render_manual_entry_flow() -> None:
    """Existing manual course-entry chat flow (unchanged from original)."""
    flow = st.session_state.xai_flow
    msgs = st.session_state.xai_messages
    step = st.session_state.xai_step

    for role, msg in msgs:
        tag = "xai-chat-ai" if role == "ai" else "xai-chat-msg"
        ico = "🤖" if role == "ai" else "👤"
        st.markdown(f'<div class="{tag}">{ico} {_md(msg)}</div>', unsafe_allow_html=True)

    if flow["phase"] == "done":
        if not st.session_state.xai_analysed:
            _show_course_summary(flow)
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                _run_analysis()
                st.rerun()
        else:
            st.success("Analysis complete! Switch to Academic Dashboard.")
        return

    q_text = _chat_question(flow)
    wtype, wkwargs = _widget_type(flow)

    with st.container():
        st.markdown(f'<div class="xai-chat-ai">🤖</div>', unsafe_allow_html=True)
        st.markdown(q_text)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        ans = _render_chat_widget(wtype, wkwargs, step)
    with col_btn:
        if st.button("Next ➜", type="primary", use_container_width=True):
            ok = _advance_flow(flow, flow["phase"], ans, msgs)
            if not ok:
                st.warning("Please fill in a value before continuing.")
            else:
                st.session_state.xai_step += 1
                st.rerun()


def _run_analysis_from_classroom() -> None:
    """
    Run the full XAI pipeline from an imported StudentXAIProfile.
    Source of truth: Google Classroom data via ClassroomImporter.
    The study plan is used only for curriculum gap context, not for inference.
    """
    cp: StudentXAIProfile = st.session_state.xai_classroom_profile
    if cp is None:
        return

    builder  = StudentProfileBuilder()
    features = builder.profile_to_feature_dict(cp)

    # Build legacy profile dict for downstream compatibility
    # avg_final here maps to avg_predicted_final (Classroom-based estimate, not official exam)
    profile = {
        "programme":           cp.programme,
        "semester":            cp.actual_semester_number,
        "avg_attendance":      cp.avg_attendance,
        "avg_assignments":     cp.avg_assignments,
        "avg_quizzes":         cp.avg_quizzes,
        "avg_labs":            cp.avg_labs,
        "avg_midterm":         cp.avg_midterm,
        "avg_final":           cp.avg_predicted_final,   # PREDICTED — not official
        "failed_courses":      cp.insufficient_data_count,
        "credits_registered":  cp.credits_total_attempted,
        "credits_passed":      cp.credits_completed_coursework,
        "passed_course_codes": cp.completed_codes,
        "failed_course_codes": cp.insufficient_data_codes,
    }
    st.session_state.xai_profile = profile

    with st.spinner("Running XAI analysis on Classroom data…"):
        eng    = get_engine()
        result = eng.explain_student(features, cp.user_email or cp.programme or "Student")

        curr_eng = get_curriculum_engine()
        curriculum_features = curr_eng.compute_features(
            programme         = cp.programme,
            passed_codes      = cp.completed_codes,
            failed_codes      = cp.insufficient_data_codes,
            credits_passed    = float(cp.credits_completed_coursework),
            semester          = cp.actual_semester_number,
            total_credits_reg = float(cp.credits_total_attempted),
        )
        curriculum_narratives = curr_eng.get_curriculum_narratives(
            curriculum_features = curriculum_features,
            programme           = cp.programme,
            passed_codes        = cp.completed_codes,
            failed_codes        = cp.insufficient_data_codes,
            predicted_gpa       = result.predicted_gpa,
        )
        curriculum_recs = curr_eng.generate_curriculum_recs(
            programme           = cp.programme,
            curriculum_features = curriculum_features,
            passed_codes        = cp.completed_codes,
            failed_codes        = cp.insufficient_data_codes,
            predicted_gpa       = result.predicted_gpa,
            semester            = cp.actual_semester_number,
        )
        graduation_status = curr_eng.get_graduation_status(
            programme      = cp.programme,
            credits_passed = float(cp.credits_completed_coursework),
            semester       = cp.actual_semester_number,
            predicted_gpa  = result.predicted_gpa,
        )
        result.curriculum_narratives = curriculum_narratives

        curriculum_shap_values = curr_eng.get_curriculum_shap_values(
            curriculum_features = curriculum_features,
            predicted_gpa       = result.predicted_gpa,
        )
        curriculum_feature_table = curr_eng.get_curriculum_feature_table(
            curriculum_features = curriculum_features,
            programme           = cp.programme,
        )
        recs = generate_recommendations(
            features,
            result.predicted_gpa,
            result.predicted_risk,
            result.top_negative,
            result.top_positive,
            curriculum_recs = curriculum_recs,
        )

    st.session_state.xai_result           = result
    st.session_state.xai_recs             = recs
    st.session_state.xai_analysed         = True
    st.session_state.xai_curriculum       = curriculum_features
    st.session_state.xai_graduation       = graduation_status
    st.session_state.xai_passed_codes     = cp.completed_codes
    st.session_state.xai_failed_codes     = cp.insufficient_data_codes
    st.session_state.xai_curriculum_shap  = curriculum_shap_values
    st.session_state.xai_curriculum_table = curriculum_feature_table


def _run_analysis():
    """Compute features from course-based flow, run XAI pipeline, add curriculum intelligence."""
    flow     = st.session_state.xai_flow
    profile  = _flow_to_profile(flow)
    features = _profile_to_features(profile)

    st.session_state.xai_profile = profile

    with st.spinner("Running AI analysis…"):
        # ── 1. Standard GPA/risk/SHAP pipeline (unchanged) ──────────────────
        eng    = get_engine()
        result = eng.explain_student(features, profile.get("programme", "Student"))

        # ── 2. Curriculum Intelligence Layer ─────────────────────────────────
        curr_eng = get_curriculum_engine()
        programme     = profile.get("programme", "CSAI")
        passed_codes  = profile.get("passed_course_codes", [])
        failed_codes  = profile.get("failed_course_codes", [])
        credits_passed = features.get("credits_passed", 0)
        semester       = int(features.get("semester", 1))

        curriculum_features = curr_eng.compute_features(
            programme      = programme,
            passed_codes   = passed_codes,
            failed_codes   = failed_codes,
            credits_passed = credits_passed,
            semester       = semester,
            total_credits_reg = features.get("credits_registered", 45),
        )
        curriculum_narratives = curr_eng.get_curriculum_narratives(
            curriculum_features = curriculum_features,
            programme           = programme,
            passed_codes        = passed_codes,
            failed_codes        = failed_codes,
            predicted_gpa       = result.predicted_gpa,
        )
        curriculum_recs = curr_eng.generate_curriculum_recs(
            programme           = programme,
            curriculum_features = curriculum_features,
            passed_codes        = passed_codes,
            failed_codes        = failed_codes,
            predicted_gpa       = result.predicted_gpa,
            semester            = semester,
        )
        graduation_status = curr_eng.get_graduation_status(
            programme      = programme,
            credits_passed = credits_passed,
            semester       = semester,
            predicted_gpa  = result.predicted_gpa,
        )

        # Attach curriculum narratives to result (backwards-compatible field)
        result.curriculum_narratives = curriculum_narratives

        # ── 3. Curriculum SHAP-equivalent values ──────────────────────────────
        curriculum_shap_values = curr_eng.get_curriculum_shap_values(
            curriculum_features = curriculum_features,
            predicted_gpa       = result.predicted_gpa,
        )

        # ── 4. Curriculum feature table for display ───────────────────────────
        curriculum_feature_table = curr_eng.get_curriculum_feature_table(
            curriculum_features = curriculum_features,
            programme           = programme,
        )

        # ── 5. Unified recommendations (curriculum critical first, then SHAP) ─
        recs = generate_recommendations(
            features,
            result.predicted_gpa,
            result.predicted_risk,
            result.top_negative,
            result.top_positive,
            curriculum_recs = curriculum_recs,
        )

    st.session_state.xai_result            = result
    st.session_state.xai_recs              = recs
    st.session_state.xai_analysed          = True
    st.session_state.xai_curriculum        = curriculum_features
    st.session_state.xai_graduation        = graduation_status
    st.session_state.xai_passed_codes      = passed_codes
    st.session_state.xai_failed_codes      = failed_codes
    st.session_state.xai_curriculum_shap   = curriculum_shap_values
    st.session_state.xai_curriculum_table  = curriculum_feature_table

    st.session_state.xai_messages.append((
        "ai",
        f"Analysis complete! "
        f"**Predicted GPA: {result.predicted_gpa:.2f}** | "
        f"**Risk Level: {result.risk_label}**  \n"
        "Switch to the **Academic Dashboard** tab to see the full breakdown.",
    ))


# ════════════════════════════════════════════════════════════════════════════════
#  Tab 2: Academic Dashboard
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
#  Classroom-sourced analytics panels
# ════════════════════════════════════════════════════════════════════════════════


def _render_course_performance_table() -> None:
    """
    Course-level performance table from actual Classroom history.
    Groups courses as: Strong (≥85%) | Passing (70–85%) | Weak (60–70%) |
    At Risk (<60%, in progress) | Failed
    All values sourced directly from Classroom grade data.
    """
    cp: StudentXAIProfile = st.session_state.get("xai_classroom_profile")
    if cp is None or not cp.course_records:
        return

    st.markdown('<p class="xai-section-title">Course Performance History</p>',
                unsafe_allow_html=True)

    # Build table rows — Classroom coursework data only, NOT academic outcomes
    _STATUS_LABEL = {
        "completed_coursework":   "✅ Completed Coursework",
        "coursework_in_progress": "🔄 In Progress",
        "insufficient_data":      "📊 Insufficient Data",
    }
    _PERF_LABEL = {
        "excellent":   "🌟 Excellent  (≥85%)",
        "strong":      "💪 Strong     (≥70%)",
        "average":     "📈 Average    (≥55%)",
        "weak":        "⚠️ Weak       (≥40%)",
        "at_risk":     "🔴 At Risk    (<40%)",
        "in_progress": "🔄 In Progress",
        "no_data":     "—",
    }
    rows = []
    for r in cp.course_records:
        credits_cell = str(r.credits) if r.credits_verified else f"{r.credits or '?'} ⚠️"
        rows.append({
            "Course":               r.course_name,
            "Code":                 r.matched_code or "❓ unmatched",
            "Section":              r.course_section or "—",
            "Credits":              credits_cell,
            "Status":               _STATUS_LABEL.get(r.pass_status, r.pass_status),
            "Coursework Score":     f"{r.overall_pct:.0f}%" if r.overall_pct > 0 else "—",
            "Assignments":          f"{r.assignments_avg:.0f}%" if r.assignments_avg > 0 else "—",
            "Quizzes":              f"{r.quizzes_avg:.0f}%" if r.quizzes_avg > 0 else "—",
            "Labs":                 f"{r.labs_avg:.0f}%" if r.labs_avg > 0 else "—",
            "Midterm":              f"{r.midterm_score:.0f}%" if r.midterm_score > 0 else "—",
            "Attendance":           f"{r.attendance_pct:.0f}%" if r.attendance_pct > 0 else "—",
            "Pred. Final (info)":   (
                f"{r.predicted_final_score:.1f}/40" if r.predicted_final_score > 0 else "—"
            ),
            "Performance Category": _PERF_LABEL.get(r.performance_category, "—"),
        })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary chips — performance categories from Classroom data only
    n_strong  = len(cp.strong_courses)
    n_weak    = len(cp.weak_courses)
    n_risk    = len(cp.risk_courses)
    n_insuff  = cp.insufficient_data_count
    if n_strong or n_weak or n_risk or n_insuff:
        parts = []
        if n_strong: parts.append(f"**{n_strong}** excellent/strong (≥70%)")
        if n_weak:   parts.append(f"**{n_weak}** weak (40–55%)")
        if n_risk:   parts.append(f"**{n_risk}** at risk (<40%)")
        if n_insuff: parts.append(f"**{n_insuff}** insufficient data")
        st.markdown(" &nbsp;|&nbsp; ".join(parts))
    st.caption(
        "⚠️ Performance categories describe observed Classroom activity only. "
        "They do NOT represent official academic pass/fail outcomes. "
        "Official grades are issued by the registrar after final exams."
    )


def _render_semester_gpa_trend() -> None:
    """
    Semester-by-semester coursework performance trend from Classroom history.
    Each point is the credit-weighted average coursework score for that semester.
    These are Classroom performance scores — NOT official GPA values.
    """
    cp: StudentXAIProfile = st.session_state.get("xai_classroom_profile")
    if cp is None or len(cp.semester_performance_scores) < 2:
        return

    st.markdown('<p class="xai-section-title">Semester Performance Trend (Coursework)</p>',
                unsafe_allow_html=True)

    labels  = cp.semester_labels or [f"Sem {i+1}" for i in range(len(cp.semester_performance_scores))]
    credits = cp.semester_credits or [0] * len(cp.semester_performance_scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=cp.semester_performance_scores,
        mode="lines+markers+text",
        line=dict(color=PURPLE_L, width=3),
        marker=dict(size=10, color=PURPLE_L),
        text=[f"{s:.0f}%" for s in cp.semester_performance_scores],
        textposition="top center",
        textfont=dict(color="#F3EFFF", size=11),
        name="Coursework Score",
        hovertemplate="<b>%{x}</b><br>Coursework Score: %{y:.1f}%<extra></extra>",
    ))
    # Reference lines for performance categories
    fig.add_hline(y=70.0, line_dash="dot", line_color="rgba(39,174,96,.7)",
                  annotation_text="Strong (70%)",
                  annotation_font_color="#F39C12", annotation_position="bottom right")
    # Credit load bars (secondary axis)
    if any(c > 0 for c in credits):
        fig.add_trace(go.Bar(
            x=labels, y=credits,
            name="Credit Load",
            marker_color="rgba(124,58,237,.2)",
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Credits: %{y}<extra></extra>",
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y", side="right",
                title=dict(text="Credits", font=dict(size=10, color="#A78BFA")),
                tickfont=dict(color="#A78BFA", size=9),
                showgrid=False,
                range=[0, max(credits) * 3],
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(t=20, b=30, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,7,32,0.6)",
        font_color="#F3EFFF",
        yaxis=dict(range=[0, 105], gridcolor="rgba(124,58,237,.15)",
                   title=dict(text="Coursework Score (%)", font=dict(size=10, color="#A78BFA"))),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A78BFA", size=10),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key="sem_gpa_trend")

    traj_msgs = {
        "improving": "Your coursework scores have been improving over recent semesters.",
        "declining": "Your coursework scores have been declining — review weak courses.",
        "volatile":  "Your coursework performance fluctuates significantly between semesters.",
        "stable":    "Your coursework performance has been consistent across semesters.",
    }
    st.caption(traj_msgs.get(cp.performance_trend, ""))
    st.caption(
        "⚠️ These are Classroom coursework scores, not official GPA values. "
        "Official GPA is issued by the registrar after final exams."
    )


def _render_dashboard_tab():
    st.markdown('<p class="xai-section-title">Academic Health Dashboard</p>', unsafe_allow_html=True)

    result = st.session_state.get("xai_result")
    if result is None:
        st.info("Complete the AI Analysis Chat first to see your dashboard.")
        return

    profile  = st.session_state.xai_profile
    features = _profile_to_features(profile)

    # ── Learning Analytics banner ────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.35);
         border-radius:12px;padding:10px 16px;margin-bottom:14px;font-size:.83rem;color:#A5B4FC;">
        📊 <strong>Learning Analytics &amp; XAI</strong> — All metrics are derived from
        Google Classroom coursework activity (assignments, quizzes, labs, midterms, attendance).
        Google Classroom does <strong>not</strong> contain official final exam results or
        registrar-confirmed grades.  These are <strong>coursework analytics</strong>,
        not official academic transcripts.
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Key metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cp_profile = st.session_state.get("xai_classroom_profile")
        cpi = cp_profile.coursework_performance_index if cp_profile else result.academic_health
        fig = _gauge_chart(cpi, 100, "Coursework Perf. Index",
                           "#27AE60" if cpi >= 70 else ("#F39C12" if cpi >= 55 else "#E74C3C"))
        st.plotly_chart(fig, use_container_width=True, key="gauge_gpa")

    with c2:
        health = result.academic_health
        fig = _gauge_chart(health, 100, "Academic Health",
                           "#27AE60" if health >= 65 else ("#F39C12" if health >= 40 else "#E74C3C"))
        st.plotly_chart(fig, use_container_width=True, key="gauge_health")

    with c3:
        att = features["avg_attendance"]
        fig = _gauge_chart(att, 100, "Attendance %",
                           "#27AE60" if att >= 80 else ("#F39C12" if att >= 70 else "#E74C3C"))
        st.plotly_chart(fig, use_container_width=True, key="gauge_att")

    with c4:
        overall = features["avg_overall"]
        fig = _gauge_chart(overall, 100, "Overall Score",
                           "#27AE60" if overall >= 80 else ("#F39C12" if overall >= 65 else "#E74C3C"))
        st.plotly_chart(fig, use_container_width=True, key="gauge_overall")

    # ── Risk badge ───────────────────────────────────────────────────────────────
    risk_color = result.risk_color
    st.markdown(f"""
    <div style="text-align:center;margin:8px 0 20px;">
        <span class="xai-risk-badge"
              style="background:{risk_color}22;border:2px solid {risk_color};color:{risk_color};">
            Risk Level: {result.risk_label}
        </span>
        <p style="color:#A78BFA;font-size:.82rem;margin-top:6px;">
            Confidence: Low {result.risk_proba[0]*100:.0f}% |
            Medium {result.risk_proba[1]*100:.0f}% |
            High {result.risk_proba[2]*100:.0f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Classroom-specific panels (only shown when Classroom data is present) ────
    if st.session_state.get("xai_classroom_profile"):
        _render_course_performance_table()
        _render_semester_gpa_trend()
        st.markdown("---")

    # ── Row 2: Radar + Bar chart ─────────────────────────────────────────────────
    col_radar, col_bar = st.columns(2)

    with col_radar:
        st.markdown("**Performance Profile** *(all values predicted/estimated)*")
        cats = ["Attendance", "Assignments", "Quizzes", "Labs", "Midterm", "Pred. Final (est.)"]
        vals = [
            features["avg_attendance"],
            features["avg_assignments"],
            features["avg_quizzes"],
            features["avg_labs"],
            features["avg_midterm"],
            features["avg_final"],   # model key "avg_final" = avg_predicted_final
        ]
        st.plotly_chart(_radar_chart(cats, vals), use_container_width=True, key="radar")

    with col_bar:
        st.markdown("**Score Component Breakdown**")
        st.plotly_chart(_trend_chart(features), use_container_width=True, key="bar_scores")

    # ── Row 3: Strengths and weaknesses ─────────────────────────────────────────
    col_str, col_weak = st.columns(2)

    with col_str:
        st.markdown("**Strengths**")
        for f in result.top_positive[:4]:
            label = f["feature"]
            sv    = f["shap_value"]
            fv    = f["feature_value"]
            st.markdown(f"""
            <div class="xai-rec-card" style="border-color:#27AE60;">
                <span style="color:#27AE60;font-weight:700;">+{sv:.3f}</span>
                &nbsp; {label} = {fv:.1f}
            </div>""", unsafe_allow_html=True)

    with col_weak:
        st.markdown("**Areas to Improve**")
        for f in result.top_negative[:4]:
            label = f["feature"]
            sv    = f["shap_value"]
            fv    = f["feature_value"]
            st.markdown(f"""
            <div class="xai-rec-card" style="border-color:#E74C3C;">
                <span style="color:#E74C3C;font-weight:700;">{sv:.3f}</span>
                &nbsp; {label} = {fv:.1f}
            </div>""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════════
#  Tab 3: Recommendations
# ════════════════════════════════════════════════════════════════════════════════

def _render_recommendations_tab():
    st.markdown('<p class="xai-section-title">Personalized Action Plan</p>', unsafe_allow_html=True)

    recs = st.session_state.get("xai_recs")
    if recs is None:
        st.info("Complete the AI Analysis Chat first.")
        return

    PRIO_COLOR = {1: "#E74C3C", 2: "#E67E22", 3: "#F1C40F", 4: "#27AE60"}
    PRIO_LABEL = {1: "Critical", 2: "High", 3: "Medium", 4: "Low"}

    _SKIP_CATEGORIES = {"Curriculum", "Prerequisites", "Graduation", "Degree Progress"}

    def _rec_card(r):
        color = PRIO_COLOR.get(r.priority, "#A78BFA")
        st.markdown(f"""
        <div class="xai-rec-card" style="border-color:{color};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;color:{color};">
                    {r.icon} {r.title}
                </span>
                <span style="background:{color}22;border:1px solid {color};border-radius:12px;
                      padding:2px 10px;font-size:0.72rem;color:{color};font-weight:700;">
                    {PRIO_LABEL.get(r.priority, "?")}
                </span>
            </div>
            <p style="color:#DDD6FE;margin:0 0 6px;font-size:.87rem;">{r.detail}</p>
            <p style="color:#A78BFA;margin:0;font-size:.80rem;font-style:italic;">{r.impact}</p>
        </div>
        """, unsafe_allow_html=True)

    academic_recs = [r for r in recs if r.category not in _SKIP_CATEGORIES]

    if academic_recs:
        st.markdown("""
        <div style="background:rgba(52,211,153,.06);border:1px solid rgba(52,211,153,.25);
                    border-radius:10px;padding:10px 16px;margin-bottom:16px;">
            <span style="color:#6EE7B7;font-weight:700;font-size:.95rem;">
                📊 Academic Performance Recommendations
            </span>
            <span style="color:#A78BFA;font-size:.80rem;margin-left:8px;">
                Based on AI prediction model and SHAP analysis
            </span>
        </div>
        """, unsafe_allow_html=True)
        for r in academic_recs:
            _rec_card(r)
    else:
        st.info("No recommendations generated. Complete the AI Analysis Chat first.")


# ════════════════════════════════════════════════════════════════════════════════
#  Tab 4: What-If Simulator
# ════════════════════════════════════════════════════════════════════════════════

def _render_whatif_tab():
    st.markdown('<p class="xai-section-title">What-If Scenario Simulator</p>', unsafe_allow_html=True)

    result = st.session_state.get("xai_result")
    if result is None:
        st.info("Complete the AI Analysis Chat first.")
        return

    profile  = st.session_state.xai_profile
    features = _profile_to_features(profile)

    st.markdown(
        "Adjust the sliders below to simulate how changes to your academic behaviour "
        "would affect your predicted GPA and risk level."
    )

    col_sliders, col_results = st.columns([1, 1])

    with col_sliders:
        st.markdown("**Adjust Your Inputs**")
        new_att   = st.slider("Attendance (%)",   40, 100, int(features["avg_attendance"]),   step=1, key="wi_att")
        new_asgn  = st.slider("Assignments",      20, 100, int(features["avg_assignments"]),  step=1, key="wi_asgn")
        new_quiz  = st.slider("Quizzes",          20, 100, int(features["avg_quizzes"]),      step=1, key="wi_quiz")
        new_labs  = st.slider("Labs",             20, 100, int(features["avg_labs"]),         step=1, key="wi_labs")
        new_mid   = st.slider("Midterm Score",    20, 100, int(features["avg_midterm"]),      step=1, key="wi_mid")
        new_fin   = st.slider("Final Exam Score", 20, 100, int(features["avg_final"]),        step=1, key="wi_fin")
        new_fail  = st.slider("Failed Courses",    0,  10, int(features["failed_courses"]),   step=1, key="wi_fail")

    changes = {
        "avg_attendance":  float(new_att),
        "avg_assignments": float(new_asgn),
        "avg_quizzes":     float(new_quiz),
        "avg_labs":        float(new_labs),
        "avg_midterm":     float(new_mid),
        "avg_final":       float(new_fin),
        "failed_courses":  float(new_fail),
    }

    wi_eng = get_whatif_engine()
    sim    = wi_eng.simulate(features, changes)

    with col_results:
        st.markdown("**Projected Outcome**")
        delta_sign = "+" if sim.delta_gpa >= 0 else ""
        delta_color = "#27AE60" if sim.delta_gpa >= 0 else "#E74C3C"

        st.markdown(f"""
        <div class="xai-metric-card" style="margin-bottom:12px;">
            <p class="xai-metric-val">{sim.new_gpa:.2f}</p>
            <p class="xai-metric-lbl">Projected GPA</p>
        </div>
        <div style="text-align:center;margin-bottom:12px;">
            <span style="font-size:1.6rem;font-weight:700;color:{delta_color};">
                {delta_sign}{sim.delta_gpa:.3f}
            </span>
            <span style="color:#A78BFA;font-size:.85rem;"> GPA change</span>
        </div>
        <div style="text-align:center;">
            <span class="xai-risk-badge"
                  style="background:{RISK_COLORS[sim.new_risk]}22;
                         border:2px solid {RISK_COLORS[sim.new_risk]};
                         color:{RISK_COLORS[sim.new_risk]};">
                {sim.new_label}
            </span>
            <p style="color:#A78BFA;font-size:.80rem;margin-top:4px;">
                (was: {sim.original_label})
            </p>
        </div>
        """, unsafe_allow_html=True)

        if sim.risk_direction == "improved":
            st.success(f"Risk level improved from {sim.original_label} to {sim.new_label}!")
        elif sim.risk_direction == "worsened":
            st.error(f"Risk level worsened from {sim.original_label} to {sim.new_label}.")

    # ── Multi-scenario comparison ────────────────────────────────────────────────
    st.divider()
    st.markdown("**Scenario Comparison**")

    preset_scenarios = [
        ({"avg_attendance": 90, "avg_midterm": 80, "avg_final": 78},   "Improve Attendance + Exams"),
        ({"avg_assignments": 85, "avg_quizzes": 80, "avg_labs": 85},    "Focus on Coursework"),
        ({"failed_courses":  0,  "avg_overall": 75},                    "No Failures + 75% Average"),
        ({"avg_attendance": 90, "avg_assignments": 85, "avg_midterm": 82,
          "avg_final": 80, "failed_courses": 0},                        "Full Improvement"),
    ]

    scenario_results = wi_eng.multi_scenario(
        features,
        [c for c, _ in preset_scenarios]
    )
    scenario_gpas   = [r.new_gpa for r in scenario_results]
    scenario_labels = [l for _, l in preset_scenarios]

    fig = _whatif_chart(sim.original_gpa, scenario_gpas, scenario_labels)
    st.plotly_chart(fig, use_container_width=True, key="whatif_chart")

    # ── Course Scenario Simulator ────────────────────────────────────────────────
    st.divider()
    st.markdown("**Course Scenario Simulator**")
    st.markdown(
        "Simulate how passing, failing, retaking, or postponing a course affects "
        "which future courses become available or blocked."
    )

    passed_codes = st.session_state.get("xai_passed_codes", [])
    failed_codes = st.session_state.get("xai_failed_codes", [])
    curriculum_f = st.session_state.get("xai_curriculum", {})
    credits_passed = profile.get("credits_passed", 0)
    semester       = profile.get("semester", 1)
    programme      = profile.get("programme", "CSAI")
    predicted_gpa  = result.predicted_gpa if result else 2.0

    all_known_codes = list(dict.fromkeys(passed_codes + failed_codes))  # dedup, preserve order

    if not all_known_codes:
        st.info("No curriculum course codes were matched from your entries. "
                "Enter course names or codes (e.g., CSAI101) in the Analysis Chat to enable this.")
    else:
        curr_col1, curr_col2 = st.columns([1, 1])

        with curr_col1:
            selected_code = st.selectbox(
                "Select a course",
                all_known_codes,
                key="curr_wi_course",
            )
            outcome = st.radio(
                "Simulate outcome",
                ["pass", "fail", "retake", "postpone"],
                horizontal=True,
                key="curr_wi_outcome",
            )
            run_curr_sim = st.button("Run Curriculum Scenario", key="btn_curr_wi")

        with curr_col2:
            if run_curr_sim:
                try:
                    from what_if_analysis.what_if_engine import simulate_curriculum_scenario
                    curr_result = simulate_curriculum_scenario(
                        base_features  = features,
                        programme      = programme,
                        passed_codes   = passed_codes,
                        failed_codes   = failed_codes,
                        credits_passed = credits_passed,
                        semester       = semester,
                        course_code    = selected_code,
                        outcome        = outcome,
                        predicted_gpa  = predicted_gpa,
                    )
                    st.session_state["_curr_wi_result"] = curr_result
                except Exception as e:
                    st.error(f"Simulation error: {e}")

            curr_wi = st.session_state.get("_curr_wi_result")
            if curr_wi:
                unblocked   = curr_wi.get("unblocked", [])
                new_blocked = curr_wi.get("new_blocked", [])

                st.markdown(f"""
                <div class="xai-card" style="padding:14px 18px;">
                    <div style="font-weight:700;color:#C4B5FD;font-size:.95rem;margin-bottom:8px;">
                        {curr_wi.get('scenario_name','Scenario')}
                    </div>
                    <div style="color:#DDD6FE;font-size:.87rem;">{curr_wi.get('curriculum_message','')}</div>
                </div>
                """, unsafe_allow_html=True)

                if unblocked:
                    st.success(f"Courses unblocked: {', '.join(unblocked)}")
                if new_blocked:
                    st.warning(f"Newly blocked: {', '.join(new_blocked)}")

    # ── Preset curriculum scenarios ──────────────────────────────────────────────
    st.divider()
    st.markdown("**Smart Curriculum Scenarios** *(auto-generated from your academic profile)*")
    st.markdown(
        "These scenarios are automatically selected based on your failed courses, "
        "high-impact prerequisites, and graduation pace."
    )

    if st.button("Generate Smart Scenarios", key="btn_preset_scenarios"):
        try:
            from what_if_analysis.what_if_engine import get_curriculum_preset_scenarios
            with st.spinner("Generating curriculum scenarios..."):
                preset = get_curriculum_preset_scenarios(
                    programme     = programme,
                    passed_codes  = passed_codes,
                    failed_codes  = failed_codes,
                    credits_passed= credits_passed,
                    semester      = semester,
                    predicted_gpa = predicted_gpa,
                )
            st.session_state["_preset_scenarios"] = preset
        except Exception as e_ps:
            st.error(f"Could not generate preset scenarios: {e_ps}")

    preset_results = st.session_state.get("_preset_scenarios", [])
    if preset_results:
        # Group by scenario group
        groups: dict[str, list[dict]] = {}
        for sc in preset_results:
            g = sc.get("group", "Other")
            groups.setdefault(g, []).append(sc)

        GROUP_COLORS = {
            "Retake Failed":    "#27AE60",
            "Risk: If You Fail": "#E74C3C",
            "High-Impact Pass": "#3498DB",
        }

        for g_name, g_items in groups.items():
            g_color = GROUP_COLORS.get(g_name, "#A78BFA")
            st.markdown(
                f'<div style="color:{g_color};font-weight:700;font-size:.9rem;'
                f'margin:12px 0 6px;">▸ {g_name}</div>',
                unsafe_allow_html=True
            )
            for sc in g_items:
                label    = sc.get("label", "Scenario")
                scenario = sc.get("scenario", {})
                unblk    = scenario.get("unblocked", [])
                blk      = scenario.get("new_blocked", [])
                msg      = scenario.get("curriculum_message", "")

                with st.expander(label):
                    st.markdown(f"""
                    <div class="xai-card" style="padding:10px 16px;">
                        <div style="color:#DDD6FE;font-size:.87rem;">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if unblk:
                        st.success(f"Unblocks: {', '.join(unblk)}")
                    if blk:
                        st.warning(f"Newly blocked: {', '.join(blk)}")


# ════════════════════════════════════════════════════════════════════════════════
#  Tab 5: Advanced Analytics (for professors / evaluators)
# ════════════════════════════════════════════════════════════════════════════════

def _render_advanced_tab():
    st.markdown('<p class="xai-section-title">Advanced Analytics (Researcher View)</p>',
                unsafe_allow_html=True)
    st.markdown(
        "This section exposes the full XAI toolkit: SHAP explanations, "
        "feature importance, PDP, and PCA visualisations.",
        unsafe_allow_html=True
    )

    result = st.session_state.get("xai_result")

    subtab_labels = ["SHAP Waterfall", "Curriculum XAI", "Global SHAP", "Feature Importance",
                     "PDP", "PCA Clusters", "Model Metrics"]
    subtabs = st.tabs(subtab_labels)

    with subtabs[0]:
        st.markdown("**SHAP Waterfall — Your Prediction Explained**")
        if result is None:
            st.info("Complete the Analysis Chat first.")
        else:
            buf = io.BytesIO()
            result.waterfall_fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            st.image(buf.getvalue(), use_container_width=True)
            st.markdown(
                "The waterfall chart shows how each feature **pushes the predicted GPA "
                "up (green) or down (red)** from the model baseline."
            )

    with subtabs[1]:
        st.markdown("**Curriculum XAI — Zewail Academic Context**")
        if result is None:
            st.info("Complete the Analysis Chat first.")
        else:
            # ── Curriculum SHAP-equivalent waterfall chart ────────────────────
            curr_shap_adv = st.session_state.get("xai_curriculum_shap", {})
            if curr_shap_adv:
                st.markdown("##### Curriculum Factor Impact (GPA contribution)")
                try:
                    from xai.explainability import ExplainabilityEngine
                    pred_gpa_adv = result.predicted_gpa if result else 2.0
                    wfall_fig = ExplainabilityEngine.curriculum_waterfall_fig(
                        curr_shap_adv, result.student_name if result else "Student", pred_gpa_adv
                    )
                    if wfall_fig:
                        buf_wf = io.BytesIO()
                        wfall_fig.savefig(buf_wf, format="png", dpi=110, bbox_inches="tight")
                        st.image(buf_wf.getvalue(), use_container_width=True)
                        import matplotlib.pyplot as plt
                        plt.close(wfall_fig)
                except Exception as e_wf:
                    # Fallback to Plotly chart if matplotlib waterfall fails
                    fig_curr_adv = _curriculum_impact_chart(curr_shap_adv)
                    if fig_curr_adv:
                        st.plotly_chart(fig_curr_adv, use_container_width=True, key="curr_impact_adv")
                st.caption(
                    "Green = curriculum factors boosting your academic trajectory. "
                    "Red = factors pulling your predicted GPA down. "
                    "Grounded in official Zewail City degree requirements and academic regulations."
                )
            else:
                st.info("Curriculum impact scores not available. Run analysis first.")

            st.divider()

            # ── Curriculum feature status table ───────────────────────────────
            curr_table_adv = st.session_state.get("xai_curriculum_table", [])
            if curr_table_adv:
                st.markdown("##### Curriculum Status Indicators")
                st.dataframe(
                    pd.DataFrame(curr_table_adv),
                    use_container_width=True,
                    hide_index=True,
                )
                st.divider()

            # ── Curriculum narratives ─────────────────────────────────────────
            narratives = getattr(result, "curriculum_narratives", [])
            if narratives:
                st.markdown("##### Detailed Curriculum Analysis")
                for n in narratives:
                    st.markdown(f"""
                    <div class="xai-card" style="padding:10px 16px;margin-bottom:8px;">
                        <span style="color:#DDD6FE;font-size:.9rem;">{n}</span>
                    </div>""", unsafe_allow_html=True)


    with subtabs[2]:
        st.markdown("**Global SHAP Summary — All Students**")
        if st.button("Generate Global SHAP Summary", key="btn_global_shap"):
            with st.spinner("Computing SHAP values over 500 students..."):
                eng = get_engine()
                fig = eng.global_summary_fig()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                st.image(buf.getvalue(), use_container_width=True)

    with subtabs[3]:
        st.markdown("**Feature Importance (Mean |SHAP|)**")
        if st.button("Show Feature Importance", key="btn_fi"):
            with st.spinner("Computing..."):
                eng = get_engine()
                fig = eng.feature_importance_fig()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                st.image(buf.getvalue(), use_container_width=True)

    with subtabs[4]:
        st.markdown("**Partial Dependence Plot**")
        feat_options = list(FEATURE_LABELS.keys())
        selected_feat = st.selectbox(
            "Select feature for PDP",
            feat_options,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key="pdp_select"
        )
        if st.button("Generate PDP", key="btn_pdp"):
            with st.spinner("Computing..."):
                eng = get_engine()
                fig = eng.pdp_fig(selected_feat)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                    st.image(buf.getvalue(), use_container_width=True)
                else:
                    st.warning("Could not generate PDP for this feature.")

    with subtabs[5]:
        st.markdown("**PCA Cluster Visualisation**")
        if st.button("Generate PCA Plot", key="btn_pca"):
            with st.spinner("Computing PCA over full dataset..."):
                eng = get_engine()
                fig = eng.pca_fig()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                st.image(buf.getvalue(), use_container_width=True)

    with subtabs[6]:
        st.markdown("**Model Evaluation Metrics**")
        metrics_path = _ROOT / "models" / "model_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### GPA Regression (XGBoost)")
                gpa_m = metrics["gpa_regression"]["xgboost"]
                st.metric("R²",   f"{gpa_m['r2']:.4f}")
                st.metric("MAE",  f"{gpa_m['mae']:.4f}")
                st.metric("RMSE", f"{gpa_m['rmse']:.4f}")
                cv = metrics.get("cv_gpa_r2", {})
                st.metric("5-Fold CV R²", f"{cv.get('mean',0):.4f} ± {cv.get('std',0):.4f}")

            with c2:
                st.markdown("##### Risk Classification (XGBoost)")
                risk_m = metrics["risk_classification"]["xgboost"]
                st.metric("Accuracy", f"{risk_m['accuracy']:.4f}")
                st.metric("F1 (weighted)", f"{risk_m['f1_weighted']:.4f}")
                st.metric("AUC", f"{risk_m.get('auc',0):.4f}")
                cv2 = metrics.get("cv_risk_f1", {})
                st.metric("5-Fold CV F1", f"{cv2.get('mean',0):.4f} ± {cv2.get('std',0):.4f}")
        else:
            st.warning("Model metrics not found. Run Phase 5 training first.")
