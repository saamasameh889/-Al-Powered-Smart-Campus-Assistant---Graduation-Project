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
from feature_engineering.feature_engineer import PROG_DIFFICULTY, PROG_CREDITS, PROG_CORE_CREDITS
from curriculum_intelligence.curriculum_engine import get_engine as get_curriculum_engine

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
        "xai_flow":          None,
        "xai_messages":      [],
        "xai_result":        None,
        "xai_recs":          None,
        "xai_analysed":      False,
        "xai_profile":       {},
        "xai_step":          0,
        "xai_curriculum":    {},     # curriculum feature dict
        "xai_graduation":    None,   # GraduationStatus object
        "xai_passed_codes":  [],     # matched official course codes passed
        "xai_failed_codes":  [],     # matched official course codes failed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.xai_flow is None:
        st.session_state.xai_flow = _new_flow()


# ── Component-type keyword map for feature extraction ─────────────────────────
_COMP_KEYWORDS = {
    "midterm":    ["midterm", "mid term", "mid-term", "mid exam", "midexam"],
    "final":      ["final", "final exam", "end term", "endterm", "end-term"],
    "assignment": ["assignment", "homework", "project", "coursework", "report", "task"],
    "quiz":       ["quiz", "quizzes", "test", "pop quiz"],
    "lab":        ["lab", "laboratory", "practical", "attendance", "lab attendance"],
}


def _profile_to_features(profile: dict) -> dict:
    """Convert collected chat profile to full feature vector."""
    att    = float(profile.get("avg_attendance",   78))
    asgn   = float(profile.get("avg_assignments",  72))
    quiz   = float(profile.get("avg_quizzes",      68))
    labs   = float(profile.get("avg_labs",         73))
    mid    = float(profile.get("avg_midterm",      70))
    fin    = float(profile.get("avg_final",        70))
    failed = float(profile.get("failed_courses",    0))
    cr     = float(profile.get("credits_registered",45))
    sem    = float(profile.get("semester",          3))
    prog   = str(profile.get("programme", "CSAI"))

    overall = 0.10*quiz + 0.15*asgn + 0.15*labs + 0.30*mid + 0.30*fin
    overall = float(np.clip(overall, 0, 100))

    total_courses = max(cr / 3, 1)
    cp = cr - failed * 3

    # Attendance risk score
    threshold = 75.0
    att_risk = ((threshold - att) / threshold) ** 1.5 if att < threshold else 0.0

    # Performance trend
    trend = fin - mid

    # Credit completion ratio
    ccr = cp / max(cr, 1)

    # Academic consistency
    scores = [asgn, quiz, labs, mid, fin]
    row_std = float(np.std(scores))
    consist = float(np.clip(1 - row_std / 30, 0, 1))

    # Study efficiency proxy
    eff = (overall / 100) / (0.35 + 0.1)

    # Semester momentum (no raw data → use trend as proxy)
    momentum = trend * 0.5

    prog_enc   = PROG_DIFFICULTY.get(prog, 0.55)
    school_enc = {"CS&AI": 2, "ENGR": 1, "SCI": 3, "BUS": 0}.get(SCHOOLS.get(prog, "CS&AI"), 2)

    # ── Curriculum-aware features (approximate — overridden when course codes known) ──
    total_req  = float(PROG_CREDITS.get(prog, 132))
    core_req   = float(PROG_CORE_CREDITS.get(prog, 86))
    cp_safe    = max(cp, 0)
    grad_prog  = min(cp_safe / total_req, 1.0)
    exp_prog   = min(sem / 8.0, 1.0)
    delay_sem  = round((exp_prog - grad_prog) * 8, 2)
    core_comp  = min(cp_safe / core_req, 1.0)
    fail_r     = failed / max(total_courses, 1)
    prereq_prx = float(np.clip(1.0 - fail_r, 0, 1))
    blocked_pr = float(np.clip(failed * 6 / total_req, 0, 0.5))
    align_prx  = float(np.clip(ccr * prereq_prx, 0, 1))
    readiness  = float(np.clip(
        0.40 * grad_prog + 0.25 * core_comp + 0.20 * prereq_prx + 0.15 * (1 - min(delay_sem / 8, 1)),
        0, 1,
    ))

    return {
        "avg_attendance":         att,
        "avg_assignments":        asgn,
        "avg_quizzes":            quiz,
        "avg_labs":               labs,
        "avg_midterm":            mid,
        "avg_final":              fin,
        "avg_overall":            round(overall, 1),
        "semester":               sem,
        "credits_registered":     cr,
        "credits_passed":         max(cp, 0),
        "failed_courses":         failed,
        "attendance_risk_score":  round(att_risk, 4),
        "course_difficulty_index": prog_enc,
        "failed_course_ratio":    round(fail_r, 4),
        "performance_trend":      round(trend, 2),
        "credit_completion_ratio": round(ccr, 4),
        "academic_consistency":   round(consist, 3),
        "study_efficiency":       round(eff, 3),
        "semester_momentum":      round(momentum, 2),
        "programme_encoded":      prog_enc,
        "school_encoded":         float(school_enc),
        "study_hours":            20.0,
        # Curriculum features (available for display/XAI; not consumed by current models)
        "graduation_progress_ratio":   round(grad_prog, 4),
        "expected_progress_ratio":     round(exp_prog,  4),
        "graduation_delay_semesters":  delay_sem,
        "core_completion_ratio":       round(core_comp, 4),
        "prereq_completion_proxy":     round(prereq_prx, 4),
        "blocked_progress_ratio":      round(blocked_pr, 4),
        "curriculum_alignment_proxy":  round(align_prx, 4),
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
    """Simple bar comparing assessment component averages."""
    labels = ["Attendance", "Assignments", "Quizzes", "Labs", "Midterm", "Final"]
    keys   = ["avg_attendance", "avg_assignments", "avg_quizzes",
              "avg_labs", "avg_midterm", "avg_final"]
    vals   = [features.get(k, 70) for k in keys]
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
        return f"**How many credit hours is {c_name} worth?**"
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
        msgs.append(("user", name))
        flow["phase"] = "course_credits"

    elif phase == "course_credits":
        wip["course_credits"] = int(ans)
        msgs.append(("user", f"{ans} credit hours"))
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
        rows.append({
            "Course":          c["name"],
            "Credits":         c["credits"],
            "Score (taken)":   f"{t_s:.0f} / {t_m:.0f}",
            "Current %":       f"{pct:.0f}%",
            "Remaining Marks": f"{rem:.0f}",
            "Components":      len(c["components"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chat_tab():
    st.markdown('<p class="xai-section-title">AI Academic Analysis Chat</p>',
                unsafe_allow_html=True)

    if st.button("↺ Start Over", use_container_width=False):
        for k in [k for k in st.session_state if k.startswith("xai_")]:
            del st.session_state[k]
        st.rerun()

    st.markdown("---")

    flow = st.session_state.xai_flow
    msgs = st.session_state.xai_messages
    step = st.session_state.xai_step

    # ── Conversation history (bold markdown converted to <strong>) ────────────
    for role, msg in msgs:
        tag = "xai-chat-ai" if role == "ai" else "xai-chat-msg"
        ico = "🤖" if role == "ai" else "👤"
        st.markdown(
            f'<div class="{tag}">{ico} {_md(msg)}</div>',
            unsafe_allow_html=True,
        )

    # ── Done state ───────────────────────────────────────────────────────────
    if flow["phase"] == "done":
        if not st.session_state.xai_analysed:
            _show_course_summary(flow)
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                _run_analysis()
                st.rerun()
        else:
            st.success(
                "Analysis complete! Switch to the **Academic Dashboard** tab to see your results."
            )
        return

    # ── Current question rendered with st.markdown (supports bold/italic) ────
    q_text = _chat_question(flow)
    wtype, wkwargs = _widget_type(flow)

    with st.container():
        st.markdown(
            f'<div class="xai-chat-ai">🤖</div>',
            unsafe_allow_html=True,
        )
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

def _render_dashboard_tab():
    st.markdown('<p class="xai-section-title">Academic Health Dashboard</p>', unsafe_allow_html=True)

    result = st.session_state.get("xai_result")
    if result is None:
        st.info("Complete the AI Analysis Chat first to see your dashboard.")
        return

    profile  = st.session_state.xai_profile
    features = _profile_to_features(profile)

    # ── Row 1: Key metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fig = _gauge_chart(result.predicted_gpa, 4.0, "Predicted GPA",
                           "#27AE60" if result.predicted_gpa >= 2.5 else
                           ("#F39C12" if result.predicted_gpa >= 2.0 else "#E74C3C"))
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

    # ── Row 2: Radar + Bar chart ─────────────────────────────────────────────────
    col_radar, col_bar = st.columns(2)

    with col_radar:
        st.markdown("**Performance Profile**")
        cats = ["Attendance", "Assignments", "Quizzes", "Labs", "Midterm", "Final"]
        vals = [
            features["avg_attendance"],
            features["avg_assignments"],
            features["avg_quizzes"],
            features["avg_labs"],
            features["avg_midterm"],
            features["avg_final"],
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

    # ── Curriculum Intelligence Panel ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="xai-section-title">Curriculum Progress Analysis</p>',
                unsafe_allow_html=True)

    grad_status  = st.session_state.get("xai_graduation")
    curr_feat    = st.session_state.get("xai_curriculum", {})
    passed_codes = st.session_state.get("xai_passed_codes", [])
    failed_codes = st.session_state.get("xai_failed_codes", [])

    if grad_status:
        # Graduation progress bar
        st.markdown(f"**Graduation Progress — {grad_status.programme}**")
        st.progress(
            min(int(grad_status.progress_ratio * 100), 100),
            text=(f"{grad_status.credits_passed} / {grad_status.total_required} credits "
                  f"({grad_status.progress_ratio*100:.0f}%)  —  "
                  f"{grad_status.credits_remaining} credits remaining"),
        )

        # Curriculum metric chips
        cc1, cc2, cc3, cc4 = st.columns(4)
        delay = curr_feat.get("graduation_delay_semesters", 0)
        cc1.metric("Graduation Pace",
                   "On Track" if delay <= 0.5 else f"{delay:.1f} sem behind",
                   delta=None)
        cc2.metric("Core Courses",
                   f"{curr_feat.get('core_course_completion_ratio', 0)*100:.0f}%",
                   delta=None)
        cc3.metric("Prereq Completion",
                   f"{curr_feat.get('prerequisite_completion_ratio', 1)*100:.0f}%",
                   delta=None)
        blocked_cr = curr_feat.get("blocked_credit_hours", 0)
        cc4.metric("Blocked Credits",
                   f"{blocked_cr} cr" if blocked_cr > 0 else "None",
                   delta=None)

        # Graduation status message
        status_color = "#27AE60" if grad_status.on_track else "#F39C12"
        if not grad_status.graduation_gpa_ok:
            status_color = "#E74C3C"
        st.markdown(f"""
        <div class="xai-card" style="border-color:{status_color}44;margin-top:12px;">
            <p style="color:{status_color};font-weight:600;margin:0;font-size:.92rem;">
                {grad_status.message}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Curriculum impact chart ───────────────────────────────────────────
        curr_shap = st.session_state.get("xai_curriculum_shap", {})
        if curr_shap:
            st.markdown("**Curriculum Factor Impact** *(estimated GPA contribution per factor)*")
            fig_curr = _curriculum_impact_chart(curr_shap)
            if fig_curr:
                st.plotly_chart(fig_curr, use_container_width=True, key="curr_impact_dash")
            st.caption(
                "Green bars = factors supporting your academic trajectory. "
                "Red bars = curriculum factors reducing predicted academic outcome. "
                "Grounded in official Zewail degree requirements and regulations."
            )

        # ── Curriculum feature status table ──────────────────────────────────
        curr_table = st.session_state.get("xai_curriculum_table", [])
        if curr_table:
            st.markdown("**Curriculum Intelligence Status**")
            import pandas as pd
            st.dataframe(
                pd.DataFrame(curr_table),
                use_container_width=True,
                hide_index=True,
            )

        # Blocked prerequisite chains
        blocked_entries = get_curriculum_engine().get_blocked_courses(failed_codes, profile.get("programme","CSAI"))
        if blocked_entries:
            st.markdown("**Blocked Academic Paths (due to failed prerequisites)**")
            for entry in blocked_entries[:3]:
                deps_str = ", ".join(entry["direct_blocked"][:4])
                st.markdown(f"""
                <div class="xai-card" style="border-color:rgba(231,76,60,.3);padding:12px 16px;">
                    <span style="color:#E74C3C;font-weight:700;">{entry['failed_course']}</span>
                    <span style="color:#DDD6FE;font-size:.85rem;"> — {entry['failed_title']}</span><br>
                    <span style="color:#A78BFA;font-size:.82rem;">
                        Blocks: {deps_str or "no direct dependents found"}
                        &nbsp;|&nbsp; {entry['blocked_credit_hours']} blocked credits
                    </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Fallback: simple progress bar using feature_engineer data
        from feature_engineering.feature_engineer import PROG_CREDITS
        prog = profile.get("programme", "CSAI")
        total_creds    = PROG_CREDITS.get(prog, 132)
        cr_passed_val  = features["credits_passed"]
        grad_readiness = (cr_passed_val / total_creds) * 100
        st.markdown("**Graduation Progress**")
        st.progress(min(int(grad_readiness), 100),
                    text=f"{cr_passed_val:.0f} / {total_creds} credits passed ({grad_readiness:.1f}%)")


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

    CURRICULUM_CATEGORIES = {"Curriculum", "Prerequisites", "Graduation", "Degree Progress"}

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

    curriculum_recs = [r for r in recs if r.category in CURRICULUM_CATEGORIES]
    academic_recs   = [r for r in recs if r.category not in CURRICULUM_CATEGORIES]

    if curriculum_recs:
        st.markdown("""
        <div style="background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.3);
                    border-radius:10px;padding:10px 16px;margin-bottom:16px;">
            <span style="color:#C4B5FD;font-weight:700;font-size:.95rem;">
                📋 Curriculum & Degree Progress Advisories
            </span>
            <span style="color:#A78BFA;font-size:.80rem;margin-left:8px;">
                Based on official Zewail City programme requirements
            </span>
        </div>
        """, unsafe_allow_html=True)
        for r in curriculum_recs:
            _rec_card(r)
        if academic_recs:
            st.divider()

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

    if not curriculum_recs and not academic_recs:
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

    # ── Curriculum What-If Simulator ────────────────────────────────────────────
    st.divider()
    st.markdown("**Curriculum What-If — Course Scenario Simulator**")
    st.markdown(
        "Simulate how passing, failing, retaking, or postponing a course affects "
        "your graduation timeline and which future courses become blocked or unblocked."
    )

    passed_codes = st.session_state.get("xai_passed_codes", [])
    failed_codes = st.session_state.get("xai_failed_codes", [])
    graduation   = st.session_state.get("xai_graduation")
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
                delay = curr_wi.get("new_graduation_delay", 0)
                delay_color = "#E74C3C" if delay > 0 else "#27AE60"
                delay_label = (
                    f"+{delay} semester(s) delay" if delay > 0
                    else "No graduation delay" if delay == 0
                    else f"{delay} semester(s) ahead"
                )
                grad_ok = curr_wi.get("new_graduation_ok", True)
                unblocked = curr_wi.get("unblocked", [])
                new_blocked = curr_wi.get("new_blocked", [])

                st.markdown(f"""
                <div class="xai-card" style="padding:14px 18px;">
                    <div style="font-weight:700;color:#C4B5FD;font-size:.95rem;margin-bottom:8px;">
                        {curr_wi.get('scenario_name','Scenario')}
                    </div>
                    <div style="display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap;">
                        <div style="text-align:center;">
                            <div style="font-size:1.4rem;font-weight:700;color:{delay_color};">{delay_label}</div>
                            <div style="color:#A78BFA;font-size:.78rem;">Graduation Impact</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.4rem;font-weight:700;color:{'#27AE60' if grad_ok else '#E74C3C'};">
                                {'On Track' if grad_ok else 'At Risk'}
                            </div>
                            <div style="color:#A78BFA;font-size:.78rem;">Graduation Status</div>
                        </div>
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
                delay    = scenario.get("new_graduation_delay", 0)
                grad_ok  = scenario.get("new_graduation_ok", True)
                unblk    = scenario.get("unblocked", [])
                blk      = scenario.get("new_blocked", [])
                msg      = scenario.get("curriculum_message", "")

                delay_txt = (
                    f"+{delay}sem delay" if delay > 0
                    else "No delay" if delay == 0
                    else f"{abs(delay)}sem ahead"
                )
                status_color = "#27AE60" if grad_ok else "#E74C3C"

                with st.expander(f"{label}  |  {delay_txt}  |  {'✅ On Track' if grad_ok else '⚠️ At Risk'}"):
                    st.markdown(f"""
                    <div class="xai-card" style="padding:10px 16px;">
                        <div style="color:#DDD6FE;font-size:.87rem;">{msg}</div>
                        <div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;">
                            <span style="color:{status_color};font-size:.82rem;">
                                Graduation: {'On Track' if grad_ok else 'At Risk'}
                            </span>
                            <span style="color:#A78BFA;font-size:.82rem;">
                                Delay: {delay_txt}
                            </span>
                        </div>
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
            failed_codes = st.session_state.get("xai_failed_codes", [])
            profile_adv  = st.session_state.get("xai_profile", {})
            prog         = profile_adv.get("programme", "CSAI")

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

            # ── Blocked courses detail table ──────────────────────────────────
            if failed_codes:
                blocked = get_curriculum_engine().get_blocked_courses(failed_codes, prog)
                if blocked:
                    st.divider()
                    st.markdown("##### Prerequisite Chain Impact Table")
                    rows = []
                    for b in blocked:
                        rows.append({
                            "Failed Course":  b["failed_course"],
                            "Title":          b["failed_title"],
                            "Direct Blocks":  len(b["direct_blocked"]),
                            "Total Blocked":  len(b["all_blocked"]),
                            "Blocked Credits": b["blocked_credit_hours"],
                            "Chain Depth":    b["chain_depth"],
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption(
                        "Chain Depth = how many prerequisite levels deep the blockage propagates. "
                        "High depth means retaking this course unlocks a long chain of future courses."
                    )

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
