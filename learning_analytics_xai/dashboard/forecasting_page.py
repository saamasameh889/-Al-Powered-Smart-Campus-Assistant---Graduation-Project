"""
forecasting_page.py — GPA Trajectory Forecasting UI  (Product C)
═══════════════════════════════════════════════════════════════════════════════
Streamlit page that:
  • Trains (or loads) a GPAForecaster LSTM on demand with a progress bar
  • Renders a GPA trajectory chart: historical semesters + 3-step forecast
    with quantile confidence band
  • Provides what-if sliders (attendance, course load) for scenario planning
  • Shows the student's archetype from Product D (if clustering has been run)

Entry point:
    from forecasting_page import render_forecasting_page
    render_forecasting_page()
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_ROOT   = _HERE.parent
_FCAST  = _ROOT / "forecasting"
_MODEL_DIR  = _ROOT / "models"
_MODEL_PATH = _MODEL_DIR / "gpa_forecaster.pt"

for p in [str(_ROOT), str(_FCAST)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sequence_generator import (
    generate_sequences,
    _PROG_ENC,
    _SCHOOL_ENC,
    _PROG_DIFFICULTY,
    _SEM_LOADS,
    _MAX_CREDITS,
    STATIC_DIM,
    TEMPORAL_DIM,
)
from gpa_forecaster import GPAForecaster

# ── Programme metadata ─────────────────────────────────────────────────────────
_PROGRAMMES = [
    "CSAI", "DSAI", "SWE",
    "MECH", "EEE",  "CIV",
    "MATH", "PHYS", "CHEM",
    "BUS",  "FIN",
]
_SCHOOL_MAP = {
    "CSAI": "CS&AI", "DSAI": "CS&AI", "SWE": "CS&AI",
    "MECH": "ENGR",  "EEE":  "ENGR",  "CIV": "ENGR",
    "MATH": "SCI",   "PHYS": "SCI",   "CHEM":"SCI",
    "BUS":  "BUS",   "FIN":  "BUS",
}
HISTORY_LEN = 4
HORIZON     = 3


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading / training
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_forecaster_cached() -> GPAForecaster | None:
    if _MODEL_PATH.exists():
        try:
            return GPAForecaster.load(_MODEL_PATH)
        except Exception:
            return None
    return None


def _train_forecaster(progress_bar, status_text) -> GPAForecaster:
    status_text.caption("Generating training sequences…")
    data = generate_sequences(
        n_augmentations=5,
        history_len=HISTORY_LEN,
        horizon=HORIZON,
        seed=42,
    )
    n_seq = len(data["static"])
    status_text.caption(f"Generated {n_seq:,} training sequences — starting LSTM training…")

    forecaster = GPAForecaster(
        static_dim=STATIC_DIM,
        temporal_dim=TEMPORAL_DIM,
        lstm_hidden=128,
        lstm_layers=2,
        horizon=HORIZON,
        dropout=0.25,
    )

    _epoch_state = {"epoch": 0, "total": 1}

    def _cb(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
        _epoch_state["epoch"] = epoch
        _epoch_state["total"] = total
        progress_bar.progress(epoch / total)
        status_text.caption(
            f"Epoch {epoch}/{total}  —  train loss {train_loss:.4f}  "
            f"val loss {val_loss:.4f}"
        )

    forecaster.fit(
        data,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        val_frac=0.20,
        patience=12,
        progress_cb=_cb,
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    forecaster.save(_MODEL_PATH)
    progress_bar.progress(1.0)
    status_text.empty()
    return forecaster


# ══════════════════════════════════════════════════════════════════════════════
#  Input → static / temporal features
# ══════════════════════════════════════════════════════════════════════════════

def _build_static(
    programme: str,
    avg_attendance: float,
    avg_final: float,
    failed_ratio: float,
    total_credits_completed: float = 80.0,
) -> np.ndarray:
    school = _SCHOOL_MAP.get(programme, "CS&AI")
    return np.array([
        _PROG_ENC.get(programme, 0.0),
        _SCHOOL_ENC.get(school, 0.0),
        avg_attendance / 100.0,
        avg_final      / 100.0,
        failed_ratio,
        _PROG_DIFFICULTY.get(programme, 0.5),
        total_credits_completed / _MAX_CREDITS,  # GPA inertia: higher = harder to move GPA
    ], dtype=np.float32)


def _build_temporal(
    gpa_history: list[float],
    load_history: list[float],
    credits_before_window: float = 0.0,
) -> np.ndarray:
    temporal = []
    running_credits = credits_before_window
    for i, (gpa, load) in enumerate(zip(gpa_history, load_history)):
        running_credits += load
        gpa_norm  = gpa / 4.0
        load_norm = load / 24.0
        risk_flag = (
            1.0 if gpa < 2.0
            else 0.5 if gpa < 2.5
            else 0.25 if gpa < 3.0
            else 0.0
        )
        gpa_delta = (gpa_history[i] - gpa_history[i - 1]) / 4.0 if i > 0 else 0.0
        cumulative_credits_norm = running_credits / _MAX_CREDITS
        temporal.append([gpa_norm, load_norm, risk_flag, gpa_delta, cumulative_credits_norm])
    return np.array(temporal, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Trajectory chart
# ══════════════════════════════════════════════════════════════════════════════

def _render_trajectory(
    gpa_history:  list[float],
    forecast:     dict,
    start_sem:    int,
    what_if_res:  dict | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed.")
        return

    hist_sems = list(range(start_sem, start_sem + len(gpa_history)))
    fore_sems = list(range(start_sem + len(gpa_history),
                           start_sem + len(gpa_history) + HORIZON))

    fig = go.Figure()

    # ── Historical semesters ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=hist_sems, y=gpa_history,
        mode="lines+markers",
        name="Historical GPA",
        line=dict(color="#8B5CF6", width=2.5),
        marker=dict(size=8, color="#C4B5FD"),
    ))

    # Bridge line (last historical → first forecast)
    bridge_x = [hist_sems[-1], fore_sems[0]]
    bridge_y = [gpa_history[-1], forecast["gpa_median"][0]]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines",
        line=dict(color="#8B5CF6", width=1.5, dash="dot"),
        showlegend=False,
    ))

    # ── Confidence band (q10–q90) ──────────────────────────────────────────────
    q10 = list(forecast["gpa_q10"])
    q90 = list(forecast["gpa_q90"])
    fig.add_trace(go.Scatter(
        x=fore_sems + fore_sems[::-1],
        y=q90 + q10[::-1],
        fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name="80% CI (q10–q90)",
    ))

    # ── Median forecast ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fore_sems, y=list(forecast["gpa_median"]),
        mode="lines+markers",
        name="Forecast (median)",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=10, color="#10B981", symbol="diamond"),
    ))

    # ── What-if overlay ────────────────────────────────────────────────────────
    if what_if_res is not None:
        fig.add_trace(go.Scatter(
            x=fore_sems, y=list(what_if_res["gpa_median"]),
            mode="lines+markers",
            name="What-if (median)",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            marker=dict(size=8, color="#F59E0B", symbol="circle"),
        ))

    # ── GPA risk zones ────────────────────────────────────────────────────────
    all_sems = hist_sems + fore_sems
    x_min, x_max = min(all_sems) - 0.3, max(all_sems) + 0.3
    for thresh, label, color in [
        (2.0, "Probation (<2.0)", "rgba(239,68,68,0.06)"),
        (2.5, "At-risk (<2.5)",   "rgba(245,158,11,0.06)"),
    ]:
        fig.add_hrect(
            y0=0, y1=thresh,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="left",
            annotation_font=dict(size=9, color="#7C6FAD"),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color="#C4B5FD", size=12),
        xaxis=dict(
            title="Semester",
            dtick=1,
            showgrid=True,
            gridcolor="rgba(139,92,246,.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title="GPA (0–4.0)",
            range=[0, 4.2],
            showgrid=True,
            gridcolor="rgba(139,92,246,.08)",
            zeroline=False,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#9D77F5")),
        margin=dict(l=0, r=0, t=14, b=0),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Training loss chart
# ══════════════════════════════════════════════════════════════════════════════

def _render_loss_chart(forecaster: GPAForecaster) -> None:
    if not forecaster.train_losses:
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    epochs = list(range(1, len(forecaster.train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=forecaster.train_losses,
        mode="lines", name="Train", line=dict(color="#8B5CF6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=forecaster.val_losses,
        mode="lines", name="Validation", line=dict(color="#10B981", width=2),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color="#C4B5FD", size=11),
        xaxis=dict(title="Epoch", showgrid=False),
        yaxis=dict(title="Pinball Loss", showgrid=True, gridcolor="rgba(139,92,246,.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ══════════════════════════════════════════════════════════════════════════════

def render_forecasting_page() -> None:
    """Entry point called from phase7_streamlit_app.py."""
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:800;color:#C4B5FD;"
        "margin-bottom:4px'>📈 GPA Forecasting & Study Tools</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:.83rem;color:#4A3A7A;margin-bottom:16px'>"
        "LSTM trajectory forecasting · GPA/grade calculators · attendance tracker · priority matrix</div>",
        unsafe_allow_html=True,
    )

    # ── Import study tools (lazy, avoids circular at module load time) ─────────
    _tools_err = ""
    try:
        from study_tools_page import render_study_tools as _render_study_tools
        _tools_ok = True
    except Exception as _te:
        _tools_ok = False
        _tools_err = str(_te)

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    tab_forecast, tab_tools = st.tabs(["📈  GPA Forecast", "🛠️  Study Tools"])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — GPA Forecast (LSTM)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_forecast:
        # ── Load or offer to train ─────────────────────────────────────────────
        forecaster: GPAForecaster | None = st.session_state.get("_forecaster_obj")
        if forecaster is None:
            forecaster = _load_forecaster_cached()
            if forecaster is not None:
                st.session_state["_forecaster_obj"] = forecaster

        if forecaster is None:
            st.info("Forecast model not found. Please ensure the model file exists.")

        if forecaster is not None:
            # ── Archetype context (from Product D, if available) ──────────────
            cluster_result = st.session_state.get("cluster_result")
            if cluster_result:
                arch  = cluster_result.get("archetype", {})
                color = arch.get("color", "#8B5CF6")
                st.markdown(
                    f"<div style='border-left:3px solid {color};padding:8px 16px;"
                    f"background:rgba(255,255,255,.02);border-radius:0 10px 10px 0;"
                    f"margin-bottom:18px'>"
                    f"<span style='font-size:.7rem;color:#5B4D8A;text-transform:uppercase;"
                    f"letter-spacing:.08em'>Archetype (from clustering)</span><br>"
                    f"<span style='font-size:.95rem;font-weight:700;color:{color}'>"
                    f"{arch.get('name','—')}</span>"
                    f"<span style='font-size:.75rem;color:#7C6FAD;margin-left:8px'>"
                    f"{arch.get('description','')}</span></div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Student profile input ──────────────────────────────────────────
            st.markdown(
                "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
                "text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px'>"
                "📋 &nbsp;Enter Your Profile</div>",
                unsafe_allow_html=True,
            )

            _cf = st.session_state.get("cluster_features", {})

            col_a, col_b = st.columns([1, 2])
            with col_a:
                programme      = st.selectbox("Programme", _PROGRAMMES, index=0, key="fc_programme")
                current_sem    = st.slider("Current Semester", 4, 8, 5, key="fc_sem")
                avg_attendance = st.slider(
                    "Avg Attendance (%)", 50, 100,
                    int(_cf.get("avg_attendance", 82)), key="fc_att",
                )
                avg_final = st.slider(
                    "Avg Final Score (%)", 30, 100,
                    int(_cf.get("avg_final", 70)), key="fc_final",
                )
                failed_ratio = st.slider(
                    "Failed Course Ratio", 0.0, 1.0,
                    float(_cf.get("failed_ratio", 0.05)), step=0.01, key="fc_fail",
                )

            with col_b:
                st.markdown(
                    "<div style='font-size:.72rem;color:#5B4D8A;margin-bottom:8px'>"
                    "Enter your GPA for the last 4 semesters:</div>",
                    unsafe_allow_html=True,
                )
                gpa_cols = st.columns(HISTORY_LEN)
                gpa_history = []
                load_history = []
                for i, gc in enumerate(gpa_cols):
                    sem_label = f"Sem {current_sem - HISTORY_LEN + i + 1}"
                    g = gc.number_input(
                        sem_label, min_value=0.5, max_value=4.0,
                        value=round(float(_cf.get("cumulative_gpa", 2.8)) + 0.05 * (i - 1), 2),
                        step=0.05,
                        key=f"gpa_h_{i}",
                    )
                    gpa_history.append(float(g))

                st.markdown(
                    "<div style='font-size:.72rem;color:#5B4D8A;margin:10px 0 8px'>"
                    "Credit load per semester:</div>",
                    unsafe_allow_html=True,
                )
                load_cols = st.columns(HISTORY_LEN)
                for i, lc in enumerate(load_cols):
                    sem_label = f"Load {current_sem - HISTORY_LEN + i + 1}"
                    lo = lc.number_input(
                        sem_label, min_value=9, max_value=24,
                        value=18, step=1,
                        key=f"load_h_{i}",
                    )
                    load_history.append(float(lo))

            # ── Forecast ──────────────────────────────────────────────────────
            if st.button("📈  Forecast My GPA", type="primary"):
                prior_sems = max(0, current_sem - HISTORY_LEN)
                credits_before_window = float(sum(_SEM_LOADS[:prior_sems]))
                total_credits_completed = credits_before_window + sum(load_history)

                static_x   = _build_static(programme, avg_attendance, avg_final, failed_ratio,
                                            total_credits_completed)
                temporal_x = _build_temporal(gpa_history, load_history, credits_before_window)

                with st.spinner("Running LSTM inference…"):
                    forecast = forecaster.predict(static_x, temporal_x)

                st.session_state["_last_forecast"]   = forecast
                st.session_state["_last_static_x"]   = static_x
                st.session_state["_last_temporal_x"]  = temporal_x
                st.session_state["_last_start_sem"]   = current_sem - HISTORY_LEN + 1
                st.session_state["_last_gpa_hist"]    = gpa_history

            forecast = st.session_state.get("_last_forecast")
            if forecast is not None:
                # ── Results ────────────────────────────────────────────────────
                st.divider()
                st.markdown(
                    "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
                    "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
                    "📊 &nbsp;Trajectory Forecast</div>",
                    unsafe_allow_html=True,
                )

                sem_start = st.session_state.get("_last_start_sem", 1) + HISTORY_LEN
                m_cols = st.columns(HORIZON)
                for i, mc in enumerate(m_cols):
                    gpa_m  = float(forecast["gpa_median"][i])
                    gpa_lo = float(forecast["gpa_q10"][i])
                    gpa_hi = float(forecast["gpa_q90"][i])
                    color  = (
                        "#10B981" if gpa_m >= 3.0
                        else "#F59E0B" if gpa_m >= 2.0
                        else "#EF4444"
                    )
                    mc.markdown(
                        f"<div style='border:1px solid {color}55;border-radius:14px;"
                        f"padding:14px;text-align:center'>"
                        f"<div style='font-size:.65rem;text-transform:uppercase;"
                        f"letter-spacing:.08em;color:#5B4D8A'>Semester {sem_start + i}</div>"
                        f"<div style='font-size:1.7rem;font-weight:800;color:{color}'>"
                        f"{gpa_m:.2f}</div>"
                        f"<div style='font-size:.7rem;color:#7C6FAD'>"
                        f"[{gpa_lo:.2f} – {gpa_hi:.2f}]</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── What-if sliders ────────────────────────────────────────────
                with st.expander("🎛️  What-If Scenario Analysis", expanded=True):
                    st.caption(
                        "Adjust your plan below and see how your forecast changes "
                        "compared to the baseline."
                    )
                    wc1, wc2 = st.columns(2)
                    wi_attendance = wc1.slider(
                        "New attendance (%)", 50, 100,
                        int(avg_attendance), key="fc_wi_att",
                    )
                    wi_load = wc2.slider(
                        "New credit load", 9, 24, 18, key="fc_wi_load",
                    )
                    run_wi = st.button("▶  Compute What-If", key="fc_wi_btn")

                    what_if_res = st.session_state.get("_what_if_res")
                    if run_wi:
                        static_x   = st.session_state.get("_last_static_x")
                        temporal_x = st.session_state.get("_last_temporal_x")
                        if static_x is not None and temporal_x is not None:
                            what_if_res = forecaster.predict_what_if(
                                static_x, temporal_x,
                                attendance_pct=wi_attendance,
                                load_credits=wi_load,
                            )
                            st.session_state["_what_if_res"] = what_if_res

                    if what_if_res:
                        wm_cols = st.columns(HORIZON)
                        for i, wmc in enumerate(wm_cols):
                            base_m = float(forecast["gpa_median"][i])
                            wi_m   = float(what_if_res["gpa_median"][i])
                            delta  = wi_m - base_m
                            arrow  = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "→")
                            dcolor = "#10B981" if delta > 0 else ("#EF4444" if delta < 0 else "#9D77F5")
                            wmc.markdown(
                                f"<div style='text-align:center;padding:10px'>"
                                f"<div style='font-size:.65rem;color:#5B4D8A'>"
                                f"Sem {sem_start + i}</div>"
                                f"<div style='font-size:1.3rem;font-weight:700;color:{dcolor}'>"
                                f"{arrow} {abs(delta):+.2f}</div>"
                                f"<div style='font-size:.72rem;color:#7C6FAD'>"
                                f"{wi_m:.2f} vs {base_m:.2f}</div></div>",
                                unsafe_allow_html=True,
                            )

                # ── Main trajectory plot ──────────────────────────────────────
                _render_trajectory(
                    gpa_history = st.session_state.get("_last_gpa_hist", []),
                    forecast    = forecast,
                    start_sem   = st.session_state.get("_last_start_sem", 1),
                    what_if_res = st.session_state.get("_what_if_res"),
                )

                # ── Training loss (expandable) ────────────────────────────────
                with st.expander("📉  Training History (LSTM loss curve)"):
                    _render_loss_chart(forecaster)
                    if forecaster.train_losses:
                        st.caption(
                            f"Trained for {len(forecaster.train_losses)} epochs.  "
                            f"Best val loss: {min(forecaster.val_losses):.4f}  "
                            f"(lower is better).  "
                            f"Loss function: pinball (quantile regression) for q=[0.10, 0.50, 0.90]."
                        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Study Tools (6 calculators)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_tools:
        if not _tools_ok:
            st.error(f"Study tools failed to load: {_tools_err}")
        else:
            _render_study_tools()
