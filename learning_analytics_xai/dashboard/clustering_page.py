"""
clustering_page.py — Student Archetype Clustering UI  (Product D)
═══════════════════════════════════════════════════════════════════════════════
Streamlit page that:
  • Trains (or loads) a StudentClusterer on demand
  • Shows a 2D UMAP scatter plot coloured by archetype
  • Renders cluster profile cards with feature summaries
  • Provides a "What's My Archetype?" form for live student input
  • Exposes the fitted clusterer for use by other pages (e.g. forecasting)

Entry point:
    from clustering_page import render_clustering_page
    render_clustering_page()
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).parent
_ROOT  = _HERE.parent
_CLUST = _ROOT / "clustering"
_MODEL_DIR = _ROOT / "models"
_MODEL_PATH = _MODEL_DIR / "student_clustering.pkl"

for p in [str(_ROOT), str(_CLUST)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from student_clustering import StudentClusterer, CLUSTER_FEATURES, load_summary_df


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_clusterer_cached() -> StudentClusterer | None:
    """Load a fitted clusterer from disk (None if not yet trained)."""
    if _MODEL_PATH.exists():
        try:
            return StudentClusterer.load(_MODEL_PATH)
        except Exception:
            return None
    return None


def _card(title: str, value: str, color: str = "#8B5CF6") -> str:
    return (
        f"<div style='background:rgba(255,255,255,.03);border:1px solid "
        f"rgba(139,92,246,.18);border-radius:14px;padding:16px 18px;"
        f"margin-bottom:10px'>"
        f"<div style='font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;"
        f"color:#5B4D8A;margin-bottom:4px'>{title}</div>"
        f"<div style='font-size:1.15rem;font-weight:700;color:{color}'>{value}</div>"
        f"</div>"
    )


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
        f"border-radius:20px;padding:3px 11px;font-size:.72rem;font-weight:600'>"
        f"{text}</span>"
    )


def _feature_bar(label: str, value: float, max_val: float, color: str) -> str:
    pct = min(100, max(0, value / max_val * 100))
    return (
        f"<div style='margin-bottom:6px'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:.72rem;color:#7C6FAD;margin-bottom:2px'>"
        f"<span>{label}</span><span>{value:.1f}</span></div>"
        f"<div style='background:rgba(255,255,255,.06);border-radius:4px;height:5px'>"
        f"<div style='width:{pct:.0f}%;background:{color};border-radius:4px;"
        f"height:5px'></div></div></div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def _train_clusterer(openai_client=None) -> StudentClusterer:
    df = load_summary_df()
    clusterer = StudentClusterer()
    clusterer.fit(df, openai_client=openai_client)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clusterer.save(_MODEL_PATH)
    return clusterer


# ══════════════════════════════════════════════════════════════════════════════
#  UMAP scatter plot
# ══════════════════════════════════════════════════════════════════════════════

def _render_scatter(clusterer: StudentClusterer) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed — cannot render scatter plot.")
        return

    df_umap = clusterer.get_umap_df()

    fig = go.Figure()
    for k in range(clusterer.n_clusters):
        mask = df_umap["_cluster"] == k
        grp  = df_umap[mask]
        arch = clusterer.cluster_archetypes.get(k, {})
        color = arch.get("color", "#8B5CF6")
        name  = arch.get("name",  f"Cluster {k+1}")

        hover_parts = [f"<b>{name}</b><br>Confidence: %{{customdata:.0%}}"]
        custom = grp["_confidence"].values

        hover_cols = []
        extra_hover = ""
        if "cumulative_gpa" in grp.columns:
            hover_cols.append(grp["cumulative_gpa"].values)
            extra_hover += "<br>GPA: %{customdata[1]:.2f}"
        if "risk_level" in grp.columns:
            hover_cols.append(grp["risk_level"].values)
            extra_hover += "<br>Risk: %{customdata[2]}"

        customdata = np.column_stack(
            [custom] + [c if len(hover_cols) > 0 else np.zeros(len(grp)) for c in hover_cols]
        ) if hover_cols else custom.reshape(-1, 1)

        fig.add_trace(
            go.Scatter(
                x=grp["_umap_x"].values,
                y=grp["_umap_y"].values,
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=5,
                    opacity=0.72,
                    line=dict(width=0),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Confidence: %{customdata[0]:.0%}"
                    + (f"<br>GPA: %{{customdata[1]:.2f}}" if "cumulative_gpa" in grp.columns else "")
                    + (f"<br>Risk: %{{customdata[2]}}" if "risk_level" in grp.columns else "")
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color="#C4B5FD", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color="#9D77F5"),
        ),
        xaxis=dict(
            title="UMAP Dim 1",
            showgrid=True,
            gridcolor="rgba(139,92,246,.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title="UMAP Dim 2",
            showgrid=True,
            gridcolor="rgba(139,92,246,.08)",
            zeroline=False,
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cluster profile cards
# ══════════════════════════════════════════════════════════════════════════════

def _render_cluster_cards(clusterer: StudentClusterer) -> None:
    n = clusterer.n_clusters
    cols_per_row = 3
    clusters = list(range(n))

    for row_start in range(0, n, cols_per_row):
        row_clusters = clusters[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_clusters))
        for col, k in zip(cols, row_clusters):
            arch    = clusterer.cluster_archetypes.get(k, {})
            profile = clusterer.cluster_profiles.get(k, {})
            color   = arch.get("color", "#8B5CF6")
            name    = arch.get("name",  f"Cluster {k+1}")
            desc    = arch.get("description", "")
            size    = profile.get("size", 0)
            frac    = profile.get("fraction", 0)
            gpa     = profile.get("cumulative_gpa", 0)
            att     = profile.get("avg_attendance", 0)
            fail_r  = profile.get("failed_ratio", 0)
            study   = profile.get("study_hours", 0)
            risk_dist = profile.get("risk_dist", {})

            with col:
                st.markdown(
                    f"<div style='border:1.5px solid {color}55;border-radius:18px;"
                    f"padding:20px;background:rgba(255,255,255,.025);height:100%'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>"
                    f"<div style='width:12px;height:12px;border-radius:50%;"
                    f"background:{color};flex-shrink:0'></div>"
                    f"<div style='font-size:.95rem;font-weight:700;color:{color}'>"
                    f"{name}</div></div>"
                    f"<div style='font-size:.73rem;color:#7C6FAD;margin-bottom:14px;"
                    f"line-height:1.45'>{desc}</div>"
                    f"<div style='font-size:.67rem;color:#5B4D8A;margin-bottom:10px'>"
                    f"{size:,} students &nbsp;·&nbsp; {frac:.0%} of cohort</div>"
                    + _feature_bar("Avg GPA",        gpa,    4.0,   color)
                    + _feature_bar("Attendance (%)", att,    100.0, color)
                    + _feature_bar("Study hrs/wk",   study,  30.0,  color)
                    + _feature_bar("Failed ratio",   fail_r, 1.0,   "#EF4444")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                # Risk breakdown badges
                if risk_dist:
                    badge_html = " ".join(
                        _badge(r, "#10B981" if "Low" in r else "#F59E0B" if "Medium" in r else "#EF4444")
                        + f" <span style='font-size:.68rem;color:#5B4D8A'>{v:.0%}</span>"
                        for r, v in risk_dist.items()
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  "What's My Archetype?" form
# ══════════════════════════════════════════════════════════════════════════════

def _render_archetype_form(clusterer: StudentClusterer) -> None:
    st.markdown(
        "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
        "text-transform:uppercase;letter-spacing:.1em;margin:24px 0 14px'>"
        "🔍 &nbsp;What's My Archetype?</div>",
        unsafe_allow_html=True,
    )

    with st.form("archetype_form"):
        c1, c2, c3 = st.columns(3)
        avg_attendance  = c1.slider("Attendance (%)",      50, 100, 82)
        avg_assignments = c2.slider("Assignments (%)",     40, 100, 75)
        avg_midterm     = c1.slider("Midterm score (%)",   30, 100, 68)
        avg_final       = c2.slider("Final score (%)",     30, 100, 70)
        avg_overall     = c3.slider("Overall score (%)",   30, 100, 69)
        failed_ratio    = c3.slider("Failed course ratio", 0.0, 1.0, 0.05, step=0.01)
        gpa             = c1.slider("Current GPA",         0.5, 4.0, 2.8, step=0.05)
        study_hours     = c2.slider("Study hours / week",  5, 40, 18)

        submitted = st.form_submit_button("Predict My Archetype →", type="primary")

    if submitted:
        features = {
            "avg_attendance":  avg_attendance,
            "avg_assignments": avg_assignments,
            "avg_midterm":     avg_midterm,
            "avg_final":       avg_final,
            "avg_overall":     avg_overall,
            "failed_ratio":    failed_ratio,
            "cumulative_gpa":  gpa,
            "study_hours":     study_hours,
        }
        result = clusterer.predict_student(features)
        arch   = result["archetype"]
        color  = arch.get("color", "#8B5CF6")
        prob   = result["probability"]

        st.markdown(
            f"<div style='border:2px solid {color};border-radius:18px;"
            f"padding:24px;background:rgba(255,255,255,.03);margin-top:16px'>"
            f"<div style='font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;"
            f"color:#5B4D8A;margin-bottom:6px'>Your archetype</div>"
            f"<div style='font-size:1.5rem;font-weight:800;color:{color};"
            f"margin-bottom:8px'>{arch.get('name','—')}</div>"
            f"<div style='font-size:.82rem;color:#7C6FAD;margin-bottom:12px'>"
            f"{arch.get('description','')}</div>"
            f"<div style='font-size:.72rem;color:#5B4D8A'>"
            f"Cluster {result['cluster_id']+1} &nbsp;·&nbsp; "
            f"Confidence {prob:.0%}</div></div>",
            unsafe_allow_html=True,
        )

        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        all_probs = result["all_probs"]
        prob_cols = st.columns(len(all_probs))
        for col, (k, p) in zip(prob_cols, all_probs.items()):
            k_arch = clusterer.cluster_archetypes.get(k, {})
            col.metric(
                label=k_arch.get("name", f"C{k+1}"),
                value=f"{p:.0%}",
            )

        # Save result to session state for Product C to consume
        st.session_state["cluster_result"] = result
        st.session_state["cluster_features"] = features


# ══════════════════════════════════════════════════════════════════════════════
#  BIC curve
# ══════════════════════════════════════════════════════════════════════════════

def _render_bic_curve(clusterer: StudentClusterer) -> None:
    if not clusterer.bic_scores:
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    ks   = clusterer.k_range_tested
    bics = clusterer.bic_scores

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks, y=bics,
            mode="lines+markers",
            line=dict(color="#8B5CF6", width=2),
            marker=dict(size=8, color="#C4B5FD"),
            name="BIC",
        )
    )
    # Highlight optimal K
    opt_bic = min(bics)
    opt_k   = ks[bics.index(opt_bic)]
    fig.add_trace(
        go.Scatter(
            x=[opt_k], y=[opt_bic],
            mode="markers",
            marker=dict(size=14, color="#10B981", symbol="star"),
            name=f"Optimal K={opt_k}",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color="#C4B5FD", size=11),
        xaxis=dict(title="Number of Clusters (K)", dtick=1, showgrid=False),
        yaxis=dict(title="BIC Score (lower = better)", showgrid=True,
                   gridcolor="rgba(139,92,246,.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ══════════════════════════════════════════════════════════════════════════════

def render_clustering_page(openai_client=None) -> None:
    """
    Main entry point.  Call from phase7_streamlit_app.py inside a tab.

    Parameters
    ----------
    openai_client : optional OpenAI client used for GPT archetype labeling.
                    If None, rule-based names are used instead.
    """
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:800;color:#C4B5FD;"
        "margin-bottom:4px'>🧩 Student Archetype Clustering</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:.83rem;color:#4A3A7A;margin-bottom:20px'>"
        "Gaussian Mixture Model + UMAP discovers natural student archetypes "
        "from academic behaviour patterns — no labels needed.</div>",
        unsafe_allow_html=True,
    )

    # ── Try loading a saved model ──────────────────────────────────────────────
    clusterer: StudentClusterer | None = st.session_state.get("_clusterer_obj")
    if clusterer is None:
        clusterer = _get_clusterer_cached()
        if clusterer is not None:
            st.session_state["_clusterer_obj"] = clusterer

    # ── Train / retrain button ─────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_btn = st.button(
            "🔄  Train Clustering Model",
            type="primary" if clusterer is None else "secondary",
            use_container_width=True,
        )
    with col_info:
        if clusterer is None:
            st.info("No trained model found. Click **Train** to cluster all students.")
        else:
            st.success(
                f"Model loaded — {clusterer.n_clusters} archetypes  "
                f"·  silhouette {clusterer.silhouette:.3f}"
            )

    if train_btn:
        with st.spinner("Training clustering model… (usually < 30 s)"):
            try:
                clusterer = _train_clusterer(openai_client)
                st.session_state["_clusterer_obj"] = clusterer
                # Invalidate cached version
                _get_clusterer_cached.clear()
                st.success(
                    f"✅ Trained — {clusterer.n_clusters} archetypes discovered  "
                    f"·  silhouette={clusterer.silhouette:.3f}"
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

    if clusterer is None:
        return   # waiting for first train

    # ── Quality metrics row ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Archetypes (K)",  clusterer.n_clusters)
    m2.metric("Silhouette",      f"{clusterer.silhouette:.3f}")
    m3.metric("Calinski-Harabasz", f"{clusterer.calinski_harabasz:.0f}")
    m4.metric("Davies-Bouldin",  f"{clusterer.davies_bouldin:.3f}")

    st.divider()

    # ── UMAP scatter + BIC curve side by side ─────────────────────────────────
    col_scatter, col_bic = st.columns([2, 1])
    with col_scatter:
        st.markdown(
            "<div style='font-size:.72rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
            "UMAP Projection</div>",
            unsafe_allow_html=True,
        )
        _render_scatter(clusterer)
    with col_bic:
        st.markdown(
            "<div style='font-size:.72rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
            "BIC Model Selection</div>",
            unsafe_allow_html=True,
        )
        _render_bic_curve(clusterer)
        # Brief interpretation
        st.caption(
            f"K={clusterer.n_clusters} minimises BIC (Bayesian Information Criterion). "
            f"Lower BIC = better balance between fit quality and model complexity."
        )

    st.divider()

    # ── Cluster profile cards ──────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px'>"
        "📋 &nbsp;Archetype Profiles</div>",
        unsafe_allow_html=True,
    )
    _render_cluster_cards(clusterer)

    st.divider()

    # ── Archetype prediction form ──────────────────────────────────────────────
    _render_archetype_form(clusterer)
