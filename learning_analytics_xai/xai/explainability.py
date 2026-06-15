"""
Phase 6 — Explainable AI (SHAP) Module

Provides:
  - Global explainability : SHAP summary plots, feature importance
  - Local explainability  : SHAP waterfall plot for one student
  - PDP                   : Partial Dependence Plots for top features
  - PCA                   : 2D visualization of student clusters

Public API used by the dashboard:
  ExplainabilityEngine.explain_student(feature_vector, student_name)
    -> ExplainResult(shap_values, top_factors, waterfall_fig, ...)

  ExplainabilityEngine.global_summary_fig()       -> matplotlib Figure
  ExplainabilityEngine.pdp_fig(feature_name)      -> matplotlib Figure
  ExplainabilityEngine.pca_fig()                  -> matplotlib Figure
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler

HERE   = Path(__file__).parent
DATA   = HERE.parent / "data"
MODELS = HERE.parent / "models"

# Human-readable feature labels (maps internal column names -> display names)
FEATURE_LABELS = {
    # ── Original 21 ML features ───────────────────────────────────────────────
    "avg_attendance":        "Attendance Rate",
    "avg_assignments":       "Assignment Scores",
    "avg_quizzes":           "Quiz Scores",
    "avg_labs":              "Lab Scores",
    "avg_midterm":           "Midterm Score",
    "avg_final":             "Final Exam Score",
    "avg_overall":           "Overall Course Score",
    "semester":              "Current Semester",
    "credits_registered":    "Credits Registered",
    "credits_passed":        "Credits Passed",
    "failed_courses":        "Failed Courses",
    "attendance_risk_score": "Attendance Risk",
    "course_difficulty_index":"Programme Difficulty",
    "failed_course_ratio":   "Failure Rate",
    "performance_trend":     "Performance Trend",
    "credit_completion_ratio":"Credit Completion Rate",
    "academic_consistency":  "Academic Consistency",
    "study_efficiency":      "Study Efficiency",
    "semester_momentum":     "Learning Momentum",
    "programme_encoded":     "Programme Type",
    "school_encoded":        "School",
    # ── Curriculum-aware features (Phase 4b) ─────────────────────────────────
    "graduation_progress_ratio":    "Graduation Progress",
    "expected_progress_ratio":      "Expected Progress Pace",
    "graduation_delay_semesters":   "Curriculum Delay (semesters)",
    "core_completion_ratio":        "Core Course Completion",
    "prereq_completion_proxy":      "Prerequisite Completion",
    "blocked_progress_ratio":       "Blocked Degree Progress",
    "curriculum_alignment_proxy":   "Curriculum Alignment",
    "curriculum_readiness_score":   "Graduation Readiness Score",
}

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#27AE60", 1: "#F39C12", 2: "#E74C3C"}


@dataclass
class ExplainResult:
    """Result from explaining a single student's prediction."""
    predicted_gpa:          float
    predicted_risk:         int
    risk_label:             str
    risk_color:             str
    risk_proba:             list[float]
    shap_values_gpa:        np.ndarray            # shape (n_features,)
    shap_base_gpa:          float
    top_positive:           list[dict]            # top features helping GPA
    top_negative:           list[dict]            # top features hurting GPA
    waterfall_fig:          Optional[object]      # matplotlib Figure
    feature_names:          list[str]
    feature_values:         np.ndarray
    academic_health:        float                 # 0-100 composite score
    curriculum_narratives:  list[str] = field(default_factory=list)  # curriculum XAI sentences


class ExplainabilityEngine:
    """Loads trained XGBoost models and provides SHAP explanations."""

    def __init__(self):
        self._gpa_model  = None
        self._risk_model = None
        self._scaler:   Optional[StandardScaler] = None
        self._feat_cols: list[str] = []
        self._shap_explainer_gpa  = None
        self._shap_explainer_risk = None
        self._X_train_sample: Optional[np.ndarray] = None   # background sample for SHAP
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

        # Load a sample of training data as SHAP background
        df_feat = pd.read_csv(DATA / "features_train.csv")
        X_all   = df_feat[self._feat_cols].values.astype(np.float32)
        X_scaled = self._scaler.transform(X_all)
        idx = np.random.default_rng(42).choice(len(X_scaled), size=min(500, len(X_scaled)), replace=False)
        self._X_train_sample = X_scaled[idx]

        # Build TreeExplainer (fast, exact SHAP for tree models)
        self._shap_explainer_gpa  = shap.TreeExplainer(self._gpa_model)
        self._shap_explainer_risk = shap.TreeExplainer(self._risk_model)
        self._loaded = True

    # ── Local explanation for one student ─────────────────────────────────────

    def explain_student(
        self,
        feature_vector: dict | np.ndarray | list,
        student_name: str = "Student",
    ) -> ExplainResult:
        """
        Explain a single student's academic prediction.

        Parameters
        ----------
        feature_vector : dict with feature names as keys, OR array in feature_cols order
        student_name   : display name for plot titles

        Returns
        -------
        ExplainResult with all explanation artefacts
        """
        self._load()

        # Normalise input to numpy array
        if isinstance(feature_vector, dict):
            x_raw = np.array([feature_vector.get(c, 0.0) for c in self._feat_cols], dtype=np.float32)
        else:
            x_raw = np.array(feature_vector, dtype=np.float32)

        x_scaled = self._scaler.transform(x_raw.reshape(1, -1))

        # Predictions
        pred_gpa      = float(np.clip(self._gpa_model.predict(x_scaled)[0], 0.0, 4.0))
        risk_proba    = self._risk_model.predict_proba(x_scaled)[0].tolist()
        pred_risk     = int(np.argmax(risk_proba))
        risk_label    = RISK_LABELS[pred_risk]
        risk_color    = RISK_COLORS[pred_risk]

        # SHAP values for GPA (regression)
        sv_obj = self._shap_explainer_gpa(x_scaled)
        sv_gpa = sv_obj.values[0]           # shape (n_features,)
        base_gpa = float(self._shap_explainer_gpa.expected_value)

        # Rank features by absolute SHAP impact
        impacts = list(zip(self._feat_cols, sv_gpa, x_raw))
        impacts.sort(key=lambda t: abs(t[1]), reverse=True)

        top_positive = [
            {
                "feature":      FEATURE_LABELS.get(c, c),
                "raw_name":     c,
                "shap_value":   round(float(s), 4),
                "feature_value": round(float(v), 2),
                "direction":    "positive",
            }
            for c, s, v in impacts if s > 0
        ][:5]

        top_negative = [
            {
                "feature":      FEATURE_LABELS.get(c, c),
                "raw_name":     c,
                "shap_value":   round(float(s), 4),
                "feature_value": round(float(v), 2),
                "direction":    "negative",
            }
            for c, s, v in impacts if s < 0
        ][:5]

        # Academic health score (0-100)
        # Maps: GPA 0→0, GPA 4→100 with a mild S-curve
        academic_health = float(np.clip(pred_gpa / 4.0 * 100, 0, 100))

        # Waterfall plot
        waterfall_fig = self._waterfall_plot(sv_gpa, base_gpa, x_raw, student_name, pred_gpa)

        return ExplainResult(
            predicted_gpa   = round(pred_gpa, 3),
            predicted_risk  = pred_risk,
            risk_label      = risk_label,
            risk_color      = risk_color,
            risk_proba      = [round(p, 4) for p in risk_proba],
            shap_values_gpa = sv_gpa,
            shap_base_gpa   = base_gpa,
            top_positive    = top_positive,
            top_negative    = top_negative,
            waterfall_fig   = waterfall_fig,
            feature_names   = self._feat_cols,
            feature_values  = x_raw,
            academic_health = round(academic_health, 1),
        )

    def _waterfall_plot(
        self, shap_vals: np.ndarray, base: float,
        feature_vals: np.ndarray, title: str, pred_gpa: float
    ):
        """Custom waterfall plot (doesn't require shap.plots.waterfall)."""
        # Take top 10 by absolute impact
        n_show = min(10, len(self._feat_cols))
        idx = np.argsort(np.abs(shap_vals))[::-1][:n_show][::-1]   # low impact at bottom

        names  = [FEATURE_LABELS.get(self._feat_cols[i], self._feat_cols[i]) for i in idx]
        vals   = shap_vals[idx]
        fvals  = feature_vals[idx]
        colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in vals]

        fig, ax = plt.subplots(figsize=(10, max(6, n_show * 0.55)))
        ax.barh(range(n_show), vals, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(
            [f"{n} = {fv:.1f}" for n, fv in zip(names, fvals)],
            fontsize=9
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on predicted GPA)", fontsize=10)
        ax.set_title(
            f"SHAP Explanation — {title}\n"
            f"Base GPA: {base:.2f} → Predicted: {pred_gpa:.2f}",
            fontsize=11, fontweight="bold"
        )
        # Legend
        pos_patch = mpatches.Patch(color="#27AE60", label="Increases GPA")
        neg_patch = mpatches.Patch(color="#E74C3C", label="Decreases GPA")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── Global explanation ─────────────────────────────────────────────────────

    def global_summary_fig(self, max_display: int = 15) -> plt.Figure:
        """SHAP summary beeswarm-style plot over training background sample."""
        self._load()
        sv = self._shap_explainer_gpa(self._X_train_sample)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv.values,
            self._X_train_sample,
            feature_names=[FEATURE_LABELS.get(c, c) for c in self._feat_cols],
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()
        fig.suptitle("Global SHAP Summary — GPA Prediction Model", fontsize=12, fontweight="bold")
        return fig

    def feature_importance_fig(self) -> plt.Figure:
        """Bar chart of mean |SHAP| feature importance."""
        self._load()
        sv = self._shap_explainer_gpa(self._X_train_sample)
        mean_abs = np.abs(sv.values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:15]

        names  = [FEATURE_LABELS.get(self._feat_cols[i], self._feat_cols[i]) for i in sorted_idx]
        values = mean_abs[sorted_idx]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_idx)))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(names))[::-1], values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title("Feature Importance (SHAP) — GPA Prediction", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── PDP ───────────────────────────────────────────────────────────────────

    def pdp_fig(self, feature_name: str) -> Optional[plt.Figure]:
        """Partial Dependence Plot for a single feature."""
        self._load()
        if feature_name not in self._feat_cols:
            return None
        feat_idx = self._feat_cols.index(feature_name)
        display_name = FEATURE_LABELS.get(feature_name, feature_name)

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            PartialDependenceDisplay.from_estimator(
                self._gpa_model,
                self._X_train_sample,
                [feat_idx],
                feature_names=[FEATURE_LABELS.get(c, c) for c in self._feat_cols],
                ax=ax,
            )
            ax.set_title(f"Partial Dependence — {display_name}", fontsize=11, fontweight="bold")
            ax.set_xlabel(display_name)
            ax.set_ylabel("Predicted GPA")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            return fig
        except Exception:
            return None

    # ── Curriculum waterfall plot ─────────────────────────────────────────────

    @staticmethod
    def curriculum_waterfall_fig(
        curriculum_shap_values: dict,
        student_name: str = "Student",
        predicted_gpa: float = 0.0,
    ) -> Optional[plt.Figure]:
        """
        Curriculum-aware SHAP-equivalent waterfall chart.

        Uses rule-based curriculum impact scores (from CurriculumEngine.get_curriculum_shap_values)
        to create a SHAP-style visualization of curriculum factors affecting academic outcomes.
        These are not ML-derived SHAP values but are grounded in Zewail degree requirements
        and academic regulations.
        """
        if not curriculum_shap_values:
            return None

        # Sort by impact magnitude, largest at bottom (waterfall convention)
        items = sorted(curriculum_shap_values.items(), key=lambda x: x[1])
        names  = [n for n, _ in items]
        values = [v for _, v in items]
        colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in values]
        n_show = len(names)

        fig, ax = plt.subplots(figsize=(10, max(5, n_show * 0.65)))
        ax.barh(range(n_show), values, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Curriculum Impact on Predicted GPA (estimated)", fontsize=10)
        ax.set_title(
            f"Curriculum XAI — {student_name}\n"
            f"Zewail Academic Context  |  Predicted GPA: {predicted_gpa:.2f}",
            fontsize=11, fontweight="bold",
        )
        pos_patch = mpatches.Patch(color="#27AE60", label="Positive curriculum factor")
        neg_patch = mpatches.Patch(color="#E74C3C", label="Negative curriculum factor")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── PCA ───────────────────────────────────────────────────────────────────

    def pca_fig(self) -> plt.Figure:
        """2D PCA of all students coloured by predicted GPA."""
        self._load()
        df_feat = pd.read_csv(DATA / "features_train.csv")
        X       = df_feat[self._feat_cols].values.astype(np.float32)
        X_sc    = self._scaler.transform(X)
        y_gpa   = self._gpa_model.predict(X_sc)
        y_risk  = self._risk_model.predict(X_sc)

        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X_sc)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("PCA — Student Clusters", fontsize=12, fontweight="bold")

        # Coloured by GPA
        sc0 = axes[0].scatter(X2[:, 0], X2[:, 1], c=y_gpa, cmap="viridis",
                               alpha=0.4, s=8, linewidths=0)
        plt.colorbar(sc0, ax=axes[0], label="Predicted GPA")
        axes[0].set_title("Coloured by Predicted GPA", fontsize=10)
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        axes[0].grid(alpha=0.2)

        # Coloured by risk
        risk_palette = np.array(["#27AE60","#F39C12","#E74C3C"])
        axes[1].scatter(X2[:, 0], X2[:, 1], c=risk_palette[y_risk],
                         alpha=0.4, s=8, linewidths=0)
        patches = [
            mpatches.Patch(color="#27AE60", label="Low Risk"),
            mpatches.Patch(color="#F39C12", label="Medium Risk"),
            mpatches.Patch(color="#E74C3C", label="High Risk"),
        ]
        axes[1].legend(handles=patches, fontsize=9, loc="upper right")
        axes[1].set_title("Coloured by Risk Level", fontsize=10)
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        return fig


# Singleton instance (shared across the Streamlit app)
_engine: Optional[ExplainabilityEngine] = None


def get_engine() -> ExplainabilityEngine:
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    print("=== Phase 6: SHAP Explainability — standalone test ===")

    eng = get_engine()
    eng._load()

    # Test with a sample student
    test_student = {
        "avg_attendance":        72.0,
        "avg_assignments":       65.0,
        "avg_quizzes":           60.0,
        "avg_labs":              70.0,
        "avg_midterm":           63.0,
        "avg_final":             61.0,
        "avg_overall":           63.5,
        "semester":              4,
        "credits_registered":    60,
        "credits_passed":        48,
        "failed_courses":        3,
        "attendance_risk_score": 0.08,
        "course_difficulty_index": 0.55,
        "failed_course_ratio":   0.15,
        "performance_trend":     -2.0,
        "credit_completion_ratio": 0.80,
        "academic_consistency":  0.65,
        "study_efficiency":      1.2,
        "semester_momentum":     -1.5,
        "programme_encoded":     0.55,
        "school_encoded":        2.0,
    }

    result = eng.explain_student(test_student, "Test Student")
    print(f"\n  Predicted GPA   : {result.predicted_gpa}")
    print(f"  Risk Level      : {result.risk_label} ({result.risk_proba})")
    print(f"  Academic Health : {result.academic_health}")
    print(f"\n  Top POSITIVE factors:")
    for f in result.top_positive[:3]:
        print(f"    + {f['feature']:30s} SHAP={f['shap_value']:+.4f}  val={f['feature_value']}")
    print(f"\n  Top NEGATIVE factors:")
    for f in result.top_negative[:3]:
        print(f"    - {f['feature']:30s} SHAP={f['shap_value']:+.4f}  val={f['feature_value']}")

    # Save waterfall to buffer (verify it works)
    buf = io.BytesIO()
    result.waterfall_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    print(f"\n  Waterfall plot generated: {buf.tell()} bytes")

    print("\n=== Phase 6 Complete ===")
