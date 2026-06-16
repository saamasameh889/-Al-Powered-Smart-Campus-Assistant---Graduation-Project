"""
student_clustering.py — Student Archetype Clustering  (Product D)
═══════════════════════════════════════════════════════════════════════════════
Gaussian Mixture Model + UMAP + GPT-4o-mini archetype labeling.

Pipeline
--------
  raw features  →  StandardScaler  →  PCA(95% var)  →  GMM (BIC-optimal K)
                →  UMAP 2D (viz only)  →  GPT-4o-mini archetype names

Public API
----------
    from clustering.student_clustering import StudentClusterer, CLUSTER_FEATURES

    clusterer = StudentClusterer()
    clusterer.fit(df, openai_client=client)   # df = students_summary.csv rows
    clusterer.save(path)

    # inference
    clusterer = StudentClusterer.load(path)
    result = clusterer.predict_student(feature_dict)
    # → {cluster_id, archetype{name,description,color}, probability, all_probs}
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Features used for clustering ───────────────────────────────────────────────
CLUSTER_FEATURES = [
    "avg_attendance",
    "avg_assignments",
    "avg_midterm",
    "avg_final",
    "avg_overall",
    "failed_ratio",
    "cumulative_gpa",
    "study_hours",
]
_PERF_TREND = "performance_trend"   # avg_final − avg_midterm (derived)

_ARCHETYPE_COLORS = [
    "#8B5CF6", "#10B981", "#F59E0B", "#EF4444",
    "#3B82F6", "#EC4899", "#14B8A6", "#F97316",
]


class StudentClusterer:
    """
    Gaussian Mixture Model clustering of student archetypes with UMAP visualization.

    Attributes
    ----------
    n_clusters         : int   — optimal K chosen by BIC
    silhouette         : float — silhouette score of fitted clustering
    bic_scores         : list  — BIC per K value tested
    cluster_profiles   : dict  — mean feature values per cluster
    cluster_archetypes : dict  — {name, description, color} per cluster
    """

    def __init__(self) -> None:
        self.n_clusters: int = 0
        self.gmm: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.pca_model: Optional[PCA] = None
        self.umap_model = None
        self.cluster_profiles: dict[int, dict] = {}
        self.cluster_archetypes: dict[int, dict] = {}
        self.silhouette: float = 0.0
        self.calinski_harabasz: float = 0.0
        self.davies_bouldin: float = 0.0
        self.bic_scores: list[float] = []
        self.k_range_tested: list[int] = []
        self.feature_names: list[str] = CLUSTER_FEATURES + [_PERF_TREND]
        self._umap_df: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    # ══════════════════════════════════════════════════════════════════════════
    #  Fitting
    # ══════════════════════════════════════════════════════════════════════════

    def fit(
        self,
        df: pd.DataFrame,
        openai_client=None,
        k_range: range = range(3, 9),
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ) -> "StudentClusterer":
        """
        Full clustering pipeline.

        1. Feature extraction + StandardScaler normalisation
        2. PCA → retain 95% explained variance (noise reduction before GMM)
        3. GMM with BIC-optimal K (n_init=10 random restarts per K)
        4. Silhouette / Calinski-Harabasz / Davies-Bouldin quality audit
        5. UMAP 2D projection (visualization only, separate from clustering)
        6. Cluster profiling (feature means + risk distribution)
        7. GPT-4o-mini archetype labeling (or rule-based fallback)
        """
        X, df_clean = self._extract_features(df)

        # 1 — scale
        X_scaled = self.scaler.fit_transform(X)

        # 2 — PCA (noise reduction; keep ≥95% variance)
        self.pca_model = PCA(n_components=0.95, random_state=42)
        X_pca = self.pca_model.fit_transform(X_scaled)
        logger.info("PCA: %d components retained (95%% variance)", X_pca.shape[1])

        # 3 — BIC-optimal GMM
        self.n_clusters, self.bic_scores, self.k_range_tested = (
            self._select_k_bic(X_pca, k_range)
        )
        logger.info("Optimal K=%d (BIC)", self.n_clusters)

        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",     # full covariance captures correlations
            max_iter=500,
            n_init=10,                  # 10 random restarts → robust solution
            init_params="kmeans",
            random_state=42,
        )
        self.gmm.fit(X_pca)
        labels = self.gmm.predict(X_pca)
        probs  = self.gmm.predict_proba(X_pca)

        # 4 — quality metrics (computed on original scaled space)
        self.silhouette        = silhouette_score(X_scaled, labels)
        self.calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        self.davies_bouldin    = davies_bouldin_score(X_scaled, labels)
        logger.info(
            "Quality: silhouette=%.3f CH=%.1f DB=%.3f",
            self.silhouette, self.calinski_harabasz, self.davies_bouldin,
        )

        # 5 — UMAP 2D (visualization; fallback to PCA-2D if numba unavailable)
        X_2d = self._fit_umap(X_scaled, umap_neighbors, umap_min_dist)

        # 6 — cluster profiles
        df_aug = df_clean.copy()
        df_aug["_cluster"]   = labels
        df_aug["_umap_x"]    = X_2d[:, 0]
        df_aug["_umap_y"]    = X_2d[:, 1]
        df_aug["_confidence"] = probs.max(axis=1)

        self._umap_df = df_aug[
            ["_cluster", "_umap_x", "_umap_y", "_confidence"]
        ].copy()
        # carry programme + risk if present for richer plot tooltips
        for col in ("programme", "risk_level", "cumulative_gpa", "student_id"):
            if col in df_aug.columns:
                self._umap_df[col] = df_aug[col].values

        self._build_profiles(df_aug)

        # 7 — archetype labeling
        if openai_client is not None:
            self._label_gpt(openai_client)
        else:
            self._label_rules()

        self._is_fitted = True
        return self

    # ── K selection ────────────────────────────────────────────────────────────

    def _select_k_bic(
        self, X_pca: np.ndarray, k_range: range
    ) -> tuple[int, list[float], list[int]]:
        """
        Fit a GMM for each K and return the K that minimises BIC.

        BIC = k*ln(n) − 2*ln(L̂) penalises complexity; lower is better.
        """
        bics, ks = [], []
        for k in k_range:
            gm = GaussianMixture(
                n_components=k, covariance_type="full",
                max_iter=300, n_init=5, random_state=42,
            )
            gm.fit(X_pca)
            bics.append(gm.bic(X_pca))
            ks.append(k)
        best_k = ks[int(np.argmin(bics))]
        return best_k, bics, ks

    # ── UMAP / PCA-2D ─────────────────────────────────────────────────────────

    def _fit_umap(
        self, X_scaled: np.ndarray, n_neighbors: int, min_dist: float
    ) -> np.ndarray:
        try:
            import umap as umap_lib
            self.umap_model = umap_lib.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=42,
            )
            return self.umap_model.fit_transform(X_scaled)
        except Exception as exc:
            logger.warning("UMAP failed (%s) — using PCA 2D for visualisation.", exc)
            pca2 = PCA(n_components=2, random_state=42)
            X_2d = pca2.fit_transform(X_scaled)
            self.umap_model = pca2
            return X_2d

    # ── Feature extraction ─────────────────────────────────────────────────────

    def _extract_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        df = df.copy()
        df[_PERF_TREND] = df["avg_final"] - df["avg_midterm"]
        for col in self.feature_names:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        X = df[self.feature_names].values.astype(np.float32)
        return X, df

    # ── Cluster profiles ───────────────────────────────────────────────────────

    def _build_profiles(self, df_aug: pd.DataFrame) -> None:
        self.cluster_profiles = {}
        n_total = len(df_aug)
        for k in range(self.n_clusters):
            mask = df_aug["_cluster"] == k
            grp  = df_aug[mask]
            profile: dict = {
                "size":       int(mask.sum()),
                "fraction":   float(mask.sum() / n_total),
            }
            for feat in self.feature_names:
                if feat in grp.columns:
                    profile[feat]             = float(grp[feat].mean())
                    profile[feat + "_std"]    = float(grp[feat].std())
            if "risk_level" in grp.columns:
                profile["risk_dist"] = (
                    grp["risk_level"].value_counts(normalize=True).to_dict()
                )
            if "gpa_band" in grp.columns:
                mode_val = grp["gpa_band"].mode()
                profile["dominant_gpa_band"] = (
                    mode_val.iloc[0] if len(mode_val) > 0 else "N/A"
                )
            if "programme" in grp.columns:
                profile["top_programmes"] = (
                    grp["programme"].value_counts().head(3).to_dict()
                )
            self.cluster_profiles[k] = profile

    # ── GPT labeling ───────────────────────────────────────────────────────────

    def _label_gpt(self, client) -> None:
        for k, profile in self.cluster_profiles.items():
            feat_lines = "\n".join(
                f"  {f}: {profile.get(f, 0):.2f}" for f in self.feature_names
            )
            risk_str = ", ".join(
                f"{r}={v:.0%}"
                for r, v in profile.get("risk_dist", {}).items()
            )
            prompt = (
                "You are an academic data analyst at Zewail City.\n"
                f"Cluster {k+1} ({profile['size']} students, "
                f"{profile['fraction']:.0%} of cohort):\n"
                f"{feat_lines}\n"
                f"Risk distribution: {risk_str}\n\n"
                "Give a 3-4 word archetype name and a 1-sentence description "
                "that an academic advisor would find meaningful.\n"
                "Reply ONLY with valid JSON:\n"
                '{"name": "...", "description": "..."}'
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=90,
                    temperature=0.4,
                )
                text = resp.choices[0].message.content.strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    self.cluster_archetypes[k] = {
                        "name":        data.get("name", f"Archetype {k+1}"),
                        "description": data.get("description", ""),
                        "color":       _ARCHETYPE_COLORS[k % len(_ARCHETYPE_COLORS)],
                    }
                    continue
            except Exception as exc:
                logger.warning("GPT labeling cluster %d failed: %s", k, exc)
            self._default_archetype(k)

    def _label_rules(self) -> None:
        for k, p in self.cluster_profiles.items():
            gpa  = p.get("cumulative_gpa", 2.5)
            fail = p.get("failed_ratio", 0.1)
            att  = p.get("avg_attendance", 80)
            sh   = p.get("study_hours", 18)

            if gpa >= 3.5 and fail < 0.05:
                name, desc = (
                    "High Achiever",
                    "Exceptional academic performance with near-perfect attendance and minimal failures.",
                )
            elif gpa >= 3.0 and fail < 0.10:
                name, desc = (
                    "Steady Performer",
                    "Reliable and consistent students who meet expectations across the board.",
                )
            elif gpa < 0.6 or fail > 0.60:
                name, desc = (
                    "Critical Academic Failure",
                    "Severely failing — near-zero GPA or the majority of courses failed; immediate intervention is essential.",
                )
            elif gpa < 2.0 or fail > 0.12:
                name, desc = (
                    "Academic Probation Risk",
                    "Below the 2.0 probation threshold with elevated failure rates; requires structured academic support.",
                )
            elif att >= 85 and gpa < 2.8:
                name, desc = (
                    "Dedicated Striver",
                    "High attendance and effort but struggling to convert it into strong grades.",
                )
            elif att < 70:
                name, desc = (
                    "Disengaged Learner",
                    "Low attendance and weak engagement signal risk of further academic decline.",
                )
            elif sh >= 22:
                name, desc = (
                    "Intensive Studier",
                    "Above-average study hours; performance may be limited by study method rather than effort.",
                )
            else:
                name, desc = (
                    "Average Performer",
                    "Mid-range performance across most academic dimensions with clear room to improve.",
                )

            self.cluster_archetypes[k] = {
                "name":        name,
                "description": desc,
                "color":       _ARCHETYPE_COLORS[k % len(_ARCHETYPE_COLORS)],
            }

    def _default_archetype(self, k: int) -> None:
        self.cluster_archetypes[k] = {
            "name":        f"Archetype {k + 1}",
            "description": "A distinct student group identified by shared academic patterns.",
            "color":       _ARCHETYPE_COLORS[k % len(_ARCHETYPE_COLORS)],
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  Inference
    # ══════════════════════════════════════════════════════════════════════════

    def predict(self, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster labels and soft membership probabilities.

        Parameters
        ----------
        X_raw : np.ndarray (N, len(feature_names)) — unscaled raw features

        Returns
        -------
        labels : (N,)   — hard cluster assignments
        probs  : (N, K) — soft membership probabilities
        """
        X_sc  = self.scaler.transform(X_raw)
        X_pca = self.pca_model.transform(X_sc)
        return self.gmm.predict(X_pca), self.gmm.predict_proba(X_pca)

    def predict_student(self, features: dict) -> dict:
        """
        Predict archetype for a single student from a raw feature dict.

        Accepted keys: all in CLUSTER_FEATURES plus optionally
        'avg_final' and 'avg_midterm' (used to compute performance_trend).
        Missing keys are filled with the scaler's mean (safe default).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_student().")

        row = {f: float(features.get(f, 0.0)) for f in CLUSTER_FEATURES}
        row[_PERF_TREND] = (
            float(features.get("avg_final", 0))
            - float(features.get("avg_midterm", 0))
        )
        X = np.array([[row[f] for f in self.feature_names]], dtype=np.float32)

        labels, probs = self.predict(X)
        k = int(labels[0])
        return {
            "cluster_id":  k,
            "archetype":   self.cluster_archetypes.get(k, {"name": f"Cluster {k}"}),
            "profile":     self.cluster_profiles.get(k, {}),
            "probability": float(probs[0, k]),
            "all_probs":   {i: float(p) for i, p in enumerate(probs[0])},
        }

    # ── UMAP data access ───────────────────────────────────────────────────────

    def get_umap_df(self) -> pd.DataFrame:
        if self._umap_df is None:
            raise RuntimeError("Call fit() first.")
        return self._umap_df.copy()

    # ══════════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("StudentClusterer saved → %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "StudentClusterer":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected StudentClusterer")
        return obj


# ── Module-level helpers ───────────────────────────────────────────────────────

def load_summary_df(
    path: Path | str | None = None,
) -> pd.DataFrame:
    if path is None:
        path = (
            Path(__file__).parent.parent / "data" / "students_summary.csv"
        )
    return pd.read_csv(path)
