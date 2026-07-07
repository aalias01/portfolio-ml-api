"""
src/scorer.py — Health score computation and anomaly detection for HVAC units.

Pipeline:
    Feature matrix (from src/features.py)
        → Isolation Forest (primary anomaly detector)
        → LOF comparison
        → Anomaly score → 0–100 health score (inverted, per-unit normalized)
        → SHAP explanations (which sensors drive this unit's score)

Health score interpretation:
    90–100: Healthy — normal operating range
    70–89:  Monitor — slightly degraded efficiency, watch trends
    50–69:  Warning — investigate; likely declining COP or ΔT drift
    0–49:   Critical — anomalous operating point; schedule inspection

Usage:
    from src.scorer import Scorer
    scorer = Scorer()
    scorer.fit(X_train)
    results = scorer.score(X_new, building_ids=ids)
    scorer.save("models/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Health score thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "healthy":  (90, 100),
    "monitor":  (70, 89),
    "warning":  (50, 69),
    "critical": (0,  49),
}

TIER_CUTOFFS = {
    "healthy": 90,
    "monitor": 70,
    "warning": 50,
}


def score_to_tier(score: float) -> str:
    if score >= TIER_CUTOFFS["healthy"]:
        return "healthy"
    if score >= TIER_CUTOFFS["monitor"]:
        return "monitor"
    if score >= TIER_CUTOFFS["warning"]:
        return "warning"
    return "critical"


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------

class Scorer:
    """
    HVAC unit health scorer: Isolation Forest + LOF + 0–100 health gauge.

    Args:
        contamination: expected fraction of anomalous operating points (default 0.05)
            Set based on industry rule of thumb: ~5% of readings are genuinely
            anomalous. Validated against physical outliers in EDA.
        n_estimators: number of trees in Isolation Forest (default 200)
        use_lof: also train a LOF model for comparison (default True)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        use_lof: bool = True,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.use_lof = use_lof

        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.lof: Optional[LocalOutlierFactor] = (
            LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                               novelty=True, n_jobs=-1)
            if use_lof else None
        )
        self.feature_names: list[str] = []
        self._unit_score_stats: dict[str, dict] = {}  # per-unit score normalization
        self._shap_explainer = None
        self.is_fitted = False

    # LOF scalability caps — kNN queries on millions of rows are intractable,
    # and a random sample preserves the density structure LOF needs.
    LOF_MAX_FIT = 100_000     # max reference points for LOF fit
    LOF_MAX_SCORE = 100_000   # max rows scored by LOF in score(); rest are NA

    def fit(self, X: pd.DataFrame, building_ids: Optional[pd.Series] = None) -> "Scorer":
        """
        Fit the scaler, Isolation Forest, and (optionally) LOF.

        Args:
            X: feature matrix from src.features.get_feature_matrix()
            building_ids: optional Series of building_id values aligned with X
                If provided, computes per-unit score baselines for normalization.
        """
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.iforest.fit(X_scaled)
        if self.lof is not None:
            # LOF is kNN-based — fitting on millions of rows makes every later
            # query intractable. A 100k random reference sample preserves the
            # density structure while keeping predict() fast.
            if len(X_scaled) > self.LOF_MAX_FIT:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_scaled), self.LOF_MAX_FIT, replace=False)
                self.lof.fit(X_scaled[idx])
            else:
                self.lof.fit(X_scaled)

        # Compute raw anomaly scores on training data for per-unit normalization
        raw_scores = self.iforest.decision_function(X_scaled)  # higher = more normal

        if building_ids is not None:
            temp = pd.DataFrame({"raw_score": raw_scores, "building_id": building_ids.values})
            for bid, grp in temp.groupby("building_id"):
                self._unit_score_stats[str(bid)] = {
                    "mean": float(grp["raw_score"].mean()),
                    "std":  float(grp["raw_score"].std() + 1e-6),
                    "p5":   float(grp["raw_score"].quantile(0.05)),
                    "p95":  float(grp["raw_score"].quantile(0.95)),
                }

        self.is_fitted = True
        self._shap_explainer = None
        # predict() would rerun decision_function over all rows — derive the
        # flag from the scores we already have (predict == -1 iff score < 0).
        n_flagged = int((raw_scores < 0).sum())
        print(f"Scorer fitted: {len(X):,} samples, {n_flagged} anomalies ({n_flagged/len(X)*100:.1f}%)")
        return self

    def _raw_to_health_score(
        self,
        raw_score: float,
        building_id: Optional[str] = None,
    ) -> float:
        """
        Convert Isolation Forest decision_function score to 0–100 health score.

        The decision_function returns higher values for more normal points.
        We invert and normalize: low anomaly score → high health score.

        If unit-level stats exist, use per-unit normalization (better for
        units with systematically different operating profiles).
        """
        if building_id and str(building_id) in self._unit_score_stats:
            stats = self._unit_score_stats[str(building_id)]
            # Normalize relative to unit's own range
            normalized = (raw_score - stats["p5"]) / (stats["p95"] - stats["p5"] + 1e-6)
        else:
            # Global normalization
            normalized = (raw_score + 0.5) / 0.5  # IF scores typically in [-0.5, 0.5]

        # Clip and convert to 0–100 (higher = healthier)
        health = float(np.clip(normalized * 100, 0, 100))
        return health

    def score(
        self,
        X: pd.DataFrame,
        building_ids: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Score new data. Returns a DataFrame with health scores and anomaly flags.

        Returns columns:
            building_id (if provided), health_score (0–100), health_tier,
            anomaly_flag (1 = anomalous), iforest_score, lof_flag (if fitted)
        """
        if not self.is_fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.iforest.decision_function(X_scaled)
        # predict == -1 iff decision_function < 0; avoids a second full pass
        if_preds = np.where(raw_scores < 0, -1, 1)   # 1=normal, -1=anomaly

        # Vectorized health-score computation (a Python loop here would take
        # hours on millions of rows). Per-unit p5/p95 normalization where unit
        # stats exist, global fallback otherwise.
        if building_ids is not None:
            bids = building_ids.astype(str).to_numpy()
            p5  = np.array([self._unit_score_stats.get(b, {}).get("p5", np.nan) for b in
                            pd.unique(bids)])
            p95 = np.array([self._unit_score_stats.get(b, {}).get("p95", np.nan) for b in
                            pd.unique(bids)])
            stats_map = pd.DataFrame({"building_id": pd.unique(bids), "p5": p5, "p95": p95})
            tmp = pd.DataFrame({"building_id": bids, "raw": raw_scores}).merge(
                stats_map, on="building_id", how="left")
            normalized = np.where(
                tmp["p5"].notna(),
                (tmp["raw"] - tmp["p5"]) / (tmp["p95"] - tmp["p5"] + 1e-6),
                (tmp["raw"] + 0.5) / 0.5,
            )
        else:
            bids = np.full(len(X), None)
            normalized = (raw_scores + 0.5) / 0.5

        health = np.clip(normalized * 100, 0, 100).round(1)
        tiers = np.select(
            [health >= 90, health >= 70, health >= 50],
            ["healthy", "monitor", "warning"],
            default="critical",
        )

        out = pd.DataFrame({
            "building_id": bids,
            "health_score": health,
            "health_tier": tiers,
            "anomaly_flag": (if_preds == -1).astype(int),
            "iforest_score": np.round(raw_scores, 4),
        })

        # LOF comparison (optional). kNN queries don't scale to millions of
        # rows — above LOF_MAX_SCORE, score a random subset and leave the rest
        # NA (pandas nullable Int8) so agreement stats remain unbiased.
        if self.lof is not None:
            if len(X_scaled) > self.LOF_MAX_SCORE:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_scaled), self.LOF_MAX_SCORE, replace=False)
                lof_flag = pd.array([pd.NA] * len(out), dtype="Int8")
                lof_flag[idx] = (self.lof.predict(X_scaled[idx]) == -1).astype(int)
            else:
                lof_flag = pd.array(
                    (self.lof.predict(X_scaled) == -1).astype(int), dtype="Int8")
            out["lof_flag"] = lof_flag
            out["if_lof_agree"] = (out["anomaly_flag"] == out["lof_flag"]).astype("Int8")

        return out

    def score_single(
        self,
        x: dict,
        building_id: Optional[str] = None,
    ) -> dict:
        """
        Score a single data point (dict of feature_name → value).
        Used by the FastAPI /score endpoint.
        """
        row = pd.DataFrame([x])[self.feature_names].fillna(0)
        result_df = self.score(row, pd.Series([building_id]) if building_id else None)
        return result_df.iloc[0].to_dict()

    # ---------------------------------------------------------------------------
    # SHAP explanations
    # ---------------------------------------------------------------------------

    def explain(self, X: pd.DataFrame, max_display: int = 6) -> pd.DataFrame:
        """
        Compute SHAP values for the Isolation Forest model.

        Returns a DataFrame of shape (n_samples, n_features) with SHAP values.
        Positive SHAP = feature pushes toward anomaly (lowers health score).
        Negative SHAP = feature pushes toward normal (raises health score).
        """
        import shap
        X_scaled = self.scaler.transform(X)
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.iforest)
        shap_values = self._shap_explainer.shap_values(X_scaled)
        return pd.DataFrame(shap_values, columns=self.feature_names, index=X.index)

    def top_shap_factors(
        self,
        x: dict,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Return the top-n SHAP factors for a single prediction.
        Used by the FastAPI /score endpoint for explainability.
        """
        row = pd.DataFrame([x])[self.feature_names].fillna(0)
        shap_df = self.explain(row)
        row_shap = shap_df.iloc[0].abs().sort_values(ascending=False)
        factors = []
        for feat in row_shap.index[:top_n]:
            val = shap_df.iloc[0][feat]
            factors.append({
                "feature": feat,
                "shap_value": round(float(val), 4),
                "direction": "worsens_health" if val > 0 else "improves_health",
                "feature_value": round(float(x.get(feat, 0)), 4),
            })
        return factors

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, model_dir: str = "models/") -> None:
        path = Path(model_dir)
        path.mkdir(exist_ok=True)
        joblib.dump(self.scaler,  path / "isolation_forest_scaler.joblib")
        joblib.dump(self.iforest, path / "isolation_forest.joblib")
        if self.lof is not None:
            joblib.dump(self.lof, path / "lof_model.joblib")
        meta = {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "use_lof": self.use_lof,
            "unit_score_stats": self._unit_score_stats,
            "thresholds": THRESHOLDS,
        }
        (path / "scorer_meta.json").write_text(__import__("json").dumps(meta, indent=2))
        print(f"Scorer saved to {path}")

    @classmethod
    def load(cls, model_dir: str = "models/") -> "Scorer":
        path = Path(model_dir)
        meta = __import__("json").loads((path / "scorer_meta.json").read_text())
        scorer = cls(
            contamination=meta["contamination"],
            n_estimators=meta["n_estimators"],
            use_lof=meta["use_lof"],
        )
        scorer.scaler         = joblib.load(path / "isolation_forest_scaler.joblib")
        scorer.iforest        = joblib.load(path / "isolation_forest.joblib")
        scorer.feature_names  = meta["feature_names"]
        scorer._unit_score_stats = meta.get("unit_score_stats", {})
        if meta["use_lof"] and (path / "lof_model.joblib").exists():
            scorer.lof = joblib.load(path / "lof_model.joblib")
        scorer._shap_explainer = None
        scorer.is_fitted = True
        return scorer
