"""
src/model.py — Training, evaluation, and threshold selection for failure classification.

Key concepts:
    1. Class imbalance handling: SMOTE + class_weight / scale_pos_weight
    2. Business-cost threshold tuning: FN=$50K, FP=$2K → threshold selected by cost
    3. Model persistence: joblib for API use

Usage:
    from src.model import FailureClassifier
    clf = FailureClassifier()
    clf.fit(X_train, y_train)
    results = clf.evaluate(X_test, y_test)
    clf.save("models/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Cost constants — tune these for the specific industry context
# ---------------------------------------------------------------------------
FN_COST = 50_000   # Missed failure → unplanned downtime
FP_COST = 2_000    # False alarm → unnecessary maintenance call


def business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return fn * FN_COST + fp * FP_COST


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Find the classification threshold that minimizes total business cost.

    Returns:
        (optimal_threshold, minimum_cost)

    Interview story:
        'Standard ML defaults to 0.5. For failure prediction, a missed failure costs
        $50K vs. $2K for unnecessary maintenance. I built a cost model and found the
        threshold that minimizes total operational cost instead of assuming 0.5.'
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.025)
    costs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        costs.append(business_cost(y_true, preds))
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


# ---------------------------------------------------------------------------
# FailureClassifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    """
    Binary failure classifier: Logistic Regression baseline → Random Forest → XGBoost.
    Handles class imbalance with SMOTE + class weights.
    Threshold selected by business cost analysis.

    Args:
        model_type: "logistic" | "rf" | "xgb" (default "xgb")
        use_smote: apply SMOTE oversampling before fitting (default True)
        fn_cost: false negative cost in dollars (default 50000)
        fp_cost: false positive cost in dollars (default 2000)
    """

    def __init__(
        self,
        model_type: str = "xgb",
        use_smote: bool = True,
        fn_cost: float = FN_COST,
        fp_cost: float = FP_COST,
        high_load_cutoff: Optional[float] = None,
    ):
        self.model_type = model_type
        self.use_smote = use_smote
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.optimal_threshold: float = 0.5
        self.feature_names: list[str] = []
        self._tree_explainer = None
        # 75th-percentile cutoff of `power` (rpm * torque) used to construct the
        # binary `high_load` engineered feature. Persisted so the API can build
        # the same feature for single-row predictions without re-deriving from
        # the training data.
        self.high_load_cutoff: Optional[float] = high_load_cutoff
        self.is_fitted = False

    def _build_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
        # Default: XGBoost
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=29,   # ~97/3 negative-to-positive ratio
            random_state=42,
            eval_metric="aucpr",
            use_label_encoder=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FailureClassifier":
        self.feature_names = list(X.columns)
        self._tree_explainer = None
        X_arr = self.scaler.fit_transform(X)

        if self.use_smote:
            sm = SMOTE(random_state=42)
            X_arr, y_arr = sm.fit_resample(X_arr, y)
            print(f"[model] After SMOTE: {np.bincount(y_arr.astype(int))}")
        else:
            y_arr = y.values

        self.model.fit(X_arr, y_arr)

        # Compute optimal threshold on training data (will refine on val set in notebook)
        train_proba = self.model.predict_proba(X_arr)[:, 1]
        self.optimal_threshold, cost = find_cost_optimal_threshold(y_arr, train_proba)
        print(f"[model] Fitted {self.model_type} | optimal_threshold={self.optimal_threshold:.2f} | train_cost=${cost:,.0f}")
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X) >= t).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: Optional[float] = None) -> dict:
        """Full evaluation suite: classification report + ROC-AUC + PR-AUC + business cost."""
        proba = self.predict_proba(X)
        t = threshold if threshold is not None else self.optimal_threshold
        preds = (proba >= t).astype(int)
        y_arr = y.values

        report = classification_report(y_arr, preds, output_dict=True)
        cost = business_cost(y_arr, preds)
        opt_t, opt_cost = find_cost_optimal_threshold(y_arr, proba)

        return {
            "threshold_used":      t,
            "roc_auc":             round(roc_auc_score(y_arr, proba), 4),
            "pr_auc":              round(average_precision_score(y_arr, proba), 4),
            "f1":                  round(f1_score(y_arr, preds), 4),
            "precision":           round(report["1"]["precision"], 4),
            "recall":              round(report["1"]["recall"], 4),
            "business_cost":       int(cost),
            "optimal_threshold":   round(opt_t, 3),
            "optimal_cost":        int(opt_cost),
            "confusion_matrix":    confusion_matrix(y_arr, preds).tolist(),
            "n_fn":                int(((preds == 0) & (y_arr == 1)).sum()),
            "n_fp":                int(((preds == 1) & (y_arr == 0)).sum()),
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """Stratified k-fold cross-validation on PR-AUC."""
        X_arr = self.scaler.transform(X[self.feature_names])
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_arr, y, cv=skf, scoring="average_precision")
        return {"cv_pr_auc_mean": round(scores.mean(), 4), "cv_pr_auc_std": round(scores.std(), 4)}

    def top_shap_factors(self, x: dict, top_n: int = 5) -> list[dict]:
        """Per-prediction SHAP factors for API response.

        Uses XGBoost's native `pred_contribs=True` for tree-based models —
        produces identical values to `shap.TreeExplainer` (which wraps the same
        XGBoost internals) but without depending on shap-vs-xgboost version
        compatibility. Falls back to shap for non-tree models.
        """
        row = pd.DataFrame([x])[self.feature_names].fillna(0)
        X_arr = self.scaler.transform(row)

        if self.model_type in ("xgb", "rf"):
            # XGBoost Booster: pred_contribs returns shape (n_rows, n_features + 1).
            # Last column is the base/expected value — drop it.
            if self.model_type == "xgb":
                import xgboost as xgb
                # XGBoost DMatrix rejects feature names containing '[', ']', '<' —
                # the AI4I raw columns have units in brackets, so pass positional.
                dmat = xgb.DMatrix(X_arr)
                contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
                sv = np.asarray(contribs[0][:-1])
            else:
                # Random Forest path uses shap (RF has stable shap support).
                import shap
                if self._tree_explainer is None:
                    self._tree_explainer = shap.TreeExplainer(self.model)
                explainer = self._tree_explainer
                shap_values = explainer.shap_values(X_arr)
                sv = np.asarray(shap_values[1][0] if isinstance(shap_values, list) else shap_values[0])
        else:
            import shap
            explainer = shap.LinearExplainer(self.model, X_arr)
            sv = np.asarray(explainer.shap_values(X_arr)[0])

        idx_sorted = np.argsort(np.abs(sv))[::-1][:top_n]
        return [
            {
                "feature":       self.feature_names[i],
                "shap_value":    round(float(sv[i]), 4),
                "direction":     "increases_risk" if sv[i] > 0 else "decreases_risk",
                "feature_value": round(float(x.get(self.feature_names[i], 0)), 4),
            }
            for i in idx_sorted
        ]

    def save(self, model_dir: str = "models/") -> None:
        path = Path(model_dir)
        path.mkdir(exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.model, path / "xgb_classifier.joblib")
        meta = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold,
            "fn_cost": self.fn_cost,
            "fp_cost": self.fp_cost,
            "high_load_cutoff": self.high_load_cutoff,
        }
        (path / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[model] Saved to {path}")

    @classmethod
    def load(cls, model_dir: str = "models/") -> "FailureClassifier":
        path = Path(model_dir)
        meta = json.loads((path / "model_meta.json").read_text())
        obj = cls(
            model_type=meta["model_type"],
            fn_cost=meta["fn_cost"],
            fp_cost=meta["fp_cost"],
            high_load_cutoff=meta.get("high_load_cutoff"),
        )
        obj.scaler = joblib.load(path / "scaler.joblib")
        obj.model = joblib.load(path / "xgb_classifier.joblib")
        obj.feature_names = meta["feature_names"]
        obj._tree_explainer = None
        obj.optimal_threshold = meta["optimal_threshold"]
        obj.is_fitted = True
        return obj
