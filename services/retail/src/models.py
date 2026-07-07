"""
src/models.py - Train, evaluate, save, and load all four models.

Models:
  1. Return-Likelihood Classifier (LightGBM / XGBoost)
  2. Excessive-Returner Anomaly Detector (Isolation Forest)
  3. Customer Segmentation (KMeans)
  4. Substitute Product Recommender, see src/recommender.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Model 1: Return-Likelihood Classifier
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    class_weight: str = "balanced",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    random_state: int = 42,
) -> LGBMClassifier:
    """Train LightGBM return-likelihood classifier.

    Uses class_weight='balanced' as the primary imbalance strategy.
    Compare to SMOTE variant in the classification notebook.
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=None,
    )
    return model


def evaluate_classifier(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return standard classification metrics + precision @ top decile."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    top_decile_n = int(len(y_test) * 0.10)
    top_decile_idx = np.argsort(y_prob)[::-1][:top_decile_n]
    precision_top_decile = y_test.iloc[top_decile_idx].mean()

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "precision_top_decile": float(precision_top_decile),
    }


def get_shap_values(
    model: LGBMClassifier,
    X: pd.DataFrame,
    sample_n: int = 500,
) -> tuple[shap.Explainer, np.ndarray]:
    """Return (explainer, shap_values) for a random sample of X."""
    if len(X) > sample_n:
        X_sample = X.sample(n=sample_n, random_state=42)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # LightGBM returns list [class0, class1]; take class 1 values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values


# ---------------------------------------------------------------------------
# Model 2: Excessive-Returner Anomaly Detection
# ---------------------------------------------------------------------------

def train_anomaly_detector(
    X_customer: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[IsolationForest, StandardScaler]:
    """Train Isolation Forest on customer-level behavioral features.

    Returns (model, scaler). Scaler is fit here so inference can reproduce
    the same normalization at predict time.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customer)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def predict_anomaly(
    model: IsolationForest,
    scaler: StandardScaler,
    X_customer: pd.DataFrame,
) -> pd.DataFrame:
    """Return anomaly scores and binary flags.

    score < 0 in Isolation Forest = anomalous.
    flag = 1 means excessive returner detected.
    """
    X_scaled = scaler.transform(X_customer)
    scores = model.decision_function(X_scaled)
    flags = (model.predict(X_scaled) == -1).astype(int)
    return pd.DataFrame(
        {"anomaly_score": scores, "anomaly_flag": flags},
        index=X_customer.index,
    )


# ---------------------------------------------------------------------------
# Model 3: Customer Segmentation
# ---------------------------------------------------------------------------

#: Default cluster_id → segment label mapping. KMeans cluster IDs are
#: arbitrary across re-runs, so this static dict is only a fallback. The
#: production label assignment is centroid-rule-based and lives in
#: ``scripts/build_api_artifacts.py`` (``assign_segment_labels``), which
#: deterministically picks the Returner cluster by max return_value_ratio,
#: then Premium Loyal by max monetary, then At-Risk by max recency, then
#: Healthy Browser as the remainder. Re-run that script whenever KMeans
#: is retrained.
SEGMENT_LABELS = {
    0: "Premium Loyal",
    1: "Healthy Browser",
    2: "At-Risk",
    3: "Returner",
}


def train_segmentation(
    X_customer: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler]:
    """Train KMeans k=4 on standardized RFM + return behavioral features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customer)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    return model, scaler


def assign_segments(
    model: KMeans,
    scaler: StandardScaler,
    X_customer: pd.DataFrame,
) -> pd.Series:
    """Return a Series of human-readable segment labels."""
    X_scaled = scaler.transform(X_customer)
    cluster_ids = model.predict(X_scaled)
    # Cluster → segment label mapping is determined in notebook 06 by inspecting
    # cluster centroids. Remap here after completing that analysis.
    return pd.Series(cluster_ids, index=X_customer.index).map(
        lambda c: SEGMENT_LABELS.get(c, f"Cluster {c}")
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(obj: Any, name: str) -> Path:
    """Save any model or scaler to models/ directory with joblib."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_model(name: str) -> Any:
    """Load a model or scaler from models/ directory."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run the corresponding training notebook first."
        )
    return joblib.load(path)
