from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from src.model import FailureClassifier
from api.schemas import SensorReading, PredictResponse, SHAPFactor

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_clf: Optional[FailureClassifier] = None
_ready = False

LOW_RISK_CUTOFF = 0.25


def load_model() -> None:
    global _clf, _ready
    if not (MODEL_DIR / "xgb_classifier.joblib").exists():
        print("[predictor] Model not found — API in degraded mode. Run notebooks first.")
        return
    _clf = FailureClassifier.load(str(MODEL_DIR))
    _ready = True
    print(f"[predictor] Model loaded. Threshold={_clf.optimal_threshold:.2f}")


def is_ready() -> bool:
    return _ready


def risk_level_for_probability(proba: float, threshold: float) -> str:
    if proba < min(LOW_RISK_CUTOFF, threshold):
        return "low"
    if proba < threshold:
        return "medium"
    return "high"


def risk_zones() -> dict[str, float]:
    high_at = _clf.optimal_threshold if _clf else 0.5
    return {"low_below": LOW_RISK_CUTOFF, "high_at": high_at}


def predict(reading: SensorReading, include_shap: bool = True) -> PredictResponse:
    if not _ready or _clf is None:
        raise RuntimeError("Model not loaded.")

    features = _reading_to_features(reading)
    row_df = pd.DataFrame([features])
    proba = float(_clf.predict_proba(row_df)[0])
    pred  = int(proba >= _clf.optimal_threshold)

    risk_level = risk_level_for_probability(proba, _clf.optimal_threshold)

    shap_factors = None
    if include_shap:
        try:
            raw = _clf.top_shap_factors(features, top_n=5)
            shap_factors = [SHAPFactor(**f) for f in raw]
        except Exception as e:
            print(f"[predictor] SHAP failed: {e}")

    return PredictResponse(
        failure_probability=round(proba, 4),
        risk_level=risk_level,
        prediction=pred,
        threshold_used=_clf.optimal_threshold,
        estimated_cost_if_ignored=_clf.fn_cost if pred == 1 else None,
        top_shap_factors=shap_factors,
    )


# Fallback only — the trained model persists the actual training-time 75th
# percentile of `power` in `models/model_meta.json` (high_load_cutoff). The API
# reads it from the loaded model so this fallback should never apply in practice.
_HIGH_LOAD_FALLBACK = 66873.75


def _reading_to_features(r: SensorReading) -> dict:
    power = r.rotational_speed_rpm * r.torque_nm
    cutoff = (_clf.high_load_cutoff if _clf and _clf.high_load_cutoff is not None
              else _HIGH_LOAD_FALLBACK)
    return {
        "Air temperature [K]":       r.air_temperature_k,
        "Process temperature [K]":   r.process_temperature_k,
        "Rotational speed [rpm]":    r.rotational_speed_rpm,
        "Torque [Nm]":               r.torque_nm,
        "Tool wear [min]":           r.tool_wear_min,
        "temp_diff":                 r.process_temperature_k - r.air_temperature_k,
        "power":                     power,
        "wear_rate":                 r.tool_wear_min / (r.rotational_speed_rpm + 1),
        "torque_per_wear":           r.torque_nm / (r.tool_wear_min + 1),
        "high_load":                 int(power >= cutoff),
    }
