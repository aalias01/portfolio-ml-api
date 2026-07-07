"""
api/predictor.py — Model loading and inference for the HVAC Health Scoring API.

Loads the Scorer at startup. Provides score_single() and score_batch() for
the FastAPI endpoints. Gracefully degrades to a "not ready" state if models
haven't been trained yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.scorer import Scorer
from api.schemas import (
    DemoScenariosResponse,
    SensorReading,
    ScoreResponse,
    SHAPFactor,
    UnitListResponse,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEMO_READINGS_PATH = MODEL_DIR / "demo_readings.json"

# Module-level singleton — loaded once at startup
_scorer: Optional[Scorer] = None
_ready = False

# In-memory unit score cache for the "all units" dashboard view
# Populated by batch scoring from notebooks; refreshed on restart
_unit_cache: list[dict] = []
_demo_cache: Optional[DemoScenariosResponse] = None


def load_scorer() -> None:
    """Load the Scorer from MODEL_DIR. Called at API startup via lifespan."""
    global _scorer, _ready
    if not (MODEL_DIR / "isolation_forest.joblib").exists():
        print(f"[predictor] Model not found at {MODEL_DIR} — API in degraded mode. "
              f"Run notebooks/03_anomaly_detection.ipynb to train models.")
        return
    _scorer = Scorer.load(str(MODEL_DIR))
    _ready = True
    print(f"[predictor] Scorer loaded. Features: {len(_scorer.feature_names)}, "
          f"Contamination: {_scorer.contamination}")


def is_ready() -> bool:
    return _ready


def get_scorer() -> Optional[Scorer]:
    return _scorer


def score_single(reading: SensorReading, include_shap: bool = True) -> ScoreResponse:
    """
    Score a single HVAC unit sensor snapshot.

    Args:
        reading: validated SensorReading from the request body
        include_shap: compute SHAP explanations (adds ~50ms latency for single prediction)

    Returns:
        ScoreResponse with health_score, health_tier, anomaly_flag, SHAP factors
    """
    if not _ready or _scorer is None:
        raise RuntimeError("Scorer not loaded. Train models first.")

    features = _reading_to_features(reading)
    result = _scorer.score_single(features, building_id=reading.building_id)

    shap_factors = None
    if include_shap:
        try:
            raw_factors = _scorer.top_shap_factors(features, top_n=5)
            shap_factors = [SHAPFactor(**f) for f in raw_factors]
        except Exception as e:
            print(f"[predictor] SHAP failed: {e}")

    return ScoreResponse(
        building_id=reading.building_id,
        health_score=result.get("health_score", 0.0),
        health_tier=result.get("health_tier", "critical"),
        anomaly_flag=int(result.get("anomaly_flag", 1)),
        iforest_score=float(result.get("iforest_score", 0.0)),
        lof_flag=result.get("lof_flag"),
        if_lof_agree=result.get("if_lof_agree"),
        top_shap_factors=shap_factors,
    )


def get_all_units() -> UnitListResponse:
    """
    Return summary health scores for all units in the cached batch results.
    The cache is populated from notebook 03 output written to models/unit_baselines.joblib.
    """
    import joblib
    baselines_path = MODEL_DIR / "unit_baselines.joblib"
    if baselines_path.exists():
        units = joblib.load(str(baselines_path))
    elif _unit_cache:
        units = _unit_cache
    else:
        units = []

    meta = _load_unit_baseline_meta()
    tiers = [u.get("health_tier", "critical") for u in units]
    return UnitListResponse(
        units=sorted(units, key=lambda u: u.get("health_score", 0)),
        n_critical=tiers.count("critical"),
        n_warning=tiers.count("warning"),
        n_monitor=tiers.count("monitor"),
        n_healthy=tiers.count("healthy"),
        total=len(units),
        snapshot_generated=meta.get("generated"),
    )


def get_demo_readings() -> DemoScenariosResponse:
    """Return curated complete readings used by the public demo."""
    global _demo_cache
    if _demo_cache is not None:
        return _demo_cache
    if not DEMO_READINGS_PATH.exists():
        raise RuntimeError("Demo readings not found. Run scripts/curate_demo_readings.py.")
    try:
        payload = json.loads(DEMO_READINGS_PATH.read_text())
        _demo_cache = DemoScenariosResponse(**payload)
        return _demo_cache
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Could not load demo readings: {exc}") from exc


def _load_unit_baseline_meta() -> dict:
    meta_path = MODEL_DIR / "unit_baselines_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[predictor] Could not read unit baseline metadata: {exc}")
        return {}


def _reading_to_features(reading: SensorReading) -> dict:
    """Convert a SensorReading to the flat dict expected by Scorer.score_single()."""
    return {
        "cop_proxy":                    reading.cop_proxy,
        "delta_t_supply_proxy":         reading.delta_t_supply_proxy,
        "delta_t_refrigerant_proxy":    reading.delta_t_refrigerant_proxy,
        "load_ratio":                   reading.load_ratio,
        "rolling_cop_mean_24h":         _default(reading.rolling_cop_mean_24h, 0.0),
        "rolling_cop_std_24h":          _default(reading.rolling_cop_std_24h, 0.0),
        "rolling_load_mean_24h":        _default(reading.rolling_load_mean_24h, 0.0),
        "rolling_cop_mean_168h":        _default(reading.rolling_cop_mean_168h, 0.0),
        "cop_deviation_from_baseline":  _default(reading.cop_deviation_from_baseline, 0.0),
        "air_temperature":              _default(reading.air_temperature, 20.0),
        "dew_temperature":              _default(reading.dew_temperature, 15.0),
        "wind_speed":                   _default(reading.wind_speed, 2.0),
        "hour_of_day":                  _default(reading.hour_of_day, 12),
        "day_of_week":                  _default(reading.day_of_week, 2),
        "is_weekend":                   _default(reading.is_weekend, 0),
        "month":                        _default(reading.month, 6),
    }


def _default(value, fallback):
    """Preserve valid numeric zeroes while filling omitted optional fields."""
    return fallback if value is None else value
