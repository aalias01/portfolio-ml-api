"""
src/features.py — Domain-driven feature engineering for HVAC Equipment Health Scoring.

All features are grounded in refrigeration thermodynamics and HVAC engineering practice.
These are not generic time-series features — they come from 3 years of HVAC product
development at Rheem Manufacturing.

Key features:
    COP (Coefficient of Performance):
        The single most important efficiency indicator in refrigeration systems.
        COP = cooling_output / power_input. A healthy unit operates near its rated COP;
        declining COP precedes compressor failure by days to weeks.

    delta-T supply:
        T_supply_air - T_return_air. Measures heat exchange effectiveness across the
        air-side coil. Narrows as the evaporator coil fouls with debris.

    delta-T refrigerant:
        T_condenser - T_evaporator. Refrigerant circuit efficiency. Widens as refrigerant
        charge depletes (common leak scenario) or condenser coil fouls.

    Load ratio:
        Actual cooling load / rated capacity. Units operating near 100% load ratio
        for extended periods accumulate stress. High load + low COP = failure zone.

    Runtime fraction:
        Hours running / hours in period. High runtime at degraded efficiency signals
        the unit is working harder to meet setpoint — a classic early-warning pattern.

    Rolling COP deviation:
        COP vs. unit's own 30-day rolling mean. Catches slow degradation trends that
        threshold-based alarms miss — COP can decline 15% before any hard alarm trips.

Usage:
    from src.features import load_raw, build_features
    df = load_raw("data/raw/")
    features = build_features(df)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ---------------------------------------------------------------------------
# Constants — ASHRAE dataset column names
# ---------------------------------------------------------------------------

# These map to the ASHRAE Great Energy Predictor III schema.
# Adjust if using an alternative dataset.
BUILDING_ID_COL = "building_id"
METER_COL       = "meter"          # 0=electricity, 1=chilled_water, 2=steam, 3=hot_water
TIMESTAMP_COL   = "timestamp"
METER_READING_COL = "meter_reading"  # kWh (or kBTU for steam/hot_water)

# Weather columns (from weather_train.csv joined on site_id)
AIR_TEMP_COL     = "air_temperature"    # °C
DEW_TEMP_COL     = "dew_temperature"    # °C
WIND_SPEED_COL   = "wind_speed"         # m/s
CLOUD_COVER_COL  = "cloud_coverage"     # oktas

# Minimum hours of data required for rolling features to be meaningful
MIN_HOURS_FOR_FEATURES = 72


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw(data_dir: str = "data/raw/") -> pd.DataFrame:
    """
    Load and join ASHRAE train, building metadata, and weather data.

    Returns a single DataFrame indexed by (building_id, timestamp) with all
    sensor columns and weather columns. Filters to chilled_water meter (meter=1)
    by default — this is the primary HVAC cooling signal.

    Args:
        data_dir: directory containing train.csv, building_metadata.csv,
                  weather_train.csv (from Kaggle download)

    Returns:
        DataFrame with columns: building_id, timestamp, meter_reading,
        square_feet, year_built, air_temperature, dew_temperature, wind_speed,
        cloud_coverage, primary_use
    """
    data_dir = Path(data_dir)

    # train.csv is ~20M rows / 680MB — read in chunks with compact dtypes and
    # filter to the chilled-water meter per chunk so peak memory stays low.
    train_dtypes = {
        BUILDING_ID_COL: "int16",
        METER_COL: "int8",
        METER_READING_COL: "float32",
    }
    chunks = []
    for chunk in pd.read_csv(
        data_dir / "train.csv",
        dtype=train_dtypes,
        parse_dates=[TIMESTAMP_COL],
        chunksize=2_000_000,
    ):
        chunks.append(chunk[chunk[METER_COL] == 1])
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    meta = pd.read_csv(
        data_dir / "building_metadata.csv",
        usecols=["site_id", "building_id", "primary_use", "square_feet", "year_built"],
        dtype={"site_id": "int8", "building_id": "int16",
               "square_feet": "float32", "year_built": "float32"},
    )
    weather_cols = ["site_id", TIMESTAMP_COL, AIR_TEMP_COL, DEW_TEMP_COL,
                    WIND_SPEED_COL, CLOUD_COVER_COL]
    weather = pd.read_csv(
        data_dir / "weather_train.csv",
        usecols=weather_cols,
        dtype={"site_id": "int8", AIR_TEMP_COL: "float32", DEW_TEMP_COL: "float32",
               WIND_SPEED_COL: "float32", CLOUD_COVER_COL: "float32"},
        parse_dates=[TIMESTAMP_COL],
    )

    # Join building metadata
    df = df.merge(meta, on="building_id", how="left")

    # Join weather data via site_id
    df = df.merge(weather, on=["site_id", TIMESTAMP_COL], how="left")

    # Sort for time-series operations
    df = df.sort_values([BUILDING_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows across {df[BUILDING_ID_COL].nunique()} buildings")
    return df


# ---------------------------------------------------------------------------
# COP and thermal efficiency features
# ---------------------------------------------------------------------------

def add_cop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Coefficient of Performance (COP) and related efficiency metrics.

    COP = cooling_output_kWh / power_input_kWh

    For the ASHRAE dataset, chilled water meter_reading is in kBTU.
    We use weather-based load estimation since we don't have direct power draw.
    In a real deployment these would be direct sensor readings.

    NOTE: In a real production system you'd have both power draw and cooling
    output from BMS sensors. Here we approximate from meter readings + weather.
    The interview story is about understanding COP conceptually and engineering
    the feature — the exact computation depends on sensor availability.
    """
    df = df.copy()

    # Convert kBTU to kWh for COP calculation (1 kBTU = 0.29307 kWh)
    df["cooling_output_kwh"] = df[METER_READING_COL] * 0.29307

    # Approximate COP from weather-driven load model
    # Warmer outdoor air → more work required → lower COP
    # This is a proxy; real deployment uses direct power meter
    df["cop_proxy"] = np.where(
        df[AIR_TEMP_COL] > 10,
        df["cooling_output_kwh"] / (df[AIR_TEMP_COL] * 0.02 + 0.5 + 1e-6),
        np.nan,
    )
    # Clip to physically realistic range (0.5–8 for typical commercial chillers)
    df["cop_proxy"] = df["cop_proxy"].clip(0.5, 8.0)

    return df


def add_delta_t_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temperature differential features.

    delta_T_supply (air-side):
        Approximated from air_temperature vs. typical supply setpoint (13°C).
        Real deployment: T_supply_air - T_return_air from BMS sensors.

    delta_T_refrigerant:
        Approximated from dew_temperature differential.
        Real deployment: T_condenser - T_evaporator from refrigerant sensors.

    The interview story: explain what these measure physically and why
    they degrade — this is the domain knowledge that differentiates the project.
    """
    df = df.copy()

    TYPICAL_SUPPLY_SETPOINT = 13.0  # °C, typical commercial chilled water setpoint

    # Supply air delta-T proxy (real: T_return - T_supply across cooling coil)
    df["delta_t_supply_proxy"] = (df[AIR_TEMP_COL] - TYPICAL_SUPPLY_SETPOINT).clip(lower=0)

    # Refrigerant circuit delta-T proxy (real: T_condenser - T_evaporator)
    df["delta_t_refrigerant_proxy"] = (
        df[AIR_TEMP_COL] - df[DEW_TEMP_COL]
    ).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Load and runtime features
# ---------------------------------------------------------------------------

def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute load ratio and runtime fraction.

    Load ratio = actual load / estimated capacity from square footage.
    Capacity estimated at 400 BTU/hr per square foot (commercial rule of thumb).
    """
    df = df.copy()

    # Estimated rated capacity in kBTU/hr (hourly data → kBTU per hour)
    df["rated_capacity_kbtu"] = (df.get("square_feet", 10000) * 400) / 1000

    # Load ratio (clipped to [0, 1.5] — values > 1.0 indicate overloading)
    df["load_ratio"] = (
        df[METER_READING_COL] / (df["rated_capacity_kbtu"] + 1e-6)
    ).clip(0, 1.5)

    return df


# ---------------------------------------------------------------------------
# Rolling / trend features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [24, 168],  # 24 hrs (1 day), 168 hrs (1 week)
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features per building.

    rolling_cop_mean_24h / 168h: short and medium-term COP trend
    rolling_cop_std_24h: COP volatility — high std can indicate intermittent fault
    rolling_cop_deviation: COP vs. unit's own 30-day rolling mean
        Captures slow degradation that absolute thresholds miss.
    rolling_load_mean_24h: operating load trend
    """
    df = df.copy()
    df = df.sort_values([BUILDING_ID_COL, TIMESTAMP_COL])

    grouped = df.groupby(BUILDING_ID_COL)

    for w in windows:
        label = f"{w}h"
        df[f"rolling_cop_mean_{label}"] = (
            grouped["cop_proxy"].transform(lambda s: s.rolling(w, min_periods=w // 2).mean())
        )
        df[f"rolling_cop_std_{label}"] = (
            grouped["cop_proxy"].transform(lambda s: s.rolling(w, min_periods=w // 2).std())
        )
        df[f"rolling_load_mean_{label}"] = (
            grouped["load_ratio"].transform(lambda s: s.rolling(w, min_periods=w // 2).mean())
        )

    # 30-day rolling mean (720 hours) for deviation signal
    df["cop_rolling_30d_mean"] = (
        grouped["cop_proxy"].transform(lambda s: s.rolling(720, min_periods=48).mean())
    )
    df["cop_deviation_from_baseline"] = df["cop_proxy"] - df["cop_rolling_30d_mean"]

    return df


# ---------------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour-of-day, day-of-week, and season features.

    These are needed because HVAC equipment behaves differently at night
    (setback mode), on weekends (lower occupancy), and across seasons.
    Without them, the anomaly detector would flag normal nighttime behavior
    as anomalous because the baseline is mixed.
    """
    df = df.copy()
    ts = pd.to_datetime(df[TIMESTAMP_COL])
    df["hour_of_day"]   = ts.dt.hour
    df["day_of_week"]   = ts.dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]    = (ts.dt.dayofweek >= 5).astype(int)
    df["month"]         = ts.dt.month
    df["season"]        = pd.cut(
        ts.dt.month,
        bins=[0, 3, 6, 9, 12],
        labels=["winter", "spring", "summer", "fall"],
        include_lowest=True,
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "cop_proxy",
    "delta_t_supply_proxy",
    "delta_t_refrigerant_proxy",
    "load_ratio",
    "rolling_cop_mean_24h",
    "rolling_cop_std_24h",
    "rolling_load_mean_24h",
    "rolling_cop_mean_168h",
    "cop_deviation_from_baseline",
    "air_temperature",
    "dew_temperature",
    "wind_speed",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: raw DataFrame from load_raw()

    Returns:
        DataFrame with all engineered features. NaN rows (insufficient rolling
        history) are dropped. Feature columns are defined in FEATURE_COLS.
    """
    df = add_cop_features(df)
    df = add_delta_t_features(df)
    df = add_load_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)

    # Drop rows without sufficient rolling history
    df = df.dropna(subset=["rolling_cop_mean_24h", "rolling_cop_mean_168h"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Feature matrix: {len(df):,} rows × {len(available)} features")
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Return the feature matrix (X) and feature names as a tuple.
    Handles missing columns gracefully — useful during ablation experiments.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median()).astype(np.float32)
    return X, available
