"""
src/features.py — Feature engineering for Industrial Failure Classification.

Dataset: AI4I 2020 Predictive Maintenance (UCI)
Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Raw features:
    Air temperature [K], Process temperature [K], Rotational speed [rpm],
    Torque [Nm], Tool wear [min]

Engineered features (domain-driven):
    temp_diff       — Process temp - Air temp (heat buildup indicator)
    power           — Rotational speed × Torque (mechanical power = stress proxy)
    wear_rate       — Tool wear / Rotational speed (wear relative to speed)
    torque_per_wear — Torque / (Tool wear + 1) (stress per unit of accumulated wear)
    high_load       — Binary flag: power in top 25%
"""

from __future__ import annotations
import io
import zipfile
from pathlib import Path
import pandas as pd
import requests

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/ai4i2020.csv"

RAW_FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURE_COLS = RAW_FEATURE_COLS + [
    "temp_diff",
    "power",
    "wear_rate",
    "torque_per_wear",
    "high_load",
]


def download_data(dest: Path = RAW_PATH) -> Path:
    """Download AI4I 2020 dataset from UCI if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[features] Already exists: {dest}")
        return dest
    print("[features] Downloading from UCI...")
    resp = requests.get(UCI_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with zf.open(csv_names[0]) as f:
            dest.write_bytes(f.read())
    print(f"[features] Saved: {dest}")
    return dest


def load_raw(csv_path: str = str(RAW_PATH)) -> pd.DataFrame:
    """Load and lightly clean the AI4I dataset."""
    df = pd.read_csv(csv_path)
    # Normalize column names (strip whitespace, consistent casing)
    df.columns = df.columns.str.strip()
    # Keep only relevant columns
    cols = RAW_FEATURE_COLS + [TARGET_COL] + FAILURE_MODE_COLS + ["Type", "Product ID"]
    existing = [c for c in cols if c in df.columns]
    df = df[existing].copy()
    print(f"[features] Loaded {len(df):,} rows | Failure rate: {df[TARGET_COL].mean()*100:.1f}%")
    return df


def build_features(
    df: pd.DataFrame,
    high_load_cutoff: float | None = None,
) -> pd.DataFrame:
    """Add domain-engineered features.

    Args:
        df: Raw dataframe with the AI4I columns.
        high_load_cutoff: Fixed cutoff for the binary `high_load` feature. If
            None, the 75th percentile of `power` in `df` is used (training-time
            behavior). For inference on a single row, pass the cutoff persisted
            in `models/model_meta.json` so the feature matches what was learned.

    Returns:
        Dataframe with engineered features appended.
    """
    df = df.copy()
    df["temp_diff"]      = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"]          = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["wear_rate"]      = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)
    df["torque_per_wear"]= df["Torque [Nm]"] / (df["Tool wear [min]"] + 1)
    cutoff = float(high_load_cutoff) if high_load_cutoff is not None else float(df["power"].quantile(0.75))
    df["high_load"]      = (df["power"] >= cutoff).astype(int)
    df.attrs["high_load_cutoff"] = cutoff
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and binary target."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET_COL]
    return X, y
