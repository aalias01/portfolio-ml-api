"""
src/features.py - Feature engineering pipeline for Retail Returns Intelligence.

Two variants are implemented here:
  - Pandas (fast iteration, notebook-friendly)
  - PySpark stubs (mirrors Pandas logic; full implementation in notebooks/09_pyspark_pipeline.ipynb)

Point-in-time safety rule: all customer-level features must be computed using only
transactions with InvoiceDate < current transaction's InvoiceDate to prevent leakage.
"""

from __future__ import annotations

import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Load UCI Online Retail II .xlsx file.

    Returns a cleaned DataFrame with standardized column names and an
    is_return flag derived from the C-prefix InvoiceNo convention.
    """
    df = pd.read_excel(path, sheet_name=None)
    # UCI II has two sheets (Year 2009-2010, Year 2010-2011). Concat them.
    df = pd.concat(df.values(), ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Standardize key column names
    rename = {
        "invoice": "invoice_no",       # UCI II uses "Invoice"
        "stockcode": "stock_code",
        "invoicedate": "invoice_date",
        "price": "unit_price",          # UCI II uses "Price" (not "UnitPrice")
        "unitprice": "unit_price",      # fallback for older UCI I format
        "customerid": "customer_id",
        "customer id": "customer_id",   # after normalize: "Customer ID" → "customer id"
        "invoiceno": "invoice_no",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_return"] = df["invoice_no"].astype(str).str.startswith("C").astype(int)
    # Ensure key identifier columns are always strings (UCI II sometimes loads StockCode as int)
    for col in ("invoice_no", "stock_code", "customer_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Restore NaN for customer_id (str conversion turns NaN → 'nan')
    df["customer_id"] = df["customer_id"].replace("nan", pd.NA)
    return df


# ---------------------------------------------------------------------------
# Transaction-level features
# ---------------------------------------------------------------------------

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-level features.

    Features engineered:
      - unit_price_z: z-score of unit_price within stock_code
      - quantity_z: z-score of quantity within customer_id history
      - is_weekend: 1 if invoice_date falls on Saturday or Sunday
      - month_end_proximity: days from end of month (wardrobing signal)
      - country: categorical (kept as-is; encode downstream)
      - revenue: quantity * unit_price, signed negative for returns
    """
    # Z-score unit_price within each product category
    price_stats = df.groupby("stock_code")["unit_price"].transform
    df["unit_price_z"] = (
        (df["unit_price"] - price_stats("mean")) / price_stats("std").clip(lower=1e-6)
    )

    # Z-score quantity within customer history
    qty_stats = df.groupby("customer_id")["quantity"].transform
    df["quantity_z"] = (
        (df["quantity"] - qty_stats("mean")) / qty_stats("std").clip(lower=1e-6)
    )

    df["is_weekend"] = df["invoice_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_month"] = df["invoice_date"].dt.day
    df["days_in_month"] = df["invoice_date"].dt.days_in_month
    df["month_end_proximity"] = df["days_in_month"] - df["day_of_month"]

    df["revenue"] = df["quantity"] * df["unit_price"]
    return df


# ---------------------------------------------------------------------------
# Customer-level behavioral features (point-in-time safe)
# ---------------------------------------------------------------------------

def build_customer_features(df: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute customer-level behavioral features.

    If as_of_date is provided, only transactions before that date are used
    (point-in-time safe for backtesting). If None, uses the full DataFrame.

    Returns one row per customer_id.
    """
    if as_of_date is not None:
        hist = df[df["invoice_date"] < as_of_date].copy()
    else:
        hist = df.copy()

    hist = hist.dropna(subset=["customer_id"])
    purchases = hist[hist["is_return"] == 0]
    returns = hist[hist["is_return"] == 1]

    cust = purchases.groupby("customer_id").agg(
        total_orders=("invoice_no", "nunique"),
        total_revenue=("revenue", "sum"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        unique_categories_purchased=("stock_code", "nunique"),
    ).reset_index()

    ret_agg = returns.groupby("customer_id").agg(
        total_return_orders=("invoice_no", "nunique"),
        total_return_value=("revenue", lambda x: abs(x.sum())),
        unique_categories_returned=("stock_code", "nunique"),
        n_weekend_returns=("is_weekend", "sum"),
    ).reset_index()

    cust = cust.merge(ret_agg, on="customer_id", how="left").fillna(0)

    cust["lifetime_return_rate"] = (
        cust["total_return_orders"] / cust["total_orders"].clip(lower=1)
    )
    cust["return_value_ratio"] = (
        cust["total_return_value"] / cust["total_revenue"].clip(lower=1)
    )
    cust["weekend_return_share"] = (
        cust["n_weekend_returns"] / cust["total_return_orders"].clip(lower=1)
    )

    now = as_of_date or hist["invoice_date"].max()
    cust["tenure_days"] = (now - cust["first_purchase"]).dt.days
    cust["recency_score"] = (now - cust["last_purchase"]).dt.days

    # RFM: recency (lower = better), frequency, monetary
    cust["frequency_score"] = cust["total_orders"]
    cust["monetary_score"] = cust["total_revenue"]

    return cust


def add_return_velocity(
    df: pd.DataFrame, customer_features: pd.DataFrame, window_days: int = 30
) -> pd.DataFrame:
    """Add 30-day return velocity to customer_features.

    Requires df to have invoice_date and is_return columns.
    """
    cutoff = df["invoice_date"].max()
    window_start = cutoff - pd.Timedelta(days=window_days)
    recent_returns = df[
        (df["is_return"] == 1) & (df["invoice_date"] >= window_start)
    ]
    velocity = (
        recent_returns.groupby("customer_id")["invoice_no"]
        .nunique()
        .reset_index()
        .rename(columns={"invoice_no": "return_velocity"})
    )
    return customer_features.merge(velocity, on="customer_id", how="left").fillna(
        {"return_velocity": 0}
    )


# ---------------------------------------------------------------------------
# Category return rate (used as transaction-level lookup feature)
# ---------------------------------------------------------------------------

def build_category_return_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical return rate per stock_code.

    Returns a DataFrame with (stock_code, category_return_rate).
    """
    agg = df.groupby("stock_code").agg(
        total=("invoice_no", "count"),
        returns=("is_return", "sum"),
    )
    agg["category_return_rate"] = agg["returns"] / agg["total"].clip(lower=1)
    return agg[["category_return_rate"]].reset_index()


# ---------------------------------------------------------------------------
# Final feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    customer_features: pd.DataFrame,
    category_rates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble the final feature matrix for Model 1 (classifier).

    Returns (X, y) where y is the is_return binary label.
    """
    df = df.merge(
        customer_features[
            [
                "customer_id",
                "lifetime_return_rate",
                "return_value_ratio",
                "return_velocity",
                "tenure_days",
                "recency_score",
                "frequency_score",
                "monetary_score",
                "unique_categories_returned",
                "weekend_return_share",
            ]
        ],
        on="customer_id",
        how="left",
    )
    df = df.merge(category_rates, on="stock_code", how="left")

    feature_cols = [
        "unit_price_z",
        "quantity_z",
        "is_weekend",
        "month_end_proximity",
        "lifetime_return_rate",
        "return_value_ratio",
        "return_velocity",
        "tenure_days",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "unique_categories_returned",
        "weekend_return_share",
        "category_return_rate",
    ]
    X = df[feature_cols].fillna(0)
    y = df["is_return"]
    return X, y
