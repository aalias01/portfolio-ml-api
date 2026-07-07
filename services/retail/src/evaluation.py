"""
src/evaluation.py — Backtesting and A/B test utilities.

Backtesting: rolling 12-month training window, predict next 30 days.
A/B testing: two-proportion z-test for policy simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from scipy import stats
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

@dataclass
class BacktestWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_backtest_windows(
    df: pd.DataFrame,
    train_months: int = 12,
    test_days: int = 30,
    step_days: int = 30,
) -> list[BacktestWindow]:
    """Generate rolling backtest window definitions.

    Walk forward across the dataset: each window trains on `train_months`
    of data and predicts the next `test_days`.
    """
    min_date = df["invoice_date"].min()
    max_date = df["invoice_date"].max()
    windows = []
    test_start = min_date + pd.DateOffset(months=train_months)
    while test_start + pd.Timedelta(days=test_days) <= max_date:
        train_start = test_start - pd.DateOffset(months=train_months)
        windows.append(
            BacktestWindow(
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + pd.Timedelta(days=test_days),
            )
        )
        test_start += pd.Timedelta(days=step_days)
    return windows


def run_backtest(
    df: pd.DataFrame,
    windows: list[BacktestWindow],
    feature_fn: Callable,
    train_fn: Callable,
    predict_fn: Callable,
) -> pd.DataFrame:
    """Execute rolling-window backtest.

    Args:
        feature_fn: (df, as_of_date) → (X, y)
        train_fn: (X_train, y_train) → model
        predict_fn: (model, X_test) → y_prob

    Returns a DataFrame with one row per window:
        window_start, window_end, brier, precision_top_decile, mae_return_rate
    """
    results = []
    for w in windows:
        train_df = df[
            (df["invoice_date"] >= w.train_start) & (df["invoice_date"] < w.train_end)
        ]
        test_df = df[
            (df["invoice_date"] >= w.test_start) & (df["invoice_date"] < w.test_end)
        ]
        if len(train_df) < 100 or len(test_df) < 10:
            continue

        X_train, y_train = feature_fn(train_df, as_of_date=w.train_end)
        X_test, y_test = feature_fn(test_df, as_of_date=w.test_start)

        model = train_fn(X_train, y_train)
        y_prob = predict_fn(model, X_test)

        brier = brier_score_loss(y_test, y_prob)
        top_n = max(1, int(len(y_test) * 0.10))
        top_idx = np.argsort(y_prob)[::-1][:top_n]
        prec_top_decile = float(np.array(y_test)[top_idx].mean())

        actual_rate = y_test.mean()
        predicted_rate = y_prob.mean()
        mae_rate = abs(predicted_rate - actual_rate)

        results.append(
            {
                "window_start": w.train_start,
                "window_end": w.test_end,
                "brier_score": brier,
                "precision_top_decile": prec_top_decile,
                "mae_return_rate": mae_rate,
                "actual_return_rate": actual_rate,
                "predicted_return_rate": predicted_rate,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# A/B Test Simulation
# ---------------------------------------------------------------------------

@dataclass
class ABTestResult:
    control_rate: float
    treatment_rate: float
    lift: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    n_control: int
    n_treatment: int
    min_detectable_effect: float


def power_analysis(
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    relative_lift: float = 0.15,
) -> int:
    """Compute minimum sample size per arm for a two-proportion z-test.

    Args:
        baseline_rate: expected control group return rate
        relative_lift: minimum detectable relative effect (e.g. 0.15 = 15% reduction)
        alpha: significance level
        power: desired statistical power

    Returns required n per arm.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 - relative_lift)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = (
        (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p1 - p2) ** 2
    )
    return int(np.ceil(n))


def simulate_ab_test(
    returner_df: pd.DataFrame,
    policy_window_days: int = 14,
    alpha: float = 0.05,
    random_state: int = 42,
) -> ABTestResult:
    """Simulate a 14-day vs. 30-day return policy test on the Returner segment.

    Methodology:
      - Randomly split Returner segment customers 50/50 (control vs. treatment)
      - Control: all observed returns count (30-day policy simulation)
      - Treatment: returns after day 14 from purchase are "rejected" → counted
        as retained revenue
      - Primary metric: return_value_ratio
      - Guardrail: total_spend (must not drop significantly)

    Args:
        returner_df: customer-level DataFrame filtered to Returner segment,
            with columns: customer_id, total_return_value, total_revenue,
            avg_days_to_return (computed in notebook)
    """
    rng = np.random.default_rng(random_state)
    n = len(returner_df)
    assignment = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    returner_df = returner_df.copy()
    returner_df["group"] = assignment

    control = returner_df[returner_df["group"] == "control"]
    treatment = returner_df[returner_df["group"] == "treatment"].copy()

    # Simulate treatment: reject returns that occurred after day 14
    # avg_days_to_return acts as a proxy — customers with avg > 14 lose their return
    treatment["effective_return_value"] = np.where(
        treatment["avg_days_to_return"] > policy_window_days,
        0.0,
        treatment["total_return_value"],
    )

    control_rate = (
        (control["total_return_value"] / control["total_revenue"].clip(lower=1)).mean()
    )
    treatment_rate = (
        (treatment["effective_return_value"] / treatment["total_revenue"].clip(lower=1)).mean()
    )

    n_control = len(control)
    n_treatment = len(treatment)
    lift = (control_rate - treatment_rate) / control_rate if control_rate > 0 else 0.0

    # Two-proportion z-test (binarized: customer is a returner if return_value_ratio > 0)
    control_returns = (control["total_return_value"] > 0).sum()
    treatment_returns = (treatment["effective_return_value"] > 0).sum()

    count = np.array([control_returns, treatment_returns])
    nobs = np.array([n_control, n_treatment])

    from statsmodels.stats.proportion import proportions_ztest, proportion_confint
    stat, p_value = proportions_ztest(count, nobs, alternative="larger")

    ci_lower, ci_upper = proportion_confint(control_returns, n_control, alpha=alpha)
    mde = power_analysis(control_rate, alpha=alpha)

    return ABTestResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        lift=lift,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        n_control=n_control,
        n_treatment=n_treatment,
        min_detectable_effect=float(mde),
    )
