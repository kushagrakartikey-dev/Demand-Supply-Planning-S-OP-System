"""
Metric library.

Every KPI used anywhere in the system is implemented exactly once, here, with
its formula documented in the docstring.  METHODOLOGY.md is written directly
against these functions so the documentation cannot drift from the code.

Two families of metrics live in this module:

1. Forecast accuracy   -- MAPE, sMAPE, WAPE, MAE, RMSE, Bias, Tracking Signal,
                          Forecast Value Added.
2. Supply-chain KPIs   -- safety stock, reorder point, weeks of cover, OTIF,
                          capacity utilisation, working capital, EOQ,
                          inventory turns.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "mape", "smape", "wape", "mae", "rmse", "bias", "bias_pct", "tracking_signal",
    "forecast_value_added", "accuracy_bundle", "inverse_error_weights",
    "safety_stock", "reorder_point", "weeks_of_cover", "capacity_utilisation",
    "working_capital", "economic_order_quantity", "inventory_turns",
    "otif_rate", "control_limits", "service_level_z", "coefficient_of_variation",
    "abc_classification", "xyz_classification",
]

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Forecast accuracy
# --------------------------------------------------------------------------- #

def _clean_pair(actual: Sequence[float], forecast: Sequence[float]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Align two series, drop non-finite pairs and return float arrays."""
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if a.shape != f.shape:
        raise ValueError(f"actual/forecast length mismatch: {a.shape} vs {f.shape}")
    mask = np.isfinite(a) & np.isfinite(f)
    return a[mask], f[mask]


def mape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """
    Mean Absolute Percentage Error (%).

        MAPE = 100 / n * sum( |A_t - F_t| / |A_t| )   for A_t != 0

    Periods with zero actual demand are excluded because the ratio is
    undefined there -- this is the standard demand-planning treatment and the
    reason WAPE is also reported alongside.
    """
    a, f = _clean_pair(actual, forecast)
    mask = np.abs(a) > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100.0)


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """
    Symmetric MAPE (%): 100/n * sum( 2|A-F| / (|A| + |F|) ).

    Bounded at 200%, defined when the actual is zero, so it is used as the
    tie-breaker for intermittent-demand SKUs.
    """
    a, f = _clean_pair(actual, forecast)
    denom = np.abs(a) + np.abs(f)
    mask = denom > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(a[mask] - f[mask]) / denom[mask]) * 100.0)


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """
    Weighted Absolute Percentage Error (%), a.k.a. MAD/Mean ratio.

        WAPE = 100 * sum|A_t - F_t| / sum|A_t|

    Volume-weighted, defined at zero demand, and the metric most FMCG
    businesses actually steer on because it aggregates cleanly up a hierarchy.
    """
    a, f = _clean_pair(actual, forecast)
    total = np.sum(np.abs(a))
    if total < _EPS:
        return float("nan")
    return float(np.sum(np.abs(a - f)) / total * 100.0)


def mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean Absolute Error in units: 1/n * sum|A_t - F_t|."""
    a, f = _clean_pair(actual, forecast)
    return float(np.mean(np.abs(a - f))) if a.size else float("nan")


def rmse(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Root Mean Squared Error in units: sqrt(1/n * sum (A_t - F_t)^2)."""
    a, f = _clean_pair(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2))) if a.size else float("nan")


def bias(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """
    Mean forecast bias in units: 1/n * sum(F_t - A_t).

    Positive => systematic over-forecasting (excess stock risk).
    Negative => systematic under-forecasting (stockout risk).
    """
    a, f = _clean_pair(actual, forecast)
    return float(np.mean(f - a)) if a.size else float("nan")


def bias_pct(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Bias as a percentage of total actual demand: 100*sum(F-A)/sum(A)."""
    a, f = _clean_pair(actual, forecast)
    total = np.sum(a)
    if abs(total) < _EPS:
        return float("nan")
    return float(np.sum(f - a) / total * 100.0)


def tracking_signal(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """
    Tracking Signal = cumulative bias / MAE.

    |TS| > 4 is the classic Plossl-Wight trigger indicating the model has
    drifted and needs re-parameterisation rather than tuning.
    """
    a, f = _clean_pair(actual, forecast)
    error_mad = np.mean(np.abs(a - f))
    if error_mad < _EPS:
        return 0.0
    return float(np.sum(f - a) / error_mad)


def forecast_value_added(actual: Sequence[float],
                         forecast: Sequence[float],
                         naive: Sequence[float]) -> float:
    """
    Forecast Value Added (percentage points of MAPE).

        FVA = MAPE(naive) - MAPE(model)

    Positive FVA means the modelling effort beat the do-nothing benchmark.
    A negative FVA is a legitimate and important finding: it says the process
    step is destroying value and should be removed.
    """
    naive_error = mape(actual, naive)
    model_error = mape(actual, forecast)
    if np.isnan(naive_error) or np.isnan(model_error):
        return float("nan")
    return float(naive_error - model_error)


def accuracy_bundle(actual: Sequence[float],
                    forecast: Sequence[float],
                    naive: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Compute the full accuracy panel in one pass."""
    result = {
        "MAPE": mape(actual, forecast),
        "sMAPE": smape(actual, forecast),
        "WAPE": wape(actual, forecast),
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "Bias": bias(actual, forecast),
        "Bias_Pct": bias_pct(actual, forecast),
        "Tracking_Signal": tracking_signal(actual, forecast),
        "N_Obs": int(len(_clean_pair(actual, forecast)[0])),
    }
    result["FVA_vs_Naive"] = (
        forecast_value_added(actual, forecast, naive) if naive is not None else float("nan")
    )
    return result


def inverse_error_weights(errors: Dict[str, float],
                          floor: float = 1.0,
                          min_weight: float = 0.0) -> Dict[str, float]:
    """
    Inverse-error ensemble weights.

        w_i = (1 / e_i) / sum_j (1 / e_j)

    `floor` prevents an implausibly small error from monopolising the blend
    (and guards against division by zero).  Models whose weight falls below
    `min_weight` are dropped and the remainder renormalised, which keeps the
    ensemble sparse and explainable to a planner.
    """
    usable = {k: max(float(v), floor) for k, v in errors.items()
              if v is not None and np.isfinite(v)}
    if not usable:
        return {}

    inverse = {k: 1.0 / v for k, v in usable.items()}
    total = sum(inverse.values())
    weights = {k: v / total for k, v in inverse.items()}

    if min_weight > 0:
        kept = {k: v for k, v in weights.items() if v >= min_weight}
        if kept:
            total_kept = sum(kept.values())
            weights = {k: v / total_kept for k, v in kept.items()}
    return weights


# --------------------------------------------------------------------------- #
# Inventory / supply metrics
# --------------------------------------------------------------------------- #

def service_level_z(service_level: float) -> float:
    """Inverse normal CDF: the safety factor Z for a target cycle service level."""
    service_level = min(max(float(service_level), 0.5), 0.9999)
    return float(stats.norm.ppf(service_level))


def safety_stock(demand_std: float,
                 lead_time_weeks: float,
                 service_level: float = 0.95,
                 avg_demand: float = 0.0,
                 lead_time_std_weeks: float = 0.0) -> float:
    """
    Safety stock accounting for BOTH demand and lead-time variability:

        SS = Z * sqrt( LT * sigma_D^2  +  mu_D^2 * sigma_LT^2 )

    Dropping the second term (the usual spreadsheet shortcut) understates the
    buffer whenever suppliers are unreliable, which is precisely when the
    buffer matters. Units are demand units per period, LT in the same periods.
    """
    z = service_level_z(service_level)
    variance = (max(lead_time_weeks, 0.0) * max(demand_std, 0.0) ** 2
                + max(avg_demand, 0.0) ** 2 * max(lead_time_std_weeks, 0.0) ** 2)
    return float(max(z * np.sqrt(max(variance, 0.0)), 0.0))


def reorder_point(avg_demand: float, lead_time_weeks: float, ss: float) -> float:
    """Reorder Point = mu_D * LT + SS (continuous review, (s,Q) policy)."""
    return float(max(avg_demand, 0.0) * max(lead_time_weeks, 0.0) + max(ss, 0.0))


def weeks_of_cover(closing_inventory: float, forward_weekly_demand: float) -> float:
    """
    Weeks of Cover = closing inventory / average forward weekly demand.

    Capped at 52 so that near-zero demand on a discontinued SKU does not blow
    up dashboards and pull category averages to infinity.
    """
    if forward_weekly_demand is None or forward_weekly_demand <= _EPS:
        return 52.0 if closing_inventory > 0 else 0.0
    return float(min(max(closing_inventory, 0.0) / forward_weekly_demand, 52.0))


def capacity_utilisation(required: float, available: float) -> float:
    """Capacity Utilisation (%) = required / available * 100."""
    if available is None or available <= _EPS:
        return 0.0 if required <= _EPS else 999.0
    return float(required / available * 100.0)


def working_capital(inventory_units: float, unit_cost: float) -> float:
    """Working capital tied in stock = units on hand * standard unit cost."""
    return float(max(inventory_units, 0.0) * max(unit_cost, 0.0))


def economic_order_quantity(annual_demand: float,
                            ordering_cost: float,
                            unit_cost: float,
                            holding_rate: float) -> float:
    """Classical EOQ: Q* = sqrt(2 D S / H) with H = holding_rate * unit_cost."""
    holding = max(holding_rate * unit_cost, _EPS)
    if annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2.0 * annual_demand * max(ordering_cost, 0.0) / holding))


def inventory_turns(annual_cogs: float, average_inventory_value: float) -> float:
    """Inventory turns = annual COGS / average inventory value."""
    if average_inventory_value <= _EPS:
        return 0.0
    return float(annual_cogs / average_inventory_value)


def coefficient_of_variation(series: Sequence[float]) -> float:
    """CV = sigma / mu, the demand-volatility input to XYZ classification."""
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or abs(arr.mean()) < _EPS:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean()) if arr.size > 1 else 0.0


def abc_classification(values: pd.Series, thresholds: tuple[float, float] = (0.70, 0.90)
                       ) -> pd.Series:
    """
    Pareto ABC classification on cumulative contribution.

    A = items making up the first 70% of value, B the next 20%, C the tail.
    """
    ordered = values.sort_values(ascending=False)
    total = ordered.sum()
    if total <= _EPS:
        return pd.Series("C", index=values.index)
    cumulative = ordered.cumsum() / total
    labels = pd.Series(
        np.where(cumulative <= thresholds[0], "A",
                 np.where(cumulative <= thresholds[1], "B", "C")),
        index=ordered.index,
    )
    return labels.reindex(values.index)


def xyz_classification(cv_values: pd.Series) -> pd.Series:
    """XYZ by demand variability: X = CV<0.5, Y = 0.5-1.0, Z = CV>1.0."""
    return pd.Series(
        np.where(cv_values < 0.5, "X", np.where(cv_values <= 1.0, "Y", "Z")),
        index=cv_values.index,
    )


# --------------------------------------------------------------------------- #
# Service level / SPC
# --------------------------------------------------------------------------- #

def otif_rate(on_time: Sequence[bool], in_full: Sequence[bool]) -> float:
    """
    OTIF (%) = orders delivered On Time AND In Full / total eligible orders.

    The conjunction is deliberate: a line that is early but short, or complete
    but late, is a service failure from the customer's point of view.
    """
    ot = np.asarray(on_time, dtype=bool)
    inf = np.asarray(in_full, dtype=bool)
    if ot.size == 0:
        return float("nan")
    return float(np.mean(ot & inf) * 100.0)


def control_limits(series: Sequence[float],
                   sigma_multiplier: float = 3.0) -> Dict[str, float]:
    """
    Shewhart individuals (I-chart) control limits.

    Sigma is estimated from the average moving range rather than the raw
    standard deviation:

        sigma_hat = MRbar / d2,   d2 = 1.128 for a moving range of n = 2

    This is the Six Sigma convention: it isolates short-term common-cause
    variation and does not get inflated by the very special causes the chart
    exists to detect.  A fallback to the sample standard deviation is used
    when there are too few points for a moving range.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "sigma": float("nan"),
                "UCL": float("nan"), "LCL": float("nan"), "n": 0}

    centre = float(arr.mean())
    if arr.size >= 3:
        moving_range = np.abs(np.diff(arr))
        sigma = float(moving_range.mean() / 1.128) if moving_range.size else 0.0
        if sigma <= _EPS:
            sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    else:
        sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    return {
        "mean": centre,
        "sigma": sigma,
        "UCL": centre + sigma_multiplier * sigma,
        "LCL": centre - sigma_multiplier * sigma,
        "n": int(arr.size),
    }
