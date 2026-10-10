"""
Module 1 -- AI/ML Demand Forecasting Engine.

Four competing forecasters, an inverse-error ensemble, honest out-of-sample
evaluation and calibrated prediction intervals.

Evaluation design (the part that matters)
-----------------------------------------
The series is cut into three disjoint blocks::

    [ ------------ training ------------ | -- CV folds -- | -- holdout -- ]

* **CV folds** (rolling origin, expanding window) drive model *selection* and
  ensemble *weighting*.
* **Holdout** (final 12 weeks) is touched exactly once, after every decision
  has been frozen. Its scores are therefore an unbiased estimate of what the
  system would have delivered in production.
* **Final forecast** refits every selected model on the full history.

This ordering is the whole point. Selecting the best model on the same window
you then report accuracy from is the single most common way portfolio
forecasting projects overstate their results, and it is structurally
impossible here.

Model tiers
-----------
SARIMA and Prophet cost 2-5 seconds per fit, which does not scale to 500 SKUs
on commodity hardware. The engine therefore runs:

* a **mass tier** (Holt-Winters + global XGBoost) across the entire portfolio,
* a **competition panel** (all four models) on a revenue-stratified sample.

The panel answers "does the expensive model earn its cost?"; the mass tier
delivers a forecast for every SKU. This is exactly how planning teams actually
deploy heavy models, and the panel results are reported as such rather than
extrapolated to the portfolio.

Target censoring
----------------
Models are trained on ``Sales_Units`` (censored, what the ERP records), not on
latent ``Demand_Units``. Accuracy is measured against the same censored series.
Training on the unobservable truth would inflate every metric in this report.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, ForecastConfig, Keys, OUTPUT_DIR
from src.shared.metrics import (accuracy_bundle, inverse_error_weights, mape,
                                service_level_z)
from src.shared.utils import get_logger, save_csv, timed
from src.shared.validation import (assert_no_leakage, holdout_split,
                                   rolling_origin_splits)

warnings.filterwarnings("ignore")
logger = get_logger("demand_engine")

# --------------------------------------------------------------------------- #
# Optional dependency probing -- every heavy model has a working fallback.
# --------------------------------------------------------------------------- #
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:                                        # pragma: no cover
    HAS_STATSMODELS = False

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:                                        # pragma: no cover
    HAS_PMDARIMA = False

try:
    import logging as _logging
    from prophet import Prophet
    # Prophet re-attaches its own handlers on import; silence them here rather
    # than in the logging config so the suppression survives the import order.
    for _name in ("prophet", "prophet.models", "prophet.forecaster", "cmdstanpy"):
        _logging.getLogger(_name).setLevel(_logging.CRITICAL)
        _logging.getLogger(_name).propagate = False
    HAS_PROPHET = True
except ImportError:                                        # pragma: no cover
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:                                        # pragma: no cover
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:                                        # pragma: no cover
    HAS_SHAP = False


def dependency_report() -> Dict[str, bool]:
    """Which optional engines are live in this environment."""
    return {
        "statsmodels": HAS_STATSMODELS,
        "pmdarima": HAS_PMDARIMA,
        "prophet": HAS_PROPHET,
        "xgboost": HAS_XGBOOST,
        "shap": HAS_SHAP,
    }


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class ForecastResult:
    """Everything the demand engine hands to the rest of the pipeline."""

    forecast: pd.DataFrame              # forward 12-week forecast + intervals
    holdout_predictions: pd.DataFrame   # per-model holdout predictions
    model_scores: pd.DataFrame          # per SKU x model x stage metrics
    ensemble_weights: pd.DataFrame      # per SKU x model weight
    sku_accuracy: pd.DataFrame          # per SKU headline accuracy
    category_accuracy: pd.DataFrame     # per category headline accuracy
    hierarchy_accuracy: pd.DataFrame    # accuracy by aggregation level
    model_leaderboard: pd.DataFrame     # aggregate model comparison
    feature_importance: pd.DataFrame
    shap_summary: pd.DataFrame
    uncertainty: pd.DataFrame           # Output 5 hand-off contract
    diagnostics: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Feature engineering for the global ML model
# --------------------------------------------------------------------------- #

class DirectMultiHorizonFeatureBuilder:
    """
    Builds the design matrix for the global XGBoost forecaster.

    A *direct multi-horizon* formulation is used: each training row pairs
    features observable at an origin week ``t`` with the target at ``t + h``,
    and the horizon ``h`` is itself a feature. Compared with recursive
    forecasting this removes error compounding and, more importantly for a
    portfolio project, removes an entire class of subtle leakage bugs -- there
    is no step where a predicted value is fed back in as if it were observed.

    Two feature families:

    * **Backward-looking** (lags, rolling statistics, EWMA) computed strictly
      at or before the origin ``t``.
    * **Forward-known** (calendar, Fourier terms, festival intensity, planned
      promotion, planned price index) evaluated at the *target* week ``t + h``.
      These are legitimately known in advance -- an S&OP process owns the promo
      grid and the price plan a quarter out. Using them is not leakage; it is
      the reason a planning-aware model beats a pure time-series one.
    """

    def __init__(self, cfg: ForecastConfig) -> None:
        self.cfg = cfg
        self.feature_names: List[str] = []

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _fourier(week_of_year: np.ndarray, order: int, period: float = 52.18
                 ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in range(1, order + 1):
            angle = 2.0 * np.pi * k * week_of_year / period
            out[f"fourier_sin_{k}"] = np.sin(angle)
            out[f"fourier_cos_{k}"] = np.cos(angle)
        return out

    def build_panel(self, history: pd.DataFrame, target_col: str = "Sales_Units"
                    ) -> pd.DataFrame:
        """
        Attach backward-looking features to every (SKU, week) observation.

        Every operation is a `shift`/`rolling` on a SKU-sorted frame, so the
        value at row t only ever sees rows <= t.
        """
        cfg = self.cfg
        panel = history.sort_values([Keys.SKU, Keys.WEEK_INDEX]).copy()
        grouped = panel.groupby(Keys.SKU, sort=False)[target_col]

        # The most recent observation available at the origin. Named
        # explicitly rather than smuggled in as "lag_0", because at prediction
        # time this is the last actual the planner has seen and it is by far
        # the strongest single predictor.
        panel["y_at_origin"] = grouped.shift(0)

        # True lags relative to the origin: lag_k is the observation k weeks
        # BEFORE the origin. Keeping the lag index honest means a causality
        # assertion on the panel is meaningful rather than trivially violated.
        for lag in cfg.xgb_lags:
            panel[f"lag_{lag}"] = grouped.shift(lag)

        for window in cfg.xgb_rolling_windows:
            panel[f"roll_mean_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(w, min_periods=2).mean())
            panel[f"roll_std_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(w, min_periods=2).std())

        for span in cfg.xgb_ewm_spans:
            panel[f"ewm_{span}"] = grouped.transform(
                lambda s, sp=span: s.ewm(span=sp, min_periods=2).mean())

        # Momentum and volatility ratios: scale-free, so one global model can
        # serve a 30-unit-a-week SKU and a 30,000-unit-a-week SKU.
        panel["momentum_4_13"] = panel["roll_mean_4"] / panel["roll_mean_13"].replace(0, np.nan)
        panel["momentum_4_26"] = panel["roll_mean_4"] / panel["roll_mean_26"].replace(0, np.nan)
        panel["cv_13"] = panel["roll_std_13"] / panel["roll_mean_13"].replace(0, np.nan)
        panel["level"] = panel["roll_mean_13"]
        panel["weeks_since_launch"] = grouped.cumcount()
        panel["promo_density_13"] = (panel.groupby(Keys.SKU, sort=False)["Is_Promo"]
                                     .transform(lambda s: s.rolling(13, min_periods=2).mean()))
        panel["stockout_density_13"] = (panel.groupby(Keys.SKU, sort=False)["Is_Stockout"]
                                        .transform(lambda s: s.rolling(13, min_periods=2).mean()))
        return panel

    def build_training_matrix(self, panel: pd.DataFrame, origin_max: int,
                              horizon: int, target_col: str = "Sales_Units"
                              ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Expand the panel into (origin, horizon) training pairs.

        Only pairs whose *target* week index is <= ``origin_max`` are emitted,
        which is what keeps the training set strictly inside the cutoff.
        """
        frames: List[pd.DataFrame] = []
        backward_cols = [c for c in panel.columns if c.startswith(
            ("lag_", "roll_", "ewm_", "momentum_", "cv_"))] + [
            "y_at_origin", "level", "weeks_since_launch",
            "promo_density_13", "stockout_density_13"]

        base = panel[[Keys.SKU, Keys.WEEK_INDEX, Keys.CATEGORY, Keys.CHANNEL,
                      "Unit_Price"] + backward_cols].copy()

        forward = panel[[Keys.SKU, Keys.WEEK_INDEX, "ISO_Week", "Month",
                         "Is_Promo", "Promo_Depth", "Price_Index",
                         "Festival_Intensity", target_col]].copy()

        for h in range(1, horizon + 1):
            fwd = forward.copy()
            fwd[Keys.WEEK_INDEX] = fwd[Keys.WEEK_INDEX] - h   # align to origin
            merged = base.merge(fwd, on=[Keys.SKU, Keys.WEEK_INDEX],
                                how="inner", suffixes=("", "_tgt"))
            merged["horizon"] = h
            merged["target_week_index"] = merged[Keys.WEEK_INDEX] + h
            merged = merged[merged["target_week_index"] <= origin_max]
            frames.append(merged)

        if not frames:
            return pd.DataFrame(), pd.Series(dtype=float)

        stacked = pd.concat(frames, ignore_index=True)
        stacked = stacked.dropna(subset=[target_col, "y_at_origin"])
        y = stacked[target_col].astype(float)
        X = self._encode(stacked)
        return X, y

    def build_prediction_matrix(self, panel: pd.DataFrame, origin_index: int,
                                future_exog: pd.DataFrame, horizon: int
                                ) -> pd.DataFrame:
        """
        Assemble the design matrix for forecasting forward from ``origin_index``.

        Backward features are taken from the origin row; forward-known features
        come from the supplied planning calendar (``future_exog``).
        """
        backward_cols = [c for c in panel.columns if c.startswith(
            ("lag_", "roll_", "ewm_", "momentum_", "cv_"))] + [
            "y_at_origin", "level", "weeks_since_launch",
            "promo_density_13", "stockout_density_13"]

        origin_rows = (panel[panel[Keys.WEEK_INDEX] <= origin_index]
                       .sort_values(Keys.WEEK_INDEX)
                       .groupby(Keys.SKU, as_index=False).tail(1))
        origin_rows = origin_rows[[Keys.SKU, Keys.WEEK_INDEX, Keys.CATEGORY,
                                   Keys.CHANNEL, "Unit_Price"] + backward_cols]

        exog = future_exog[future_exog["horizon"].between(1, horizon)]
        grid = origin_rows.merge(exog, on=Keys.SKU, how="inner")
        grid["target_week_index"] = origin_index + grid["horizon"]
        return grid

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric-only design matrix with stable column ordering."""
        work = frame.copy()
        work["cat_code"] = work[Keys.CATEGORY].astype("category").cat.codes
        work["chan_code"] = work[Keys.CHANNEL].astype("category").cat.codes
        work["Is_Promo"] = work["Is_Promo"].astype(int)

        fourier = self._fourier(work["ISO_Week"].to_numpy(dtype=float),
                                self.cfg.fourier_orders)
        for name, values in fourier.items():
            work[name] = values

        work["week_sin"] = np.sin(2 * np.pi * work["ISO_Week"] / 52.18)
        work["week_cos"] = np.cos(2 * np.pi * work["ISO_Week"] / 52.18)
        work["month_sin"] = np.sin(2 * np.pi * work["Month"] / 12.0)
        work["month_cos"] = np.cos(2 * np.pi * work["Month"] / 12.0)
        work["log_price"] = np.log1p(work["Unit_Price"].clip(lower=0))

        drop = {Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, "Sales_Units",
                "Demand_Units", "target_week_index", Keys.WEEK, "Festival_Names"}
        cols = [c for c in work.columns
                if c not in drop and pd.api.types.is_numeric_dtype(work[c])]
        if not self.feature_names:
            self.feature_names = cols
        else:
            for missing in set(self.feature_names) - set(cols):
                work[missing] = np.nan
            cols = self.feature_names
        return work[cols].astype(float)


# --------------------------------------------------------------------------- #
# Per-SKU statistical models
# --------------------------------------------------------------------------- #

def _log_forward(y: np.ndarray) -> np.ndarray:
    """log1p transform: multiplicative demand becomes additive."""
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))


def _log_inverse(y_hat: np.ndarray, residual_var: float = 0.0) -> np.ndarray:
    """
    Back-transform from log space with a Duan-style smearing correction.

    expm1(E[log y]) estimates the *median*, not the mean. Multiplying by
    exp(sigma^2 / 2) recovers the conditional mean, which is what a supply plan
    needs -- forecasting the median of a right-skewed demand distribution
    systematically under-plans and manufactures stockouts.
    """
    correction = float(np.exp(min(residual_var, 2.0) / 2.0))
    return np.clip(np.expm1(np.asarray(y_hat, dtype=float)) * correction, 0.0, None)


def _sanitise(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def forecast_holt_winters(y: np.ndarray, horizon: int, seasonal_periods: int = 13
                          ) -> np.ndarray:
    """
    Damped-trend Holt-Winters.

    ``seasonal_periods`` defaults to 13 (a quarter), not 52. With 104 weekly
    observations a 52-period seasonal component consumes half the sample as
    initial state and statsmodels legitimately refuses to fit it on any
    training subset. Annual seasonality is instead captured by the Fourier and
    festival regressors in the ML and Prophet models -- an explicit and
    documented division of labour rather than a silent degradation.
    """
    y = _sanitise(y)
    if not HAS_STATSMODELS or y.size < 2 * seasonal_periods + 4 or y.std() < 1e-6:
        return _naive_drift(y, horizon)

    # Fitted in log space: FMCG demand shocks are proportional, so an additive
    # model in logs is a multiplicative model in units -- and it cannot emit a
    # negative forecast.
    y_log = _log_forward(y)
    for seasonal in ("add", None):
        try:
            model = ExponentialSmoothing(
                y_log, trend="add", damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            resid_var = float(np.nanvar(model.resid)) if hasattr(model, "resid") else 0.0
            return _log_inverse(np.asarray(model.forecast(horizon)), resid_var)
        except Exception:
            continue
    return _naive_drift(y, horizon)


def forecast_sarima(y: np.ndarray, horizon: int, seasonal_periods: int = 13,
                    cached_order: Optional[Tuple] = None
                    ) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    Seasonal ARIMA.

    Order discovery via ``pmdarima.auto_arima`` is ~2s per series, so the
    discovered order is cached by the caller and re-applied through the much
    faster ``SARIMAX`` path on subsequent folds. Falls back to a fixed
    (1,1,1)(1,0,0,s) specification when pmdarima is unavailable.
    """
    y = _sanitise(y)
    if y.size < 25 or y.std() < 1e-6:
        return _naive_drift(y, horizon), cached_order

    y_log = _log_forward(y)

    if cached_order is None and HAS_PMDARIMA:
        try:
            model = pm.auto_arima(
                y_log, seasonal=True, m=seasonal_periods, stepwise=True,
                max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
                with_intercept=True, suppress_warnings=True,
                error_action="ignore", information_criterion="aicc")
            resid_var = float(np.nanvar(model.resid()))
            preds = _log_inverse(np.asarray(model.predict(n_periods=horizon)), resid_var)
            return preds, (model.order, model.seasonal_order)
        except Exception:
            cached_order = ((1, 1, 1), (1, 0, 0, seasonal_periods))

    order, seasonal_order = cached_order or ((1, 1, 1), (1, 0, 0, seasonal_periods))
    if not HAS_STATSMODELS:
        return _naive_drift(y, horizon), cached_order
    try:
        fitted = SARIMAX(y_log, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False, maxiter=60)
        resid_var = float(np.nanvar(fitted.resid))
        preds = _log_inverse(np.asarray(fitted.forecast(horizon)), resid_var)
        return preds, (order, seasonal_order)
    except Exception:
        return _naive_drift(y, horizon), cached_order


def forecast_prophet(y: np.ndarray, dates: pd.DatetimeIndex, horizon: int,
                     future_dates: pd.DatetimeIndex,
                     holidays: Optional[pd.DataFrame] = None,
                     promo_flags: Optional[np.ndarray] = None,
                     future_promo: Optional[np.ndarray] = None,
                     festival_intensity: Optional[np.ndarray] = None,
                     future_festival: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Prophet with the Indian festive calendar supplied as holiday regressors and
    the promotion grid plus festival intensity as extra regressors.

    Lunar-calendar festivals move several weeks year to year, so a fixed
    week-of-year seasonal index cannot represent them. Explicit holiday
    regressors are the reason Prophet is in this competition at all.

    Two performance decisions matter here and are deliberate:

    * The holiday window is +/- one week, not the three-week build-up used in
      the data generator. On weekly data Prophet materialises one dummy column
      per holiday-day, so a 29-day window creates ~350 columns and a 10x
      slowdown for no accuracy gain. The build-up shape is instead carried by
      the continuous ``festival`` regressor.
    * ``uncertainty_samples=0`` disables Prophet's internal Monte-Carlo bands.
      Intervals in this system are produced by a single calibrated,
      cross-validated method applied uniformly to every model, so per-model
      bands would be inconsistent as well as expensive.
    """
    y = _sanitise(y)
    if not HAS_PROPHET or y.size < 30 or y.std() < 1e-6:
        return _naive_drift(y, horizon)
    try:
        train = pd.DataFrame({"ds": dates[-y.size:], "y": y})
        kwargs = dict(weekly_seasonality=False, daily_seasonality=False,
                      yearly_seasonality=4, seasonality_mode="multiplicative",
                      changepoint_prior_scale=0.05, uncertainty_samples=0)
        if holidays is not None and not holidays.empty:
            kwargs["holidays"] = holidays
        model = Prophet(**kwargs)

        future = pd.DataFrame({"ds": future_dates[:horizon]})
        if promo_flags is not None:
            train["promo"] = np.asarray(promo_flags, dtype=float)[-y.size:]
            model.add_regressor("promo")
            future["promo"] = (np.asarray(future_promo, dtype=float)[:horizon]
                               if future_promo is not None and len(future_promo) >= horizon
                               else 0.0)
        if festival_intensity is not None:
            train["festival"] = np.asarray(festival_intensity, dtype=float)[-y.size:]
            model.add_regressor("festival")
            future["festival"] = (np.asarray(future_festival, dtype=float)[:horizon]
                                  if future_festival is not None
                                  and len(future_festival) >= horizon else 0.0)
        model.fit(train)
        forecast = model.predict(future)
        return np.clip(forecast["yhat"].to_numpy(), 0.0, None)
    except Exception:
        return _naive_drift(y, horizon)


def _naive_drift(y: np.ndarray, horizon: int) -> np.ndarray:
    """Last-value-plus-drift fallback used whenever a model cannot be fitted."""
    y = _sanitise(y)
    if y.size == 0:
        return np.zeros(horizon)
    level = float(np.mean(y[-4:])) if y.size >= 4 else float(y[-1])
    if y.size >= 8:
        drift = (float(np.mean(y[-4:])) - float(np.mean(y[-8:-4]))) / 4.0
    else:
        drift = 0.0
    return np.clip(level + drift * np.arange(1, horizon + 1), 0.0, None)


def seasonal_naive(y: np.ndarray, horizon: int, period: int = 52) -> np.ndarray:
    """
    Benchmark forecast: repeat the same week last year.

    This is the "do nothing clever" comparator that Forecast Value Added is
    measured against. If the sophisticated pipeline cannot beat it, the
    pipeline is not worth running -- and saying so is the point of FVA.
    """
    y = _sanitise(y)
    if y.size >= period + horizon:
        return np.clip(y[-period:-period + horizon] if period > horizon
                       else y[-period:][:horizon], 0.0, None)
    if y.size >= period:
        block = y[-period:]
        return np.clip(np.resize(block, horizon), 0.0, None)
    return _naive_drift(y, horizon)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

class DemandForecastEngine:
    """Orchestrates validation, model competition, ensembling and forecasting."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: ForecastConfig = config.forecast
        self.builder = DirectMultiHorizonFeatureBuilder(self.cfg)
        self._sarima_orders: Dict[str, Tuple] = {}
        self.checkpoint_dir = OUTPUT_DIR / "cache" / "stages"
        self._xgb_models: Dict[str, object] = {}
        self._panel_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _prepare(self, sales: pd.DataFrame) -> pd.DataFrame:
        needed = [Keys.SKU, Keys.WEEK, Keys.WEEK_INDEX, Keys.CATEGORY, Keys.CHANNEL,
                  "ISO_Week", "Month", "Sales_Units", "Demand_Units", "Is_Promo",
                  "Promo_Depth", "Price_Index", "Unit_Price", "Festival_Intensity",
                  "Is_Stockout"]
        frame = sales[needed].copy()
        frame[Keys.WEEK] = pd.to_datetime(frame[Keys.WEEK])
        frame["Is_Promo"] = frame["Is_Promo"].astype(int)
        frame["Is_Stockout"] = frame["Is_Stockout"].astype(int)
        return frame.sort_values([Keys.SKU, Keys.WEEK_INDEX]).reset_index(drop=True)

    def _select_panel(self, sales: pd.DataFrame) -> List[str]:
        """
        Revenue-stratified sample for the four-model competition.

        Stratifying by category AND revenue decile keeps the panel
        representative: sampling the top 40 SKUs would flatter every model,
        because high-volume series are far easier to forecast.
        """
        size = min(self.cfg.competition_panel_size, sales[Keys.SKU].nunique())
        if size <= 0:
            return []
        revenue = (sales.assign(rev=sales["Sales_Units"] * sales["Unit_Price"])
                        .groupby([Keys.SKU, Keys.CATEGORY])["rev"].sum()
                        .reset_index())
        revenue["decile"] = pd.qcut(revenue["rev"].rank(method="first"), 5,
                                    labels=False, duplicates="drop")
        rng = np.random.default_rng(self.config.seed)
        picked: List[str] = []
        strata = revenue.groupby([Keys.CATEGORY, "decile"])
        per_stratum = max(1, size // max(strata.ngroups, 1))
        for _, group in strata:
            take = min(per_stratum, len(group))
            picked.extend(rng.choice(group[Keys.SKU].to_numpy(), take, replace=False))
        if len(picked) > size:
            picked = list(rng.choice(picked, size, replace=False))
        elif len(picked) < size:
            remaining = list(set(revenue[Keys.SKU]) - set(picked))
            extra = min(size - len(picked), len(remaining))
            picked.extend(rng.choice(remaining, extra, replace=False))
        return sorted(set(picked))

    def _holiday_frame(self, festivals: pd.DataFrame) -> pd.DataFrame:
        """Prophet-format holiday table built from the festive calendar."""
        if festivals is None or festivals.empty:
            return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
        frame = festivals.copy()
        frame[Keys.WEEK] = pd.to_datetime(frame[Keys.WEEK])
        out = (frame[frame["Week_Offset"] == 0][["Festival", "Event_Date"]]
               .drop_duplicates()
               .rename(columns={"Festival": "holiday", "Event_Date": "ds"}))
        out["ds"] = pd.to_datetime(out["ds"])
        out["lower_window"] = -6
        out["upper_window"] = 1
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Forward planning calendar (exogenous regressors for the horizon)
    # ------------------------------------------------------------------ #
    def build_future_calendar(self, sales: pd.DataFrame,
                              festivals: pd.DataFrame) -> pd.DataFrame:
        """
        The forward-known planning calendar for the next 12 weeks.

        In a live S&OP process this is not a guess: trade marketing owns the
        promotion grid and revenue management owns the price plan a quarter
        ahead. Here it is simulated deterministically, but it enters the model
        the same way it would in production -- as a *known* future regressor.
        """
        rng = np.random.default_rng(self.config.seed + 77)
        horizon = self.cfg.horizon_weeks
        last_index = int(sales[Keys.WEEK_INDEX].max())
        last_week = pd.to_datetime(sales[Keys.WEEK].max())

        future_weeks = pd.date_range(last_week + pd.Timedelta(weeks=1),
                                     periods=horizon, freq="W-MON")

        fest = festivals.copy()
        fest[Keys.WEEK] = pd.to_datetime(fest[Keys.WEEK])
        weekly_fest = (fest.groupby(Keys.WEEK)["Intensity"].sum()
                       .reindex(future_weeks).fillna(0.0).clip(upper=1.6))

        skus = sales[[Keys.SKU]].drop_duplicates()
        promo_rate = (sales.groupby(Keys.SKU)["Is_Promo"].mean()
                      .reindex(skus[Keys.SKU]).fillna(0.1).to_numpy())

        rows = []
        for h, (week, intensity) in enumerate(zip(future_weeks, weekly_fest.to_numpy()), start=1):
            # Festive weeks attract more promotional activity -- the plan is
            # correlated with the calendar, as it is in reality.
            rate = np.clip(promo_rate * (1.0 + 1.4 * intensity), 0.0, 0.8)
            promo = rng.random(len(skus)) < rate
            depth = np.where(promo, rng.uniform(0.08, 0.35, len(skus)), 0.0)
            rows.append(pd.DataFrame({
                Keys.SKU: skus[Keys.SKU].to_numpy(),
                "horizon": h,
                "target_week_index": last_index + h,
                Keys.WEEK: week,
                "ISO_Week": week.isocalendar()[1],
                "Month": week.month,
                "Is_Promo": promo.astype(int),
                "Promo_Depth": np.round(depth, 3),
                "Price_Index": np.round(1.0 - depth, 4),
                "Festival_Intensity": round(float(intensity), 4),
            }))
        calendar = pd.concat(rows, ignore_index=True)
        save_csv(calendar, "future_calendar.csv", processed=True)
        return calendar

    def _historical_exog(self, panel: pd.DataFrame, origin_index: int,
                         horizon: int) -> pd.DataFrame:
        """
        Exogenous regressors for an *evaluation* window.

        These are the actual historical promo/price/festival values for the
        weeks being predicted. That is not leakage: the promotion grid and
        price plan are known in advance in a real process, so a live model
        would have had exactly this information. The target itself never
        enters.
        """
        window = panel[(panel[Keys.WEEK_INDEX] > origin_index) &
                       (panel[Keys.WEEK_INDEX] <= origin_index + horizon)].copy()
        window["horizon"] = window[Keys.WEEK_INDEX] - origin_index
        return window[[Keys.SKU, "horizon", Keys.WEEK, "ISO_Week", "Month",
                       "Is_Promo", "Promo_Depth", "Price_Index",
                       "Festival_Intensity"]]

    # ------------------------------------------------------------------ #
    # Global XGBoost
    # ------------------------------------------------------------------ #
    def _fit_xgboost(self, panel: pd.DataFrame, origin_index: int,
                     stage: str) -> Optional[object]:
        """
        Fit the global gradient-boosted model on every SKU-week pair whose
        target falls at or before ``origin_index``.

        One model serves all 500 SKUs. Beyond being ~500x cheaper than a model
        per SKU, cross-learning is a real accuracy advantage: a new or
        low-volume SKU inherits the promotion and festival response estimated
        from the whole portfolio instead of trying to learn it from its own
        thin history.
        """
        if not HAS_XGBOOST:
            return None
        X, y = self.builder.build_training_matrix(
            panel, origin_max=origin_index, horizon=self.cfg.horizon_weeks)
        if X.empty or len(X) < 500:
            return None

        # Cap the training matrix for tractability on modest hardware; the
        # sample is drawn without replacement and is therefore unbiased.
        max_rows = 150_000
        if len(X) > max_rows:
            idx = np.random.default_rng(self.config.seed).choice(
                len(X), max_rows, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]

        params = dict(self.cfg.xgb_params)
        model = xgb.XGBRegressor(**params, random_state=self.config.seed)
        model.fit(X, y, verbose=False)
        self._xgb_models[stage] = model
        self.builder.feature_names = list(X.columns)

        # The booster is persisted next to the stage checkpoint. Without this a
        # run resumed from cache would produce empty feature importance and no
        # SHAP attribution -- the explainability would silently disappear, which
        # is a worse failure than the cache simply not existing.
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(self._xgb_model_path(stage))
            pd.Series(self.builder.feature_names).to_csv(
                self._xgb_features_path(stage), index=False, header=["feature"])
        except Exception as exc:                               # pragma: no cover
            logger.debug("Could not persist XGBoost model for %s: %s", stage, exc)

        logger.info("XGBoost [%s]: trained on %s rows x %d features.",
                    stage, f"{len(X):,}", X.shape[1])
        return model

    def _predict_xgboost(self, model, panel: pd.DataFrame, origin_index: int,
                         exog: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if model is None:
            return pd.DataFrame(columns=[Keys.SKU, "horizon", "Forecast"])
        grid = self.builder.build_prediction_matrix(panel, origin_index, exog, horizon)
        if grid.empty:
            return pd.DataFrame(columns=[Keys.SKU, "horizon", "Forecast"])
        X = self.builder._encode(grid)
        preds = np.clip(model.predict(X), 0.0, None)
        return pd.DataFrame({
            Keys.SKU: grid[Keys.SKU].to_numpy(),
            "horizon": grid["horizon"].to_numpy(),
            "Forecast": preds,
        })

    # ------------------------------------------------------------------ #
    # Stage runner
    # ------------------------------------------------------------------ #
    def _config_fingerprint(self) -> str:
        """
        Short hash of every setting that changes what a stage computes.

        Without this, a checkpoint from one configuration could be silently
        reused by another. A 60-SKU test run and a 500-SKU production run
        produce identical (stage, origin) keys, so the test cache would be
        loaded as though it were the real forecast -- a wrong answer delivered
        confidently, which is the worst failure mode a cache can have.
        """
        cfg = self.cfg
        signature = "|".join(str(x) for x in (
            self.config.seed,
            self.config.data.n_skus,
            self.config.data.history_weeks,
            cfg.horizon_weeks,
            cfg.cv_folds,
            cfg.competition_panel_size,
            cfg.mass_tier_models,
            cfg.panel_tier_models,
            cfg.xgb_lags,
            cfg.xgb_rolling_windows,
            sorted(cfg.xgb_params.items(), key=lambda kv: kv[0]),
        ))
        return hashlib.sha1(signature.encode()).hexdigest()[:10]

    def _xgb_model_path(self, stage: str) -> str:
        return str(self.checkpoint_dir /
                   f"xgb_{stage}_{self._config_fingerprint()}.json")

    def _xgb_features_path(self, stage: str):
        return self.checkpoint_dir / f"xgbfeat_{stage}_{self._config_fingerprint()}.csv"

    def _restore_xgboost(self, stage: str) -> bool:
        """Reload a persisted booster so explainability survives a resumed run."""
        if not HAS_XGBOOST:
            return False
        path = self._xgb_model_path(stage)
        features = self._xgb_features_path(stage)
        if not (Path(path).exists() and features.exists()):
            return False
        try:
            model = xgb.XGBRegressor()
            model.load_model(path)
            self._xgb_models[stage] = model
            if not self.builder.feature_names:
                self.builder.feature_names = pd.read_csv(
                    features)["feature"].tolist()
            logger.info("Restored XGBoost model for stage %s from checkpoint.", stage)
            return True
        except Exception as exc:                               # pragma: no cover
            logger.debug("Could not restore XGBoost model for %s: %s", stage, exc)
            return False

    def _stage_checkpoint_path(self, stage: str, origin_index: int):
        return (self.checkpoint_dir /
                f"{stage}_origin{origin_index}_{self._config_fingerprint()}.parquet")

    def _run_stage(self, panel: pd.DataFrame, series_map: Dict[str, pd.DataFrame],
                   origin_index: int, horizon: int, stage: str,
                   sku_models: Dict[str, Sequence[str]],
                   holidays: pd.DataFrame,
                   future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Produce every model's forecast for one cutoff.

        Returns a long frame: SKU_ID, horizon, Model, Forecast, Stage.

        Each cutoff is checkpointed to ``outputs/cache/stages``. Fitting four
        model families across 500 SKUs takes minutes per cutoff, and an
        interrupted run -- a timeout, a laptop lid, a cancelled CI job --
        should not discard completed work. On re-run, finished cutoffs are
        reloaded and only the outstanding ones are fitted.

        Checkpoints are keyed by stage AND origin index, so a configuration
        change that moves the cutoffs produces different keys and cannot
        silently reuse stale predictions.
        """
        checkpoint = self._stage_checkpoint_path(stage, origin_index)
        if checkpoint.exists():
            try:
                cached = pd.read_parquet(checkpoint)
                self._restore_xgboost(stage)
                logger.info("Stage %s (origin %d) restored from checkpoint (%s rows).",
                            stage, origin_index, f"{len(cached):,}")
                return cached
            except Exception as exc:                           # pragma: no cover
                logger.warning("Checkpoint %s unreadable (%s); recomputing.",
                               checkpoint.name, exc)
        horizon_exog = (future_exog if future_exog is not None
                        else self._historical_exog(panel, origin_index, horizon))

        records: List[pd.DataFrame] = []

        # --- global ML model ------------------------------------------- #
        xgb_model = self._fit_xgboost(panel[panel[Keys.WEEK_INDEX] <= origin_index],
                                      origin_index, stage)
        xgb_preds = self._predict_xgboost(xgb_model, panel, origin_index,
                                          horizon_exog, horizon)
        if not xgb_preds.empty:
            xgb_preds["Model"] = "XGBoost"
            records.append(xgb_preds)

        # --- per-SKU statistical models --------------------------------- #
        future_dates = self._future_dates(panel, origin_index, horizon)
        stat_rows: List[Dict[str, object]] = []

        for sku, models in sku_models.items():
            series = series_map[sku]
            train = series[series[Keys.WEEK_INDEX] <= origin_index]
            y = train["Sales_Units"].to_numpy(dtype=float)
            if y.size < 12:
                continue
            dates = pd.DatetimeIndex(train[Keys.WEEK])

            if "HoltWinters" in models:
                preds = forecast_holt_winters(y, horizon)
                stat_rows.extend(self._long(sku, "HoltWinters", preds))

            if "SARIMA" in models:
                cached = self._sarima_orders.get(sku)
                preds, order = forecast_sarima(y, horizon, cached_order=cached)
                if order is not None:
                    self._sarima_orders[sku] = order
                stat_rows.extend(self._long(sku, "SARIMA", preds))

            if "Prophet" in models:
                sku_exog = horizon_exog[horizon_exog[Keys.SKU] == sku].sort_values("horizon")
                preds = forecast_prophet(
                    y, dates, horizon, future_dates, holidays,
                    promo_flags=train["Is_Promo"].to_numpy(dtype=float),
                    future_promo=(sku_exog["Is_Promo"].to_numpy(dtype=float)
                                  if not sku_exog.empty else None),
                    festival_intensity=train["Festival_Intensity"].to_numpy(dtype=float),
                    future_festival=(sku_exog["Festival_Intensity"].to_numpy(dtype=float)
                                     if not sku_exog.empty else None))
                stat_rows.extend(self._long(sku, "Prophet", preds))

            # Benchmark for Forecast Value Added.
            stat_rows.extend(self._long(sku, "SeasonalNaive",
                                        seasonal_naive(y, horizon)))

        if stat_rows:
            records.append(pd.DataFrame(stat_rows))

        if not records:
            return pd.DataFrame(columns=[Keys.SKU, "horizon", "Model", "Forecast", "Stage"])

        out = pd.concat(records, ignore_index=True)
        out["Stage"] = stage
        out["origin_index"] = origin_index

        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            out.to_parquet(checkpoint, index=False)
        except Exception as exc:                               # pragma: no cover
            logger.debug("Could not write checkpoint %s: %s", checkpoint.name, exc)
        return out

    @staticmethod
    def _long(sku: str, model: str, preds: np.ndarray) -> List[Dict[str, object]]:
        return [{Keys.SKU: sku, "horizon": h, "Model": model, "Forecast": float(v)}
                for h, v in enumerate(np.asarray(preds).ravel(), start=1)]

    @staticmethod
    def _future_dates(panel: pd.DataFrame, origin_index: int,
                      horizon: int) -> pd.DatetimeIndex:
        anchor = panel.loc[panel[Keys.WEEK_INDEX] == origin_index, Keys.WEEK]
        start = (pd.to_datetime(anchor.iloc[0]) if len(anchor)
                 else pd.to_datetime(panel[Keys.WEEK].max()))
        return pd.date_range(start + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score(self, predictions: pd.DataFrame, panel: pd.DataFrame,
               stage: str) -> pd.DataFrame:
        """Join predictions to actuals and compute the accuracy panel per SKU x model."""
        if predictions.empty:
            return pd.DataFrame()

        preds = predictions.copy()
        preds["target_week_index"] = preds["origin_index"] + preds["horizon"]
        actuals = panel[[Keys.SKU, Keys.WEEK_INDEX, "Sales_Units", Keys.CATEGORY,
                         Keys.CHANNEL]].rename(
            columns={Keys.WEEK_INDEX: "target_week_index", "Sales_Units": "Actual"})
        merged = preds.merge(actuals, on=[Keys.SKU, "target_week_index"], how="inner")
        if merged.empty:
            return pd.DataFrame()

        naive = (merged[merged["Model"] == "SeasonalNaive"]
                 [[Keys.SKU, "target_week_index", "Forecast"]]
                 .rename(columns={"Forecast": "Naive"}))
        merged = merged.merge(naive, on=[Keys.SKU, "target_week_index"], how="left")

        rows: List[Dict[str, object]] = []
        for (sku, model), group in merged.groupby([Keys.SKU, "Model"], sort=False):
            bundle = accuracy_bundle(group["Actual"], group["Forecast"],
                                     group["Naive"] if group["Naive"].notna().all() else None)
            bundle.update({
                Keys.SKU: sku, "Model": model, "Stage": stage,
                Keys.CATEGORY: group[Keys.CATEGORY].iloc[0],
                Keys.CHANNEL: group[Keys.CHANNEL].iloc[0],
                "Mean_Actual": float(group["Actual"].mean()),
                "Total_Actual": float(group["Actual"].sum()),
            })
            rows.append(bundle)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Ensemble
    # ------------------------------------------------------------------ #
    def _build_weights(self, cv_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse-MAPE ensemble weights, derived exclusively from CV folds.

            w_i = (1 / MAPE_i) / sum_j (1 / MAPE_j)

        SeasonalNaive is excluded from the blend: it is the benchmark the
        ensemble is measured against, not a member of it.
        """
        if cv_scores.empty:
            return pd.DataFrame(columns=[Keys.SKU, "Model", "Weight"])

        eligible = cv_scores[cv_scores["Model"] != "SeasonalNaive"]
        aggregated = (eligible.groupby([Keys.SKU, "Model"])["MAPE"].mean()
                      .reset_index())

        rows: List[Dict[str, object]] = []
        for sku, group in aggregated.groupby(Keys.SKU, sort=False):
            errors = dict(zip(group["Model"], group["MAPE"]))
            weights = inverse_error_weights(
                errors, floor=self.cfg.mape_floor,
                min_weight=self.cfg.min_ensemble_weight)
            for model, weight in weights.items():
                rows.append({Keys.SKU: sku, "Model": model,
                             "Weight": round(float(weight), 6)})
        return pd.DataFrame(rows)

    @staticmethod
    def _apply_weights(predictions: pd.DataFrame, weights: pd.DataFrame,
                       label: str = "Ensemble") -> pd.DataFrame:
        """Blend member forecasts into the weighted ensemble forecast."""
        if predictions.empty or weights.empty:
            return pd.DataFrame()
        merged = predictions.merge(weights, on=[Keys.SKU, "Model"], how="inner")
        if merged.empty:
            return pd.DataFrame()
        merged["weighted"] = merged["Forecast"] * merged["Weight"]
        blended = (merged.groupby([Keys.SKU, "horizon", "origin_index", "Stage"],
                                  as_index=False)
                   .agg(Forecast=("weighted", "sum"), weight_sum=("Weight", "sum")))
        blended["Forecast"] = blended["Forecast"] / blended["weight_sum"].replace(0, np.nan)
        blended["Model"] = label
        return blended.drop(columns=["weight_sum"])

    def _select_models(self, cv_scores: pd.DataFrame,
                       ensemble_cv: pd.DataFrame) -> pd.DataFrame:
        """
        Per-SKU champion selection on CV performance only.

        The ensemble competes as one more candidate; it does not win by
        default. Where a single model is genuinely better, that model is
        selected, and the leaderboard reports how often that happens.
        """
        candidates = pd.concat(
            [cv_scores[cv_scores["Model"] != "SeasonalNaive"], ensemble_cv],
            ignore_index=True)
        if candidates.empty:
            return pd.DataFrame(columns=[Keys.SKU, "Selected_Model", "CV_MAPE"])
        agg = (candidates.groupby([Keys.SKU, "Model"])["MAPE"].mean().reset_index())
        agg = agg.dropna(subset=["MAPE"])
        best = agg.loc[agg.groupby(Keys.SKU)["MAPE"].idxmin()]
        return best.rename(columns={"Model": "Selected_Model", "MAPE": "CV_MAPE"})

    # ------------------------------------------------------------------ #
    # Prediction intervals
    # ------------------------------------------------------------------ #
    def _calibrate_intervals(self, cv_predictions: pd.DataFrame,
                             panel: pd.DataFrame,
                             selection: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Empirical, self-calibrating prediction intervals.

        Per SKU, sigma_1 is the standard deviation of the champion model's
        cross-validation residuals at short horizons. Uncertainty then grows
        with the horizon as

            sigma_h = sigma_1 * sqrt(1 + lambda * (h - 1))

        Rather than accepting the configured lambda on faith, it is *fitted*
        from the observed residual spread at long versus short horizons across
        the whole portfolio, then floored at the configured value. The
        resulting coverage is measured on the untouched holdout and reported.
        """
        if cv_predictions.empty:
            return pd.DataFrame(columns=[Keys.SKU, "sigma_1"]), self.cfg.horizon_variance_growth

        preds = cv_predictions.copy()
        preds["target_week_index"] = preds["origin_index"] + preds["horizon"]
        actuals = panel[[Keys.SKU, Keys.WEEK_INDEX, "Sales_Units"]].rename(
            columns={Keys.WEEK_INDEX: "target_week_index", "Sales_Units": "Actual"})
        merged = preds.merge(actuals, on=[Keys.SKU, "target_week_index"], how="inner")
        merged = merged.merge(selection[[Keys.SKU, "Selected_Model"]], on=Keys.SKU, how="inner")
        merged = merged[merged["Model"] == merged["Selected_Model"]]
        if merged.empty:
            return pd.DataFrame(columns=[Keys.SKU, "sigma_1"]), self.cfg.horizon_variance_growth

        merged["residual"] = merged["Actual"] - merged["Forecast"]

        short = merged[merged["horizon"] <= 4]["residual"]
        long_h = merged[merged["horizon"] >= 9]["residual"]
        lam = self.cfg.horizon_variance_growth
        if len(short) > 50 and len(long_h) > 50 and short.std() > 1e-6:
            ratio = (long_h.std() / short.std()) ** 2      # variance ratio
            # sigma_h^2 / sigma_1^2 = 1 + lambda*(h-1); midpoints h=2.5 and h=10.5
            implied = (ratio - 1.0) / max(10.5 - 2.5, 1.0)
            lam = float(np.clip(implied, self.cfg.horizon_variance_growth, 0.9))

        sigma = (merged[merged["horizon"] <= 4]
                 .groupby(Keys.SKU)["residual"].std().rename("sigma_1").reset_index())
        overall = (merged.groupby(Keys.SKU)["residual"].std().rename("sigma_all")
                   .reset_index())
        sigma = sigma.merge(overall, on=Keys.SKU, how="outer")
        sigma["sigma_1"] = sigma["sigma_1"].fillna(sigma["sigma_all"]).fillna(0.0)
        return sigma[[Keys.SKU, "sigma_1"]], lam

    def _attach_intervals(self, forecast: pd.DataFrame, sigma: pd.DataFrame,
                          lam: float) -> pd.DataFrame:
        z80 = service_level_z(0.90)     # two-sided 80% -> 90th percentile
        z95 = service_level_z(0.975)    # two-sided 95% -> 97.5th percentile

        out = forecast.merge(sigma, on=Keys.SKU, how="left")
        # SKUs without CV residuals fall back to a coefficient-of-variation
        # proxy so that no forecast is ever published without an interval.
        fallback = out["Forecast"] * 0.35
        out["sigma_1"] = out["sigma_1"].fillna(fallback).fillna(0.0)
        out["sigma_h"] = out["sigma_1"] * np.sqrt(1.0 + lam * (out["horizon"] - 1.0))

        out["P10"] = np.clip(out["Forecast"] - z80 * out["sigma_h"], 0.0, None)
        out["P90"] = out["Forecast"] + z80 * out["sigma_h"]
        out["P2_5"] = np.clip(out["Forecast"] - z95 * out["sigma_h"], 0.0, None)
        out["P97_5"] = out["Forecast"] + z95 * out["sigma_h"]
        return out

    @staticmethod
    def _interval_coverage(holdout_forecast: pd.DataFrame,
                           panel: pd.DataFrame) -> Dict[str, float]:
        """Empirical coverage of the 80% and 95% bands on the untouched holdout."""
        if holdout_forecast.empty:
            return {"coverage_80": float("nan"), "coverage_95": float("nan")}
        merged = holdout_forecast.merge(
            panel[[Keys.SKU, Keys.WEEK_INDEX, "Sales_Units"]].rename(
                columns={Keys.WEEK_INDEX: "target_week_index", "Sales_Units": "Actual"}),
            on=[Keys.SKU, "target_week_index"], how="inner")
        if merged.empty:
            return {"coverage_80": float("nan"), "coverage_95": float("nan")}
        return {
            "coverage_80": float(((merged["Actual"] >= merged["P10"]) &
                                  (merged["Actual"] <= merged["P90"])).mean() * 100),
            "coverage_95": float(((merged["Actual"] >= merged["P2_5"]) &
                                  (merged["Actual"] <= merged["P97_5"])).mean() * 100),
        }

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def _explain(self, panel: pd.DataFrame, origin_index: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Gain-based feature importance plus a SHAP global attribution summary."""
        model = self._xgb_models.get("FINAL")
        empty = pd.DataFrame(columns=["Feature", "Importance"])
        if model is None:
            return empty, pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP"])

        names = self.builder.feature_names
        gains = getattr(model, "feature_importances_", None)
        if gains is None:
            importance = empty
        else:
            importance = (pd.DataFrame({"Feature": names, "Importance": gains})
                          .sort_values("Importance", ascending=False)
                          .reset_index(drop=True))
            importance["Importance_Pct"] = (
                importance["Importance"] / importance["Importance"].sum() * 100).round(3)

        shap_summary = pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP"])
        if HAS_SHAP:
            try:
                X, _ = self.builder.build_training_matrix(
                    panel, origin_max=origin_index, horizon=self.cfg.horizon_weeks)
                if not X.empty:
                    sample = X.sample(min(self.cfg.shap_background_rows, len(X)),
                                      random_state=self.config.seed)
                    explainer = shap.TreeExplainer(model)
                    values = explainer.shap_values(sample, check_additivity=False)
                    shap_summary = (pd.DataFrame({
                        "Feature": sample.columns,
                        "Mean_Abs_SHAP": np.abs(values).mean(axis=0)})
                        .sort_values("Mean_Abs_SHAP", ascending=False)
                        .reset_index(drop=True))
                    shap_summary["Share_Pct"] = (
                        shap_summary["Mean_Abs_SHAP"] /
                        shap_summary["Mean_Abs_SHAP"].sum() * 100).round(3)
                    logger.info("SHAP attribution computed on %d rows.", len(sample))
            except Exception as exc:                       # pragma: no cover
                logger.warning("SHAP explanation unavailable: %s", exc)
        return importance, shap_summary

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self, sales: pd.DataFrame, festivals: pd.DataFrame) -> ForecastResult:
        cfg = self.cfg
        panel_raw = self._prepare(sales)
        n_weeks = int(panel_raw[Keys.WEEK_INDEX].max()) + 1
        horizon = cfg.horizon_weeks

        holidays = self._holiday_frame(festivals)
        competition_panel = set(self._select_panel(panel_raw))
        all_skus = panel_raw[Keys.SKU].unique().tolist()

        sku_models: Dict[str, Sequence[str]] = {
            sku: (cfg.panel_tier_models if sku in competition_panel
                  else tuple(m for m in cfg.mass_tier_models if m != "XGBoost"))
            for sku in all_skus
        }

        # Feature panel is built once on the full history; every downstream
        # cutoff simply filters it, so features are identical across stages.
        feature_panel = self.builder.build_panel(panel_raw)
        series_map = {sku: g for sku, g in panel_raw.groupby(Keys.SKU, sort=False)}

        splits = rolling_origin_splits(
            n_obs=n_weeks, horizon=horizon, n_folds=cfg.cv_folds,
            min_train=cfg.min_train_weeks, step=cfg.cv_step_weeks, holdout=horizon)
        assert_no_leakage(splits)
        hold = holdout_split(n_weeks, horizon)
        logger.info("Validation plan: %d CV fold(s) at origins %s + holdout origin %s.",
                    len(splits), [s.train_end - 1 for s in splits],
                    hold.train_end - 1 if hold else "n/a")

        # ---------------- Cross-validation ------------------------------ #
        cv_predictions: List[pd.DataFrame] = []
        for split in splits:
            with timed(f"CV fold {split.label} (origin week {split.train_end - 1})", logger):
                cv_predictions.append(self._run_stage(
                    feature_panel, series_map, split.train_end - 1, horizon,
                    split.label, sku_models, holidays))
        cv_pred = (pd.concat(cv_predictions, ignore_index=True)
                   if cv_predictions else pd.DataFrame())
        cv_scores = self._score(cv_pred, panel_raw, "CV")

        # ---------------- Ensemble & selection (CV only) ----------------- #
        weights = self._build_weights(cv_scores)
        ensemble_cv_pred = self._apply_weights(cv_pred, weights)
        ensemble_cv_scores = self._score(ensemble_cv_pred, panel_raw, "CV")
        selection = self._select_models(cv_scores, ensemble_cv_scores)
        logger.info("Champion model selected for %d SKUs. Mix: %s", len(selection),
                    selection["Selected_Model"].value_counts().to_dict()
                    if not selection.empty else {})

        # ---------------- Honest holdout evaluation ---------------------- #
        holdout_pred = pd.DataFrame()
        holdout_scores = pd.DataFrame()
        if hold is not None:
            with timed("Holdout evaluation (untouched final 12 weeks)", logger):
                holdout_pred = self._run_stage(
                    feature_panel, series_map, hold.train_end - 1, horizon,
                    "HOLDOUT", sku_models, holidays)
                ensemble_holdout = self._apply_weights(holdout_pred, weights)
                holdout_pred = pd.concat([holdout_pred, ensemble_holdout],
                                         ignore_index=True)
                holdout_scores = self._score(holdout_pred, panel_raw, "HOLDOUT")

        # ---------------- Final refit & forward forecast ----------------- #
        future_calendar = self.build_future_calendar(panel_raw, festivals)
        with timed("Final refit on full history + 12-week forecast", logger):
            final_pred = self._run_stage(
                feature_panel, series_map, n_weeks - 1, horizon, "FINAL",
                sku_models, holidays, future_exog=future_calendar)
            ensemble_final = self._apply_weights(final_pred, weights)
            final_pred = pd.concat([final_pred, ensemble_final], ignore_index=True)

        # Keep only each SKU's champion forecast for the published plan.
        champion = final_pred.merge(
            selection[[Keys.SKU, "Selected_Model", "CV_MAPE"]], on=Keys.SKU, how="inner")
        champion = champion[champion["Model"] == champion["Selected_Model"]].copy()

        sigma, lam = self._calibrate_intervals(cv_pred, panel_raw, selection)
        champion = self._attach_intervals(champion, sigma, lam)
        champion["target_week_index"] = champion["origin_index"] + champion["horizon"]
        champion = champion.merge(
            future_calendar[[Keys.SKU, "horizon", Keys.WEEK, "Is_Promo",
                             "Festival_Intensity"]],
            on=[Keys.SKU, "horizon"], how="left")
        champion = champion.merge(
            sales[[Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, Keys.SUPPLIER,
                   Keys.WAREHOUSE, "Unit_Price", "Unit_Cost"]].drop_duplicates(Keys.SKU),
            on=Keys.SKU, how="left")
        champion["Forecast_Value"] = (champion["Forecast"] * champion["Unit_Price"]).round(2)
        champion = champion.rename(columns={"Forecast": "Forecast_Units"})

        # Interval coverage measured on the holdout, using the same recipe.
        holdout_champion = holdout_pred.merge(
            selection[[Keys.SKU, "Selected_Model"]], on=Keys.SKU, how="inner")
        holdout_champion = holdout_champion[
            holdout_champion["Model"] == holdout_champion["Selected_Model"]].copy()
        if not holdout_champion.empty:
            holdout_champion = self._attach_intervals(holdout_champion, sigma, lam)
            holdout_champion["target_week_index"] = (
                holdout_champion["origin_index"] + holdout_champion["horizon"])
        coverage = self._interval_coverage(holdout_champion, panel_raw)

        # ---------------- Reporting frames ------------------------------ #
        importance, shap_summary = self._explain(feature_panel, n_weeks - 1)
        all_scores = pd.concat([s for s in (cv_scores, ensemble_cv_scores, holdout_scores)
                                if not s.empty], ignore_index=True)
        leaderboard = self._leaderboard(all_scores, competition_panel)
        sku_accuracy = self._sku_accuracy(holdout_scores, selection, competition_panel)
        category_accuracy = self._category_accuracy(holdout_scores, selection)
        hierarchy_accuracy = self._hierarchical_accuracy(
            holdout_pred, selection, panel_raw)
        uncertainty = self._uncertainty_contract(champion, panel_raw, sigma, lam)

        diagnostics = {
            "dependencies": dependency_report(),
            "n_skus": len(all_skus),
            "n_weeks": n_weeks,
            "cv_folds": len(splits),
            "competition_panel_size": len(competition_panel),
            "horizon_variance_lambda": round(lam, 4),
            "interval_coverage": coverage,
            "portfolio_holdout_wape": self._portfolio_metric(
                holdout_scores, selection, "WAPE"),
            "portfolio_holdout_mape": self._portfolio_metric(
                holdout_scores, selection, "MAPE"),
            "mape_target": cfg.portfolio_mape_target,
            "hierarchy": (hierarchy_accuracy.to_dict("records")
                          if not hierarchy_accuracy.empty else []),
        }
        logger.info("Holdout (unbiased): median SKU MAPE %.1f%% | volume-weighted WAPE %.1f%% "
                    "| PI80 coverage %.1f%%",
                    diagnostics["portfolio_holdout_mape"] or float("nan"),
                    diagnostics["portfolio_holdout_wape"] or float("nan"),
                    coverage.get("coverage_80", float("nan")))

        return ForecastResult(
            forecast=champion,
            holdout_predictions=holdout_pred,
            model_scores=all_scores,
            ensemble_weights=weights,
            sku_accuracy=sku_accuracy,
            category_accuracy=category_accuracy,
            hierarchy_accuracy=hierarchy_accuracy,
            model_leaderboard=leaderboard,
            feature_importance=importance,
            shap_summary=shap_summary,
            uncertainty=uncertainty,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Reporting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _portfolio_metric(holdout_scores: pd.DataFrame, selection: pd.DataFrame,
                          metric: str) -> Optional[float]:
        if holdout_scores.empty or selection.empty:
            return None
        merged = holdout_scores.merge(selection[[Keys.SKU, "Selected_Model"]],
                                      on=Keys.SKU, how="inner")
        champ = merged[merged["Model"] == merged["Selected_Model"]]
        if champ.empty:
            return None
        if metric == "WAPE":
            # Volume-weighted: the number a supply chain director steers on.
            weights = champ["Total_Actual"].clip(lower=0)
            if weights.sum() <= 0:
                return float(champ["WAPE"].median())
            return float(np.average(champ["WAPE"].fillna(champ["WAPE"].median()),
                                    weights=weights))
        return float(champ[metric].median())

    def _leaderboard(self, scores: pd.DataFrame,
                     panel_skus: set) -> pd.DataFrame:
        """Aggregate model comparison, split by tier so the comparison is fair."""
        if scores.empty:
            return pd.DataFrame()
        work = scores.copy()
        work["Tier"] = np.where(work[Keys.SKU].isin(panel_skus),
                                "Competition Panel", "Full Portfolio")
        rows = (work.groupby(["Stage", "Tier", "Model"])
                .agg(SKUs=(Keys.SKU, "nunique"),
                     Median_MAPE=("MAPE", "median"),
                     Mean_MAPE=("MAPE", "mean"),
                     Median_WAPE=("WAPE", "median"),
                     Median_MAE=("MAE", "median"),
                     Median_RMSE=("RMSE", "median"),
                     Mean_Bias_Pct=("Bias_Pct", "mean"),
                     Median_FVA=("FVA_vs_Naive", "median"))
                .reset_index())
        for col in rows.columns:
            if rows[col].dtype.kind == "f":
                rows[col] = rows[col].round(3)
        return rows.sort_values(["Stage", "Tier", "Median_MAPE"])

    @staticmethod
    def _sku_accuracy(holdout_scores: pd.DataFrame, selection: pd.DataFrame,
                      panel_skus: set) -> pd.DataFrame:
        if holdout_scores.empty:
            return pd.DataFrame()
        merged = holdout_scores.merge(selection, on=Keys.SKU, how="left")
        champ = merged[merged["Model"] == merged["Selected_Model"]].copy()
        champ["Tier"] = np.where(champ[Keys.SKU].isin(panel_skus),
                                 "Competition Panel", "Full Portfolio")
        champ["Accuracy_Band"] = pd.cut(
            champ["MAPE"], bins=[-0.01, 10, 20, 35, 1e9],
            labels=["Excellent (<10%)", "Good (10-20%)",
                    "Acceptable (20-35%)", "Poor (>35%)"])
        keep = [Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, "Selected_Model", "CV_MAPE",
                "MAPE", "sMAPE", "WAPE", "MAE", "RMSE", "Bias", "Bias_Pct",
                "Tracking_Signal", "FVA_vs_Naive", "Mean_Actual", "Total_Actual",
                "Tier", "Accuracy_Band"]
        out = champ[[c for c in keep if c in champ.columns]].copy()
        for col in out.select_dtypes("float").columns:
            out[col] = out[col].round(3)
        return out.sort_values("MAPE")

    @staticmethod
    def _category_accuracy(holdout_scores: pd.DataFrame,
                           selection: pd.DataFrame) -> pd.DataFrame:
        if holdout_scores.empty:
            return pd.DataFrame()
        merged = holdout_scores.merge(selection[[Keys.SKU, "Selected_Model"]],
                                      on=Keys.SKU, how="inner")
        champ = merged[merged["Model"] == merged["Selected_Model"]]
        if champ.empty:
            return pd.DataFrame()
        agg = (champ.groupby(Keys.CATEGORY)
               .apply(lambda g: pd.Series({
                   "SKUs": g[Keys.SKU].nunique(),
                   "Median_MAPE": g["MAPE"].median(),
                   "Weighted_WAPE": (np.average(g["WAPE"].fillna(g["WAPE"].median()),
                                                weights=g["Total_Actual"].clip(lower=1))
                                     if g["Total_Actual"].sum() > 0 else g["WAPE"].median()),
                   # Volume-weighted, not a plain mean. Averaging per-SKU
                   # percentage bias lets a 20-unit SKU with +300% error
                   # outvote a 200,000-unit SKU at +2%, which produces
                   # category "findings" that are arithmetic artefacts.
                   "Weighted_Bias_Pct": (
                       (g["Bias"] * g["N_Obs"]).sum() / g["Total_Actual"].sum() * 100
                       if g["Total_Actual"].sum() > 0 else np.nan),
                   "Median_Bias_Pct": g["Bias_Pct"].median(),
                   "Median_FVA": g["FVA_vs_Naive"].median(),
                   "Total_Volume": g["Total_Actual"].sum(),
               }), include_groups=False)
               .reset_index())
        for col in agg.select_dtypes("float").columns:
            agg[col] = agg[col].round(3)
        return agg.sort_values("Weighted_WAPE")

    def _hierarchical_accuracy(self, holdout_pred: pd.DataFrame,
                               selection: pd.DataFrame,
                               panel: pd.DataFrame) -> pd.DataFrame:
        """
        Accuracy measured at every aggregation level a planning scorecard uses.

        This matters more than any single headline number. Forecast error
        cancels as you aggregate -- a 30% error on one SKU-week and a -28%
        error on the next net out at category level. Quoting only the
        SKU-week figure makes a good system look broken; quoting only the
        national-monthly figure makes a bad one look excellent. Real S&OP
        scorecards report the whole ladder, and so does this one.

        Levels, coarse to fine:
            Total Portfolio / 12-Week   -- the number in the board pack
            Category / Month            -- the S&OP demand review level
            Category / Week
            SKU / 12-Week Total         -- the replenishment-relevant level
            SKU / Week                  -- the hardest possible cut
        """
        if holdout_pred.empty or selection.empty:
            return pd.DataFrame()

        preds = holdout_pred.merge(selection[[Keys.SKU, "Selected_Model"]],
                                   on=Keys.SKU, how="inner")
        champ = preds[preds["Model"] == preds["Selected_Model"]].copy()
        naive = holdout_pred[holdout_pred["Model"] == "SeasonalNaive"].copy()
        if champ.empty:
            return pd.DataFrame()

        for frame in (champ, naive):
            frame["target_week_index"] = frame["origin_index"] + frame["horizon"]

        actuals = panel[[Keys.SKU, Keys.WEEK_INDEX, Keys.CATEGORY, Keys.CHANNEL,
                         "Sales_Units", Keys.WEEK]].rename(
            columns={Keys.WEEK_INDEX: "target_week_index", "Sales_Units": "Actual"})
        merged = champ.merge(actuals, on=[Keys.SKU, "target_week_index"], how="inner")
        if not naive.empty:
            merged = merged.merge(
                naive[[Keys.SKU, "target_week_index", "Forecast"]].rename(
                    columns={"Forecast": "Naive"}),
                on=[Keys.SKU, "target_week_index"], how="left")
        else:
            merged["Naive"] = np.nan
        if merged.empty:
            return pd.DataFrame()

        merged["Month"] = pd.to_datetime(merged[Keys.WEEK]).dt.to_period("M").astype(str)

        levels = [
            ("Total Portfolio / 12-Week", []),
            ("Category / Month", [Keys.CATEGORY, "Month"]),
            ("Category / Week", [Keys.CATEGORY, "target_week_index"]),
            ("Channel / Month", [Keys.CHANNEL, "Month"]),
            ("SKU / 12-Week Total", [Keys.SKU]),
            ("SKU / Week", [Keys.SKU, "target_week_index"]),
        ]

        rows: List[Dict[str, object]] = []
        for label, keys in levels:
            if keys:
                agg = (merged.groupby(keys, as_index=False)
                       .agg(Actual=("Actual", "sum"), Forecast=("Forecast", "sum"),
                            Naive=("Naive", "sum")))
            else:
                agg = pd.DataFrame([{
                    "Actual": merged["Actual"].sum(),
                    "Forecast": merged["Forecast"].sum(),
                    "Naive": merged["Naive"].sum()}])
            bundle = accuracy_bundle(agg["Actual"], agg["Forecast"],
                                     agg["Naive"] if agg["Naive"].notna().all() else None)
            rows.append({
                "Aggregation_Level": label,
                "Series_Count": int(len(agg)),
                "MAPE": round(bundle["MAPE"], 2) if np.isfinite(bundle["MAPE"]) else np.nan,
                "WAPE": round(bundle["WAPE"], 2) if np.isfinite(bundle["WAPE"]) else np.nan,
                "Bias_Pct": round(bundle["Bias_Pct"], 2)
                if np.isfinite(bundle["Bias_Pct"]) else np.nan,
                "MAE": round(bundle["MAE"], 1),
                "RMSE": round(bundle["RMSE"], 1),
                "FVA_vs_Naive": round(bundle["FVA_vs_Naive"], 2)
                if np.isfinite(bundle["FVA_vs_Naive"]) else np.nan,
                "Meets_15pct_Benchmark": bool(
                    np.isfinite(bundle["WAPE"]) and
                    bundle["WAPE"] < self.cfg.portfolio_mape_target),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _uncertainty_contract(forecast: pd.DataFrame, panel: pd.DataFrame,
                              sigma: pd.DataFrame, lam: float) -> pd.DataFrame:
        """
        The formal hand-off to Output 5 (Inventory Optimisation).

        Safety stock needs sigma_demand, not a point forecast. Publishing the
        forecast error standard deviation -- rather than the historical demand
        standard deviation -- is the technically correct input, because the
        buffer exists to absorb *forecast error*, not total demand variation.
        """
        if forecast.empty:
            return pd.DataFrame()
        agg = (forecast.groupby(Keys.SKU)
               .agg(Forecast_Units_12W=("Forecast_Units", "sum"),
                    Avg_Weekly_Forecast=("Forecast_Units", "mean"),
                    Selected_Model=("Selected_Model", "first"),
                    Supplier_ID=(Keys.SUPPLIER, "first"),
                    Warehouse_ID=(Keys.WAREHOUSE, "first"),
                    Category=(Keys.CATEGORY, "first"),
                    Channel=(Keys.CHANNEL, "first"),
                    Unit_Cost=("Unit_Cost", "first"),
                    Unit_Price=("Unit_Price", "first"))
               .reset_index())

        recent = panel[panel[Keys.WEEK_INDEX] >= panel[Keys.WEEK_INDEX].max() - 25]
        hist = (recent.groupby(Keys.SKU)["Sales_Units"]
                .agg(Historical_Demand_Std="std", Historical_Weekly_Mean="mean")
                .reset_index())

        out = agg.merge(hist, on=Keys.SKU, how="left").merge(sigma, on=Keys.SKU, how="left")
        out["Forecast_Error_Std"] = out["sigma_1"].fillna(
            out["Historical_Demand_Std"]).fillna(0.0).round(2)
        out["Demand_CV"] = (out["Historical_Demand_Std"] /
                            out["Historical_Weekly_Mean"].replace(0, np.nan)).round(4)
        out["Horizon_Variance_Lambda"] = round(lam, 4)
        out = out.drop(columns=["sigma_1"])
        for col in out.select_dtypes("float").columns:
            out[col] = out[col].round(3)
        return out


def run_demand_forecasting(config: AppConfig, sales: pd.DataFrame,
                           festivals: pd.DataFrame) -> ForecastResult:
    """Functional entry point used by the pipeline."""
    return DemandForecastEngine(config).run(sales, festivals)
