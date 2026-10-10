"""
Central configuration for the S&OP Intelligence System.

Every business rule, threshold and model hyper-parameter used anywhere in the
pipeline is declared here.  Nothing downstream is allowed to hard-code a
business number -- if you find a magic constant in a module, it belongs in this
file.

The configuration is exposed as frozen dataclasses so that it is immutable at
runtime and trivially serialisable into the audit trail of every run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
LOG_DIR: Path = PROJECT_ROOT / "outputs" / "logs"


def ensure_directories() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for directory in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = 20260215


# --------------------------------------------------------------------------- #
# Data-generation configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataConfig:
    """Shape and behaviour of the synthetic supply-chain dataset."""

    n_skus: int = 500
    n_suppliers: int = 40
    n_warehouses: int = 5
    n_orders: int = 5_000

    history_weeks: int = 104          # 24 months of weekly observations
    history_end: str = "2026-08-09"   # last completed ISO week (Sunday)

    categories: Tuple[str, ...] = (
        "Beverages",
        "Snacks",
        "Dairy",
        "Personal Care",
        "Home Care",
        "Packaged Foods",
        "Confectionery",
        "Health & Wellness",
    )

    channels: Tuple[str, ...] = (
        "Modern Trade",
        "General Trade",
        "E-Commerce",
        "Quick Commerce",
        "HoReCa / B2B",
    )

    # Channel share of total volume (must sum to 1.0)
    channel_mix: Tuple[float, ...] = (0.27, 0.34, 0.17, 0.12, 0.10)

    # Annualised organic trend by channel (compounding, applied weekly)
    channel_trend: Dict[str, float] = field(
        default_factory=lambda: {
            "Modern Trade": 0.06,
            "General Trade": 0.01,
            "E-Commerce": 0.24,
            "Quick Commerce": 0.42,
            "HoReCa / B2B": 0.09,
        }
    )

    # Coefficient of variation of weekly noise by channel
    channel_noise_cv: Dict[str, float] = field(
        default_factory=lambda: {
            "Modern Trade": 0.16,
            "General Trade": 0.13,
            "E-Commerce": 0.24,
            "Quick Commerce": 0.33,
            "HoReCa / B2B": 0.21,
        }
    )

    warehouses: Tuple[Tuple[str, str, str], ...] = (
        ("WH_MUM", "Bhiwandi DC", "West"),
        ("WH_DEL", "Bilaspur NCR DC", "North"),
        ("WH_BLR", "Hoskote DC", "South"),
        ("WH_KOL", "Dankuni DC", "East"),
        ("WH_AHM", "Sanand DC", "West"),
    )

    # Seasonal amplitude of the annual sine wave, by category
    seasonal_amplitude: Dict[str, float] = field(
        default_factory=lambda: {
            "Beverages": 0.38,
            "Snacks": 0.14,
            "Dairy": 0.11,
            "Personal Care": 0.09,
            "Home Care": 0.12,
            "Packaged Foods": 0.10,
            "Confectionery": 0.20,
            "Health & Wellness": 0.17,
        }
    )

    # Week-of-year at which the category peaks (1-52)
    seasonal_peak_week: Dict[str, int] = field(
        default_factory=lambda: {
            "Beverages": 20,      # pre-monsoon summer peak
            "Snacks": 44,         # festive
            "Dairy": 42,
            "Personal Care": 43,
            "Home Care": 41,      # Diwali deep-clean
            "Packaged Foods": 45,
            "Confectionery": 44,
            "Health & Wellness": 3,   # new-year resolutions
        }
    )

    # Multiplicative festive lift by category (applied to festival intensity)
    festive_sensitivity: Dict[str, float] = field(
        default_factory=lambda: {
            "Beverages": 0.55,
            "Snacks": 0.70,
            "Dairy": 0.45,
            "Personal Care": 0.85,
            "Home Care": 0.75,
            "Packaged Foods": 0.60,
            "Confectionery": 1.20,
            "Health & Wellness": 0.30,
        }
    )

    promo_weeks_share: float = 0.13       # share of SKU-weeks under promotion
    promo_uplift_range: Tuple[float, float] = (1.18, 2.30)
    price_elasticity_range: Tuple[float, float] = (-2.4, -0.8)

    stockout_week_share: float = 0.045    # share of SKU-weeks with lost sales
    stockout_severity_range: Tuple[float, float] = (0.30, 0.90)

    npd_share: float = 0.06               # share of SKUs launched mid-history
    discontinued_share: float = 0.03      # share of SKUs phased out


# --------------------------------------------------------------------------- #
# Forecasting configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForecastConfig:
    """Model, validation and ensemble settings for the demand engine."""

    horizon_weeks: int = 12               # planning horizon / final holdout size
    seasonal_period: int = 52

    # Rolling-origin cross-validation used for MODEL SELECTION and ENSEMBLE
    # WEIGHTS. The final 12-week holdout is never used for either, so reported
    # holdout accuracy stays an honest out-of-sample estimate.
    cv_folds: int = 2
    cv_step_weeks: int = 12
    min_train_weeks: int = 60

    # Statistical models are fitted per SKU; SARIMA and Prophet are expensive,
    # so they run on a stratified "model competition" panel while the fast
    # models run across the full 500-SKU portfolio. Set to 0 to disable a tier.
    competition_panel_size: int = 24
    mass_tier_models: Tuple[str, ...] = ("HoltWinters", "XGBoost")
    panel_tier_models: Tuple[str, ...] = ("HoltWinters", "SARIMA", "Prophet", "XGBoost")

    # Ensemble
    ensemble_metric: str = "MAPE"
    min_ensemble_weight: float = 0.02     # models below this are dropped
    mape_floor: float = 1.0               # guards 1/MAPE against divide-by-zero

    # Prediction intervals
    interval_levels: Tuple[float, float] = (0.80, 0.95)
    horizon_variance_growth: float = 0.15  # lambda in sigma_h = sigma_1*sqrt(1+l(h-1))

    # XGBoost (global model across the whole panel)
    xgb_lags: Tuple[int, ...] = (1, 2, 4, 8, 12, 26, 52)
    xgb_rolling_windows: Tuple[int, ...] = (4, 8, 13, 26)
    xgb_ewm_spans: Tuple[int, ...] = (4, 13)
    fourier_orders: int = 3
    xgb_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 350,
            "learning_rate": 0.045,
            "max_depth": 7,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 1.6,
            "reg_alpha": 0.15,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
        }
    )

    shap_sample_skus: int = 6
    shap_background_rows: int = 800

    portfolio_mape_target: float = 15.0   # reporting benchmark only, never forced


# --------------------------------------------------------------------------- #
# Consensus planning configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConsensusConfig:
    """Rules governing the commercial / operational consensus workflow."""

    override_alert_threshold_pct: float = 20.0    # |dev| above this => alert
    override_escalation_pct: float = 45.0         # requires S&OP board sign-off
    share_of_skus_with_commercial_input: float = 0.42

    # Scenario blend weights: how much of the *commercial delta* is accepted
    scenario_commercial_weight: Dict[str, float] = field(
        default_factory=lambda: {
            "Conservative": 0.35,
            "Balanced": 0.70,
            "Aggressive": 1.00,
        }
    )
    # Multiplier applied on top of the blended plan (risk appetite)
    scenario_bias: Dict[str, float] = field(
        default_factory=lambda: {
            "Conservative": 0.96,
            "Balanced": 1.00,
            "Aggressive": 1.05,
        }
    )
    # Whether the operational ceiling is binding in the scenario
    scenario_respect_constraints: Dict[str, bool] = field(
        default_factory=lambda: {
            "Conservative": True,
            "Balanced": True,
            "Aggressive": True,
        }
    )
    default_scenario: str = "Balanced"

    commercial_drivers: Tuple[str, ...] = (
        "Sales Override",
        "Promotion Uplift",
        "New Product Introduction",
        "Customer Commitment",
        "Market Opportunity",
        "Cannibalisation Adjustment",
    )


# --------------------------------------------------------------------------- #
# Supply planning configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupplyConfig:
    """MRP, safety-stock and risk-scoring parameters."""

    horizon_weeks: int = 12

    # Service-level targets by ABC class -> Z score
    service_level_by_class: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.99, "B": 0.95, "C": 0.90}
    )
    abc_thresholds: Tuple[float, float] = (0.70, 0.90)   # cumulative revenue

    holding_cost_rate: float = 0.22        # annual, % of unit cost
    ordering_cost: float = 4_500.0         # INR per purchase order
    weeks_per_year: int = 52

    target_weeks_of_cover: float = 4.0
    excess_cover_weeks: float = 8.0        # above this => excess / slow-moving
    critical_cover_weeks: float = 1.5

    capacity_alert_threshold: float = 0.95   # utilisation triggering an alert
    capacity_breach_threshold: float = 1.00

    # Supply Risk Index - weights must sum to 1.0
    risk_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "stockout_proximity": 0.32,
            "capacity_utilisation": 0.22,
            "weeks_of_cover": 0.18,
            "lead_time_risk": 0.14,
            "forecast_uncertainty": 0.14,
        }
    )
    risk_amber_threshold: float = 35.0
    risk_red_threshold: float = 65.0

    max_lead_time_weeks_reference: float = 8.0   # normaliser for lead-time risk


# --------------------------------------------------------------------------- #
# OTIF / SPC configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KPIConfig:
    """Service-level measurement and Six Sigma control-chart settings."""

    otif_world_class: float = 95.0
    otif_acceptable: float = 90.0

    in_full_tolerance: float = 0.98        # >=98% of ordered qty counts in-full
    on_time_grace_days: int = 0            # days of grace on promised date

    spc_sigma_multiplier: float = 3.0      # UCL/LCL band
    spc_alert_sigma: float = 2.0           # >2 sigma below mean => special cause
    spc_min_points: int = 8

    root_causes: Tuple[str, ...] = (
        "Supplier Delay",
        "Warehouse Delay",
        "Transportation Delay",
        "Inventory Shortage",
        "Forecast Error",
        "Quality Issue",
        "Documentation Issue",
    )


# --------------------------------------------------------------------------- #
# Reporting / dashboard configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReportConfig:
    """Excel pack and dashboard presentation settings."""

    currency_symbol: str = "\u20b9"        # INR
    company_label: str = "S&OP Intelligence System"
    subtitle: str = "Synthetic / simulated data - portfolio demonstration"

    brand_navy: str = "1F3864"
    brand_blue: str = "2E75B6"
    brand_light: str = "D9E2F3"
    brand_green: str = "0F7B4F"
    brand_amber: str = "B7791F"
    brand_red: str = "9C2B2B"
    brand_grey: str = "F2F2F2"

    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050

    top_n_insights: int = 12


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object passed through the pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    supply: SupplyConfig = field(default_factory=SupplyConfig)
    kpi: KPIConfig = field(default_factory=KPIConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    seed: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_config() -> AppConfig:
    """
    Build the application configuration.

    Environment overrides (useful for CI / quick smoke runs):
        SOP_N_SKUS, SOP_PANEL_SIZE, SOP_CV_FOLDS, SOP_SEED
    """
    data_kwargs: Dict[str, object] = {}
    fc_kwargs: Dict[str, object] = {}

    if os.getenv("SOP_N_SKUS"):
        data_kwargs["n_skus"] = int(os.environ["SOP_N_SKUS"])
    if os.getenv("SOP_N_ORDERS"):
        data_kwargs["n_orders"] = int(os.environ["SOP_N_ORDERS"])
    if os.getenv("SOP_PANEL_SIZE"):
        fc_kwargs["competition_panel_size"] = int(os.environ["SOP_PANEL_SIZE"])
    if os.getenv("SOP_CV_FOLDS"):
        fc_kwargs["cv_folds"] = int(os.environ["SOP_CV_FOLDS"])

    seed = int(os.getenv("SOP_SEED", RANDOM_SEED))

    return AppConfig(
        data=DataConfig(**data_kwargs),
        forecast=ForecastConfig(**fc_kwargs),
        seed=seed,
    )


# Canonical column names shared across Output 1 / 2 / 5 of the
# Supply Chain Command Center. Changing these breaks cross-output integration.
class Keys:
    SKU = "SKU_ID"
    SUPPLIER = "Supplier_ID"
    WAREHOUSE = "Warehouse_ID"
    ORDER = "Order_ID"
    PO = "PO_ID"
    CATEGORY = "Category"
    CHANNEL = "Channel"
    WEEK = "Week_Start"
    WEEK_INDEX = "Week_Index"
