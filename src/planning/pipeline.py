"""
Pipeline orchestrator.

Runs the full S&OP workflow end to end and returns every intermediate result,
so ``run_demo.py``, the test suite and the dashboard all execute exactly the
same code path. Anything that only works when driven from the demo script is
not really working.

Flow::

    Generate -> Validate -> Forecast -> Consensus -> Supply Plan
             -> OTIF/SPC -> Insights -> Excel + CSV -> Run Manifest
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.planning.config import (AppConfig, OUTPUT_DIR, PROCESSED_DIR, Keys,
                                 ensure_directories, load_config)
from src.planning.consensus import run_consensus_planning
from src.planning.demand_engine import dependency_report, run_demand_forecasting
from src.planning.insights import generate_insights
from src.planning.kpi_tracker import run_otif_analysis
from src.planning.saop_pack import build_reports
from src.planning.supply_planner import run_supply_planning
from src.shared.data_generator import SupplyChainData, generate_supply_chain_data
from src.shared.utils import get_logger, load_csv, timed
from src.shared.validation import validate_datasets

logger = get_logger("pipeline")

FORECAST_CACHE = OUTPUT_DIR / "cache" / "forecast_result.pkl"


def _cache_forecast(result) -> None:
    """
    Persist the fitted forecast result.

    Forecasting is by far the most expensive stage -- minutes, against seconds
    for everything downstream. Caching it means iterating on consensus rules,
    supply logic or report formatting does not require refitting 500 SKUs, and
    it makes the expensive stage restartable on constrained hardware.
    """
    try:
        FORECAST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with FORECAST_CACHE.open("wb") as handle:
            pickle.dump(result, handle)
        logger.info("Forecast cached to %s", FORECAST_CACHE)
    except Exception as exc:                                   # pragma: no cover
        logger.warning("Could not cache forecast result: %s", exc)


def _load_cached_forecast():
    """Reload a cached forecast, or return None if absent or unreadable."""
    if not FORECAST_CACHE.exists():
        return None
    try:
        with FORECAST_CACHE.open("rb") as handle:
            result = pickle.load(handle)
        logger.info("Reusing cached forecast from %s (%d SKUs). "
                    "Delete the file or omit --reuse-forecast to refit.",
                    FORECAST_CACHE.name, result.diagnostics.get("n_skus", 0))
        return result
    except Exception as exc:                                   # pragma: no cover
        logger.warning("Cached forecast unreadable (%s); refitting.", exc)
        return None


@dataclass
class PipelineResult:
    """Everything the pipeline produced, in one object."""

    data: SupplyChainData
    forecast: object
    consensus: object
    supply: object
    kpi: object
    insights: pd.DataFrame
    reports: Dict[str, object] = field(default_factory=dict)
    manifest: Dict[str, object] = field(default_factory=dict)


def _load_existing_data(config: AppConfig) -> Optional[SupplyChainData]:
    """Reload a previously generated dataset from data/raw, if complete."""
    names = ["sku_master", "supplier_master", "warehouse_master", "sales_history",
             "promotions", "price_history", "inventory", "purchase_orders",
             "orders", "festivals", "supplier_capacity"]
    try:
        frames = {}
        for name in names:
            date_cols = {
                "sales_history": [Keys.WEEK],
                "price_history": [Keys.WEEK],
                "promotions": [Keys.WEEK],
                "festivals": [Keys.WEEK],
                "orders": ["Promised_Date", "Actual_Delivery_Date"],
                "purchase_orders": ["Order_Date", "Expected_Receipt_Date"],
                "inventory": ["Snapshot_Week"],
            }.get(name)
            frames[name] = load_csv(f"{name}.csv", parse_dates=date_cols)
        return SupplyChainData(**frames)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        logger.info("Could not reuse cached data (%s); regenerating.", type(exc).__name__)
        return None


def run_pipeline(config: Optional[AppConfig] = None,
                 scenario: Optional[str] = None,
                 regenerate_data: bool = True,
                 export_reports: bool = True,
                 reuse_forecast: bool = False,
                 forecast_only: bool = False,
                 split_workbooks: bool = False,
                 full_master_data: bool = False) -> PipelineResult:
    """Execute the complete S&OP pipeline."""
    config = config or load_config()
    scenario = scenario or config.consensus.default_scenario
    ensure_directories()
    started = time.perf_counter()

    logger.info("=" * 78)
    logger.info("S&OP INTELLIGENCE SYSTEM  |  scenario=%s  |  seed=%d", scenario, config.seed)
    logger.info("Optional engines available: %s", dependency_report())
    logger.info("=" * 78)

    # -- 1. Data ------------------------------------------------------------ #
    with timed("Stage 1/7  Synthetic data generation", logger):
        data = None if regenerate_data else _load_existing_data(config)
        if data is None:
            data = generate_supply_chain_data(config, persist=True)

    with timed("Stage 2/7  Data validation", logger):
        validator = validate_datasets(data.as_dict())
        validator.log_summary()
        validator.raise_for_errors()
        validation_report = validator.to_frame()

    # -- 2. Demand ---------------------------------------------------------- #
    forecast = _load_cached_forecast() if reuse_forecast else None
    if forecast is None:
        with timed("Stage 3/7  Demand forecasting (4 models + ensemble)", logger):
            forecast = run_demand_forecasting(config, data.sales_history, data.festivals)
        _cache_forecast(forecast)
    else:
        logger.info("Stage 3/7  Demand forecasting  [SKIPPED - using cache]")

    if forecast_only:
        logger.info("forecast_only requested; stopping after the demand stage.")
        return PipelineResult(data=data, forecast=forecast, consensus=None,
                              supply=None, kpi=None, insights=pd.DataFrame())

    # -- 3. Consensus ------------------------------------------------------- #
    with timed("Stage 4/7  Consensus S&OP planning", logger):
        consensus = run_consensus_planning(
            config, forecast.forecast, data.sku_master,
            data.supplier_master, data.warehouse_master, scenario=scenario)

    # -- 4. Supply ---------------------------------------------------------- #
    with timed("Stage 5/7  Supply planning & constraint analysis", logger):
        supply = run_supply_planning(
            config,
            consensus_plan=consensus.consensus_plan,
            forecast_uncertainty=forecast.uncertainty,
            sku_master=data.sku_master,
            supplier_master=data.supplier_master,
            warehouse_master=data.warehouse_master,
            inventory=data.inventory,
            purchase_orders=data.purchase_orders,
            sales_history=data.sales_history,
            scenario=scenario)

    # -- 5. Service --------------------------------------------------------- #
    with timed("Stage 6/7  OTIF measurement & Six Sigma SPC", logger):
        kpi = run_otif_analysis(config, data.orders, data.supplier_master)

    # -- 6. Insights + reports ---------------------------------------------- #
    with timed("Stage 7/7  Executive insights & report generation", logger):
        insights = generate_insights(
            config, forecast_result=forecast, consensus_result=consensus,
            supply_result=supply, kpi_result=kpi)

        reports: Dict[str, object] = {}
        if export_reports:
            reports = build_reports(config, forecast, consensus, supply, kpi,
                                    insights, split_workbooks=split_workbooks,
                                    data=data,
                                    validation_report=validation_report,
                                    full_master_data=full_master_data)
            validation_report.to_csv(PROCESSED_DIR / "validation_report.csv", index=False)

            from src.planning.tableau_export import export_for_tableau
            try:
                reports["tableau"] = export_for_tableau(
                    config, data, forecast, consensus, supply, kpi, insights)
            except Exception as exc:
                logger.warning("Tableau export failed: %s", exc)

    elapsed = time.perf_counter() - started
    manifest = _build_manifest(config, data, forecast, consensus, supply, kpi,
                               insights, reports, elapsed, scenario)
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 78)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("=" * 78)

    return PipelineResult(data=data, forecast=forecast, consensus=consensus,
                          supply=supply, kpi=kpi, insights=insights,
                          reports=reports, manifest=manifest)


def _build_manifest(config, data, forecast, consensus, supply, kpi, insights,
                    reports, elapsed, scenario) -> Dict[str, object]:
    """
    Machine-readable record of exactly what this run produced.

    Written to outputs/run_manifest.json so a reviewer can verify the headline
    claims without re-running anything, and so a regression in results is
    visible as a diff rather than requiring a side-by-side rerun.
    """
    hierarchy = forecast.hierarchy_accuracy
    review = (hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
              if not hierarchy.empty else pd.DataFrame())

    return {
        "run_timestamp": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "seed": config.seed,
        "scenario": scenario,
        "dependencies": dependency_report(),
        "datasets": data.summary().to_dict("records"),
        "forecast": {
            "skus": forecast.diagnostics["n_skus"],
            "weeks_history": forecast.diagnostics["n_weeks"],
            "cv_folds": forecast.diagnostics["cv_folds"],
            "competition_panel": forecast.diagnostics["competition_panel_size"],
            "models": sorted(forecast.model_scores["Model"].unique().tolist())
            if not forecast.model_scores.empty else [],
            "champion_mix": (forecast.sku_accuracy["Selected_Model"]
                             .value_counts().to_dict()
                             if not forecast.sku_accuracy.empty else {}),
            "holdout_median_sku_mape": forecast.diagnostics["portfolio_holdout_mape"],
            "holdout_weighted_wape": forecast.diagnostics["portfolio_holdout_wape"],
            "review_level_wape": (float(review.iloc[0]["WAPE"])
                                  if not review.empty else None),
            "interval_coverage": forecast.diagnostics["interval_coverage"],
            "accuracy_by_level": (hierarchy.to_dict("records")
                                  if not hierarchy.empty else []),
        },
        "consensus": consensus.diagnostics,
        "supply": supply.diagnostics,
        "service": {k: v for k, v in kpi.diagnostics.items()
                    if k != "failure_mode_mix"},
        "insights": {
            "total": int(len(insights)),
            "by_severity": (insights["Severity"].value_counts().to_dict()
                            if not insights.empty else {}),
        },
        "outputs": {
            "workbook": str(reports.get("workbook", "")),
            "workbooks": [],
            "csv_exports": len(reports.get("csv_exports", [])),
            "tableau_tables": (reports.get("tableau") or {}).get("tables", 0),
            "tableau_rows": (reports.get("tableau") or {}).get("rows", 0),
        },
    }
