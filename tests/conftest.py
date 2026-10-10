"""
Shared pytest fixtures.

The heavy fixtures are session-scoped: generating data and fitting models once
and reusing the result keeps the suite fast enough to actually run on every
change, which is the only way a test suite earns its keep.

Tests run against a deliberately reduced portfolio (60 SKUs, 8-SKU competition
panel, 1 CV fold). The *logic* under test is scale-invariant -- MRP netting,
OTIF arithmetic, control limits and governance rules behave identically at 60
SKUs and at 500 -- so testing at full scale would buy nothing but minutes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before load_config() is first called.
os.environ.setdefault("SOP_N_SKUS", "60")
os.environ.setdefault("SOP_N_ORDERS", "1200")
os.environ.setdefault("SOP_PANEL_SIZE", "8")
os.environ.setdefault("SOP_CV_FOLDS", "1")

from src.planning.config import load_config                    # noqa: E402
from src.planning.consensus import run_consensus_planning      # noqa: E402
from src.planning.demand_engine import run_demand_forecasting  # noqa: E402
from src.planning.insights import generate_insights            # noqa: E402
from src.planning.kpi_tracker import run_otif_analysis         # noqa: E402
from src.planning.supply_planner import run_supply_planning    # noqa: E402
from src.shared.data_generator import generate_supply_chain_data  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def data(config):
    """The synthetic dataset, generated once and held in memory."""
    return generate_supply_chain_data(config, persist=False)


@pytest.fixture(scope="session")
def forecast(config, data):
    return run_demand_forecasting(config, data.sales_history, data.festivals)


@pytest.fixture(scope="session")
def consensus(config, forecast, data):
    return run_consensus_planning(
        config, forecast.forecast, data.sku_master,
        data.supplier_master, data.warehouse_master)


@pytest.fixture(scope="session")
def supply(config, consensus, forecast, data):
    return run_supply_planning(
        config,
        consensus_plan=consensus.consensus_plan,
        forecast_uncertainty=forecast.uncertainty,
        sku_master=data.sku_master,
        supplier_master=data.supplier_master,
        warehouse_master=data.warehouse_master,
        inventory=data.inventory,
        purchase_orders=data.purchase_orders,
        sales_history=data.sales_history)


@pytest.fixture(scope="session")
def kpi(config, data):
    return run_otif_analysis(config, data.orders, data.supplier_master)


@pytest.fixture(scope="session")
def insights(config, forecast, consensus, supply, kpi):
    return generate_insights(config, forecast_result=forecast,
                             consensus_result=consensus,
                             supply_result=supply, kpi_result=kpi)
