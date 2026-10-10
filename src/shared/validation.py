"""
Validation layer.

Two independent concerns, deliberately kept in one module because they answer
the same question -- "is this analysis trustworthy?":

1. **Temporal validation** -- rolling-origin splits that make random
   train/test shuffling structurally impossible, plus an explicit leakage
   assertion used by the test suite.
2. **Schema validation** -- contract checks on every generated dataset before
   any downstream module is allowed to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.shared.utils import get_logger

logger = get_logger("validation")


# --------------------------------------------------------------------------- #
# Temporal validation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TimeSplit:
    """One rolling-origin fold. Indices are positional into the series."""

    fold: int
    train_end: int          # exclusive
    test_start: int         # inclusive
    test_end: int           # exclusive
    label: str

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)

    @property
    def horizon(self) -> int:
        return self.test_end - self.test_start


def rolling_origin_splits(n_obs: int,
                          horizon: int,
                          n_folds: int,
                          min_train: int,
                          step: Optional[int] = None,
                          holdout: int = 0) -> List[TimeSplit]:
    """
    Generate expanding-window, rolling-origin validation folds.

    The last `holdout` observations are carved off first and are *never*
    returned as a fold.  That reserved block is the honest out-of-sample test
    set: model selection and ensemble weighting see only the folds, so the
    holdout score is not contaminated by the selection decision.

    Layout for n_obs=104, horizon=12, holdout=12, folds=2, step=12::

        |------------------ train -------------------|--f2--|
        |--------- train ----------|--f1--|
                                                            |-- holdout --|
        0                                                80     92      104

    Returns folds ordered oldest-origin first.  An empty list is returned when
    the series is too short, which callers must handle by falling back to the
    holdout-only evaluation path.
    """
    step = step or horizon
    usable = n_obs - holdout
    if usable < min_train + horizon or n_folds <= 0:
        return []

    splits: List[TimeSplit] = []
    # Latest possible origin sits so that the fold ends exactly at `usable`.
    for k in range(n_folds):
        test_end = usable - k * step
        test_start = test_end - horizon
        train_end = test_start
        if train_end < min_train:
            break
        splits.append(
            TimeSplit(
                fold=n_folds - k,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                label=f"CV{n_folds - k}",
            )
        )
    return sorted(splits, key=lambda s: s.train_end)


def holdout_split(n_obs: int, horizon: int) -> Optional[TimeSplit]:
    """The final untouched holdout block of length `horizon`."""
    if n_obs <= horizon:
        return None
    return TimeSplit(
        fold=0,
        train_end=n_obs - horizon,
        test_start=n_obs - horizon,
        test_end=n_obs,
        label="HOLDOUT",
    )


def assert_no_leakage(splits: Sequence[TimeSplit]) -> None:
    """Raise if any fold's training window overlaps its own test window."""
    for split in splits:
        if split.train_end > split.test_start:
            raise AssertionError(
                f"Temporal leakage in {split.label}: train ends at {split.train_end} "
                f"but test starts at {split.test_start}"
            )


def assert_feature_causality(frame: pd.DataFrame,
                             time_col: str,
                             feature_cols: Iterable[str],
                             target_col: str) -> None:
    """
    Guard against accidentally engineering a feature from the current or a
    future target value.

    A lag/rolling feature must, by construction, be perfectly correlated with
    nothing at lag 0.  We check the cheap, decisive case: no engineered feature
    may be numerically identical to the contemporaneous target.
    """
    ordered = frame.sort_values(time_col)
    target = ordered[target_col].to_numpy(dtype=float)
    for col in feature_cols:
        if col not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(target)
        if mask.sum() > 10 and np.allclose(values[mask], target[mask]):
            raise AssertionError(
                f"Feature '{col}' is identical to the target '{target_col}' -- "
                "this is future-data leakage."
            )


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

@dataclass
class ValidationIssue:
    dataset: str
    severity: str        # "ERROR" | "WARNING"
    message: str


class DataValidator:
    """Contract checks applied to every dataset before downstream consumption."""

    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []

    # -- primitives -------------------------------------------------------- #
    def _add(self, dataset: str, severity: str, message: str) -> None:
        self.issues.append(ValidationIssue(dataset, severity, message))

    def require_columns(self, df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            self._add(name, "ERROR", f"missing required columns: {missing}")

    def require_non_empty(self, df: pd.DataFrame, name: str, min_rows: int = 1) -> None:
        if len(df) < min_rows:
            self._add(name, "ERROR", f"expected >= {min_rows} rows, found {len(df)}")

    def require_unique(self, df: pd.DataFrame, name: str, key: str) -> None:
        if key in df.columns and df[key].duplicated().any():
            dupes = int(df[key].duplicated().sum())
            self._add(name, "ERROR", f"{dupes} duplicate values in key '{key}'")

    def require_no_nulls(self, df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
        for col in columns:
            if col in df.columns and df[col].isna().any():
                self._add(name, "ERROR",
                          f"column '{col}' has {int(df[col].isna().sum())} nulls")

    def require_non_negative(self, df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
        for col in columns:
            if col in df.columns:
                negatives = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
                if negatives:
                    self._add(name, "ERROR",
                              f"column '{col}' has {negatives} negative values")

    def require_referential_integrity(self, child: pd.DataFrame, child_name: str,
                                      child_key: str, parent_keys: Iterable) -> None:
        if child_key not in child.columns:
            return
        parent = set(parent_keys)
        orphans = set(child[child_key].dropna().unique()) - parent
        if orphans:
            sample = list(sorted(orphans))[:5]
            self._add(child_name, "ERROR",
                      f"{len(orphans)} orphan '{child_key}' values (e.g. {sample})")

    def warn_if(self, condition: bool, name: str, message: str) -> None:
        if condition:
            self._add(name, "WARNING", message)

    # -- reporting --------------------------------------------------------- #
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def is_valid(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["Dataset", "Severity", "Message"])
        return pd.DataFrame(
            [{"Dataset": i.dataset, "Severity": i.severity, "Message": i.message}
             for i in self.issues]
        )

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "\n  - ".join(f"[{i.dataset}] {i.message}" for i in self.errors)
            raise ValueError(f"Data validation failed:\n  - {detail}")

    def log_summary(self) -> None:
        if self.errors:
            for issue in self.errors:
                logger.error("VALIDATION [%s] %s", issue.dataset, issue.message)
        for issue in self.warnings:
            logger.warning("VALIDATION [%s] %s", issue.dataset, issue.message)
        if not self.issues:
            logger.info("Data validation passed with zero issues.")
        else:
            logger.info("Data validation: %d error(s), %d warning(s).",
                        len(self.errors), len(self.warnings))


def validate_datasets(datasets: Dict[str, pd.DataFrame]) -> DataValidator:
    """
    Apply the full dataset contract for the S&OP system.

    `datasets` is keyed by logical name (``sku_master``, ``sales_history``...).
    Missing datasets are reported as errors rather than raising, so a single
    run surfaces every problem at once instead of failing on the first.
    """
    v = DataValidator()
    required = [
        "sku_master", "supplier_master", "warehouse_master", "sales_history",
        "promotions", "inventory", "purchase_orders", "orders", "festivals",
    ]
    for name in required:
        if name not in datasets:
            v._add(name, "ERROR", "dataset not generated")

    sku = datasets.get("sku_master")
    if sku is not None:
        v.require_columns(sku, "sku_master",
                          ["SKU_ID", "Category", "Channel", "Supplier_ID",
                           "Warehouse_ID", "Unit_Cost", "Unit_Price"])
        v.require_unique(sku, "sku_master", "SKU_ID")
        v.require_non_negative(sku, "sku_master", ["Unit_Cost", "Unit_Price"])
        v.require_no_nulls(sku, "sku_master", ["SKU_ID", "Category", "Channel"])
        if {"Unit_Cost", "Unit_Price"}.issubset(sku.columns):
            v.warn_if(bool((sku["Unit_Price"] <= sku["Unit_Cost"]).any()),
                      "sku_master", "some SKUs priced at or below unit cost")

    sup = datasets.get("supplier_master")
    if sup is not None:
        v.require_columns(sup, "supplier_master",
                          ["Supplier_ID", "Lead_Time_Weeks", "MOQ",
                           "Weekly_Capacity_Units", "Reliability_Score"])
        v.require_unique(sup, "supplier_master", "Supplier_ID")
        v.require_non_negative(sup, "supplier_master",
                               ["Lead_Time_Weeks", "MOQ", "Weekly_Capacity_Units"])

    sales = datasets.get("sales_history")
    if sales is not None:
        v.require_columns(sales, "sales_history",
                          ["SKU_ID", "Week_Start", "Week_Index", "Demand_Units",
                           "Sales_Units", "Is_Promo", "Is_Stockout", "Price"])
        v.require_non_empty(sales, "sales_history", min_rows=1000)
        v.require_non_negative(sales, "sales_history",
                               ["Demand_Units", "Sales_Units", "Price"])
        v.require_no_nulls(sales, "sales_history", ["SKU_ID", "Week_Start"])
        if sku is not None:
            v.require_referential_integrity(sales, "sales_history", "SKU_ID",
                                            sku["SKU_ID"])
        if {"Sales_Units", "Demand_Units"}.issubset(sales.columns):
            violations = int((sales["Sales_Units"] > sales["Demand_Units"] + 1e-6).sum())
            if violations:
                v._add("sales_history", "ERROR",
                       f"{violations} rows where censored sales exceed true demand")

    orders = datasets.get("orders")
    if orders is not None:
        v.require_columns(orders, "orders",
                          ["Order_ID", "Supplier_ID", "SKU_ID", "Promised_Date",
                           "Actual_Delivery_Date", "Ordered_Qty", "Delivered_Qty",
                           "On_Time", "In_Full", "OTIF"])
        v.require_unique(orders, "orders", "Order_ID")
        v.require_non_negative(orders, "orders", ["Ordered_Qty", "Delivered_Qty"])
        if {"Delivered_Qty", "Ordered_Qty"}.issubset(orders.columns):
            v.warn_if(bool((orders["Delivered_Qty"] > orders["Ordered_Qty"] * 1.05).any()),
                      "orders", "over-delivery beyond 5% tolerance detected")

    inv = datasets.get("inventory")
    if inv is not None:
        v.require_columns(inv, "inventory",
                          ["SKU_ID", "Warehouse_ID", "On_Hand_Units", "Unit_Cost"])
        v.require_non_negative(inv, "inventory", ["On_Hand_Units"])

    pos = datasets.get("purchase_orders")
    if pos is not None:
        v.require_columns(pos, "purchase_orders",
                          ["PO_ID", "SKU_ID", "Supplier_ID", "Order_Qty",
                           "Expected_Receipt_Week"])
        v.require_unique(pos, "purchase_orders", "PO_ID")
        v.require_non_negative(pos, "purchase_orders", ["Order_Qty"])

    return v
