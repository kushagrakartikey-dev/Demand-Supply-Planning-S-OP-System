"""
Module 4 -- OTIF & Six Sigma KPI Engine.

Measures inbound service performance and applies statistical process control
to separate signal from noise.

Why SPC and not a threshold
---------------------------
Most service dashboards colour a month red the moment OTIF drops below target.
That is a category error. A process with a true mean of 92% and a standard
deviation of 2% will produce a month at 89% roughly one time in twenty *purely
by chance*, and reacting to it -- launching a supplier review, changing a
process -- is what Deming called tampering: it adds variation rather than
removing it.

Control charts answer the only question worth asking: is this month different
because the process changed, or because processes vary? Only the first
deserves a corrective action, and only the first is flagged here.

Sigma is estimated from the average moving range (MRbar / d2, d2 = 1.128),
not from the raw standard deviation. This is deliberate and it matters: the
raw standard deviation is inflated by the very special causes the chart exists
to detect, which widens the limits and hides the signal. The moving-range
estimator captures short-term common-cause variation only.

Western Electric rules
----------------------
Beyond the 3-sigma limits, four supplementary rules detect patterns that are
individually inside the limits but collectively improbable -- a slow drift or a
sustained shift that a limit test alone would miss for months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, Keys, KPIConfig
from src.shared.metrics import control_limits, otif_rate
from src.shared.utils import get_logger, safe_div

logger = get_logger("kpi_tracker")


@dataclass
class KPIResult:
    """Outputs of the OTIF / SPC engine."""

    order_detail: pd.DataFrame
    otif_by_supplier: pd.DataFrame
    otif_by_category: pd.DataFrame
    otif_by_channel: pd.DataFrame
    otif_by_month: pd.DataFrame
    otif_by_sku: pd.DataFrame
    spc_chart: pd.DataFrame
    spc_violations: pd.DataFrame
    root_cause_analysis: pd.DataFrame
    supplier_scorecard: pd.DataFrame
    diagnostics: Dict[str, object]


class OTIFTracker:
    """Computes OTIF across every dimension and runs the control chart."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: KPIConfig = config.kpi

    # ------------------------------------------------------------------ #
    # Order-level preparation
    # ------------------------------------------------------------------ #
    def prepare_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute the service flags from raw dates and quantities.

        The flags are deliberately *derived here* rather than trusted from the
        source file. In a real deployment the ERP's own OTIF flag is frequently
        computed against a revised promise date rather than the original one,
        which quietly launders lateness out of the metric. Recomputing from
        promised date and ordered quantity is the auditable version.
        """
        cfg = self.cfg
        work = orders.copy()
        work["Promised_Date"] = pd.to_datetime(work["Promised_Date"])
        work["Actual_Delivery_Date"] = pd.to_datetime(work["Actual_Delivery_Date"])

        work["Delay_Days"] = (work["Actual_Delivery_Date"]
                              - work["Promised_Date"]).dt.days
        work["On_Time"] = work["Delay_Days"] <= cfg.on_time_grace_days
        work["Fill_Rate"] = (work["Delivered_Qty"]
                             / work["Ordered_Qty"].replace(0, np.nan)).fillna(0.0)
        work["In_Full"] = work["Fill_Rate"] >= cfg.in_full_tolerance
        work["OTIF"] = work["On_Time"] & work["In_Full"]

        work["Failure_Mode"] = np.select(
            [work["OTIF"],
             ~work["On_Time"] & ~work["In_Full"],
             ~work["On_Time"] & work["In_Full"],
             work["On_Time"] & ~work["In_Full"]],
            ["OTIF", "Late and Short", "Late Only", "Short Only"],
            default="OTIF")

        work["Order_Month"] = work["Promised_Date"].dt.to_period("M").astype(str)
        work["Order_Week"] = work["Promised_Date"].dt.to_period("W").astype(str)
        work["Shortfall_Units"] = np.clip(
            work["Ordered_Qty"] - work["Delivered_Qty"], 0, None)
        return work

    # ------------------------------------------------------------------ #
    # Dimensional OTIF
    # ------------------------------------------------------------------ #
    def _otif_by(self, orders: pd.DataFrame, dimension: str | List[str],
                 min_orders: int = 1) -> pd.DataFrame:
        """
        OTIF cut by any dimension, with an explicit order-count column.

        The count is not decoration. A supplier at 100% OTIF on three orders
        and one at 94% on four hundred are not comparable, and a scorecard that
        ranks them side by side without showing volume actively misleads. Cuts
        below `min_orders` are marked rather than silently dropped.
        """
        keys = [dimension] if isinstance(dimension, str) else list(dimension)
        grouped = (orders.groupby(keys, as_index=False)
                   .agg(Orders=(Keys.ORDER, "count"),
                        OTIF_Orders=("OTIF", "sum"),
                        On_Time_Orders=("On_Time", "sum"),
                        In_Full_Orders=("In_Full", "sum"),
                        Avg_Delay_Days=("Delay_Days", "mean"),
                        Max_Delay_Days=("Delay_Days", "max"),
                        Avg_Fill_Rate=("Fill_Rate", "mean"),
                        Ordered_Units=("Ordered_Qty", "sum"),
                        Delivered_Units=("Delivered_Qty", "sum"),
                        Shortfall_Units=("Shortfall_Units", "sum"),
                        Order_Value=("Order_Value", "sum")))

        grouped["OTIF_Pct"] = (grouped["OTIF_Orders"] / grouped["Orders"] * 100).round(2)
        grouped["On_Time_Pct"] = (grouped["On_Time_Orders"] / grouped["Orders"] * 100).round(2)
        grouped["In_Full_Pct"] = (grouped["In_Full_Orders"] / grouped["Orders"] * 100).round(2)
        grouped["Volume_Fill_Rate_Pct"] = (
            grouped["Delivered_Units"] / grouped["Ordered_Units"].replace(0, np.nan)
            * 100).round(2)
        grouped["Avg_Delay_Days"] = grouped["Avg_Delay_Days"].round(2)
        grouped["Avg_Fill_Rate"] = grouped["Avg_Fill_Rate"].round(4)

        grouped["Benchmark"] = np.select(
            [grouped["OTIF_Pct"] >= self.cfg.otif_world_class,
             grouped["OTIF_Pct"] >= self.cfg.otif_acceptable],
            ["World Class", "Acceptable"], default="At Risk")
        grouped["Statistically_Reliable"] = grouped["Orders"] >= max(min_orders, 20)
        return grouped.sort_values("OTIF_Pct").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Statistical process control
    # ------------------------------------------------------------------ #
    def build_control_chart(self, monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Individuals control chart on monthly OTIF, plus Western Electric tests.

        Returns (chart, violations).
        """
        cfg = self.cfg
        chart = monthly.sort_values("Order_Month").reset_index(drop=True).copy()
        if len(chart) < cfg.spc_min_points:
            logger.warning("Only %d monthly points -- below the %d needed for stable "
                           "control limits. Chart produced but treat limits as indicative.",
                           len(chart), cfg.spc_min_points)

        series = chart["OTIF_Pct"].to_numpy(dtype=float)
        limits = control_limits(series, cfg.spc_sigma_multiplier)

        centre, sigma = limits["mean"], limits["sigma"]
        chart["Center_Line"] = round(centre, 3)
        chart["Sigma"] = round(sigma, 4)
        for k in (1, 2, 3):
            chart[f"UCL_{k}Sigma"] = round(centre + k * sigma, 3)
            chart[f"LCL_{k}Sigma"] = round(centre - k * sigma, 3)
        # OTIF is a bounded percentage, so a computed UCL above 100% or an LCL
        # below 0% is not attainable. The raw sigma-based limits are retained
        # in the *_3Sigma columns for statistical transparency; the UCL/LCL
        # columns used for charting and alerting are clipped to the feasible
        # range. Clipping the display without clipping the underlying sigma
        # keeps the process-capability arithmetic intact.
        chart["UCL"] = chart["UCL_3Sigma"].clip(upper=100.0)
        chart["LCL"] = chart["LCL_3Sigma"].clip(lower=0.0)
        chart["UCL_Truncated_At_Bound"] = chart["UCL_3Sigma"] > 100.0

        chart["Sigma_Deviation"] = (
            (chart["OTIF_Pct"] - centre) / sigma if sigma > 1e-9 else 0.0).round(3)

        # Moving range chart -- detects instability in variation itself.
        chart["Moving_Range"] = chart["OTIF_Pct"].diff().abs().round(3)
        mr_bar = float(chart["Moving_Range"].mean(skipna=True) or 0.0)
        chart["MR_Bar"] = round(mr_bar, 3)
        chart["MR_UCL"] = round(3.267 * mr_bar, 3)   # D4 constant for n = 2

        violations = self._western_electric(chart, centre, sigma)
        chart["Special_Cause"] = chart.index.isin(
            violations["Point_Index"].unique()) if not violations.empty else False

        # The project's explicit business rule, separate from the WE tests.
        chart["Below_2Sigma_Alert"] = chart["Sigma_Deviation"] <= -cfg.spc_alert_sigma
        chart["Process_State"] = np.select(
            [chart["Special_Cause"] | chart["Below_2Sigma_Alert"]],
            ["Special Cause - Investigate"], default="In Control")
        return chart, violations

    def _western_electric(self, chart: pd.DataFrame, centre: float,
                          sigma: float) -> pd.DataFrame:
        """
        The four classical Western Electric zone tests.

        Rule 1 catches a shock. Rules 2-4 catch the far more dangerous case:
        a process that has quietly shifted and is now producing a run of
        points that are each individually unremarkable.
        """
        if sigma <= 1e-9 or chart.empty:
            return pd.DataFrame(columns=["Point_Index", "Order_Month", "Rule",
                                         "Description", "OTIF_Pct"])

        values = chart["OTIF_Pct"].to_numpy(dtype=float)
        z = (values - centre) / sigma
        months = chart["Order_Month"].to_numpy()
        found: List[Dict[str, object]] = []

        def record(idx: int, rule: str, description: str) -> None:
            found.append({
                "Point_Index": int(idx),
                "Order_Month": months[idx],
                "Rule": rule,
                "Description": description,
                "OTIF_Pct": round(float(values[idx]), 2),
                "Sigma_Deviation": round(float(z[idx]), 2),
            })

        # Rule 1: any point beyond 3 sigma.
        for i in np.where(np.abs(z) > 3)[0]:
            record(i, "WE Rule 1", "Point beyond 3-sigma control limit")

        # Rule 2: nine consecutive points on the same side of the centre line.
        for i in range(8, len(z)):
            window = z[i - 8:i + 1]
            if np.all(window > 0) or np.all(window < 0):
                side = "above" if window[0] > 0 else "below"
                record(i, "WE Rule 2", f"Nine consecutive points {side} the centre line")

        # Rule 3: six consecutive points steadily increasing or decreasing.
        for i in range(5, len(z)):
            window = values[i - 5:i + 1]
            diffs = np.diff(window)
            if np.all(diffs > 0) or np.all(diffs < 0):
                direction = "increasing" if diffs[0] > 0 else "decreasing"
                record(i, "WE Rule 3", f"Six consecutive points {direction} (trend)")

        # Rule 4: two of three consecutive points beyond 2 sigma, same side.
        for i in range(2, len(z)):
            window = z[i - 2:i + 1]
            for sign, label in ((1, "above"), (-1, "below")):
                if np.sum(sign * window > 2) >= 2 and sign * z[i] > 2:
                    record(i, "WE Rule 4",
                           f"Two of three consecutive points beyond 2-sigma {label}")
                    break

        if not found:
            return pd.DataFrame(columns=["Point_Index", "Order_Month", "Rule",
                                         "Description", "OTIF_Pct"])
        return (pd.DataFrame(found)
                .drop_duplicates(subset=["Point_Index", "Rule"])
                .sort_values("Point_Index").reset_index(drop=True))

    # ------------------------------------------------------------------ #
    # Root cause
    # ------------------------------------------------------------------ #
    def analyse_root_causes(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Pareto analysis of service failures, ranked by business impact.

        Failures are ranked by *value at risk*, not by count. The most frequent
        failure mode is rarely the most expensive one, and a corrective action
        plan built on frequency alone tends to optimise the cheap tail while
        the costly failures continue.
        """
        failures = orders[~orders["OTIF"]].copy()
        if failures.empty:
            return pd.DataFrame(columns=["Root_Cause", "Failed_Orders", "Share_Pct"])

        agg = (failures.groupby("Root_Cause", as_index=False)
               .agg(Failed_Orders=(Keys.ORDER, "count"),
                    Avg_Delay_Days=("Delay_Days", "mean"),
                    Shortfall_Units=("Shortfall_Units", "sum"),
                    Value_At_Risk=("Order_Value", "sum"),
                    Suppliers_Involved=(Keys.SUPPLIER, "nunique"),
                    Categories_Involved=(Keys.CATEGORY, "nunique")))

        total_failures = agg["Failed_Orders"].sum()
        agg["Share_Pct"] = (agg["Failed_Orders"] / total_failures * 100).round(2)
        agg["Avg_Delay_Days"] = agg["Avg_Delay_Days"].round(2)

        agg = agg.sort_values("Value_At_Risk", ascending=False).reset_index(drop=True)
        agg["Cumulative_Value_Pct"] = (
            agg["Value_At_Risk"].cumsum() / agg["Value_At_Risk"].sum() * 100).round(2)
        # The "vital few": causes making up the first 80% of value at risk.
        agg["Vital_Few"] = agg["Cumulative_Value_Pct"] <= 80.0
        agg.loc[0, "Vital_Few"] = True     # the top cause is always vital

        agg["Owner"] = agg["Root_Cause"].map({
            "Supplier Delay": "Procurement / Supplier Development",
            "Warehouse Delay": "Warehouse Operations",
            "Transportation Delay": "Logistics",
            "Inventory Shortage": "Supply Planning",
            "Forecast Error": "Demand Planning",
            "Quality Issue": "Quality Assurance",
            "Documentation Issue": "Trade Compliance",
        }).fillna("Supply Chain")

        agg["Corrective_Action"] = agg["Root_Cause"].map({
            "Supplier Delay": "Supplier capacity review and confirmed delivery windows",
            "Warehouse Delay": "Inbound dock scheduling and receiving capacity review",
            "Transportation Delay": "Carrier performance review and lane re-routing",
            "Inventory Shortage": "Raise safety stock on affected SKUs; review ROP",
            "Forecast Error": "Bias correction in demand review; tighten override governance",
            "Quality Issue": "Supplier quality audit and incoming inspection tightening",
            "Documentation Issue": "Pre-clearance documentation checklist for import lanes",
        }).fillna("Investigate in S&OP supply review")
        return agg

    # ------------------------------------------------------------------ #
    # Supplier scorecard
    # ------------------------------------------------------------------ #
    def build_supplier_scorecard(self, orders: pd.DataFrame,
                                 supplier_master: pd.DataFrame) -> pd.DataFrame:
        """
        Supplier league table with a trend, not just a level.

        The trend column compares the most recent quarter against the prior
        three. A supplier at 91% and improving is a different management
        conversation from a supplier at 91% and deteriorating, and the level
        alone cannot distinguish them.
        """
        base = self._otif_by(orders, Keys.SUPPLIER).rename(
            columns={"OTIF_Pct": "OTIF_Pct_Overall"})

        months = sorted(orders["Order_Month"].unique())
        recent_months = set(months[-3:])
        prior_months = set(months[:-3])

        recent = orders[orders["Order_Month"].isin(recent_months)]
        prior = orders[orders["Order_Month"].isin(prior_months)]

        recent_otif = (recent.groupby(Keys.SUPPLIER)["OTIF"].mean() * 100).rename("OTIF_Recent_Q")
        prior_otif = (prior.groupby(Keys.SUPPLIER)["OTIF"].mean() * 100).rename("OTIF_Prior")

        card = (base.merge(recent_otif, on=Keys.SUPPLIER, how="left")
                    .merge(prior_otif, on=Keys.SUPPLIER, how="left"))
        card["OTIF_Delta_Pts"] = (card["OTIF_Recent_Q"] - card["OTIF_Prior"]).round(2)
        card["Trend"] = np.select(
            [card["OTIF_Delta_Pts"] > 2.0, card["OTIF_Delta_Pts"] < -2.0],
            ["Improving", "Deteriorating"], default="Stable")

        if not supplier_master.empty:
            meta = supplier_master.set_index(Keys.SUPPLIER)
            for col in ("Archetype", "Country", "Lead_Time_Weeks",
                        "Reliability_Score", "Is_Single_Source"):
                if col in meta.columns:
                    card[col] = card[Keys.SUPPLIER].map(meta[col])

        card["Risk_Flag"] = np.select(
            [(card["Benchmark"] == "At Risk") & card.get("Is_Single_Source", False).fillna(False),
             card["Benchmark"] == "At Risk",
             card["Trend"] == "Deteriorating"],
            ["CRITICAL - single source below threshold",
             "At risk - qualify alternate supply",
             "Watch - performance deteriorating"],
            default="Acceptable")
        card["OTIF_Recent_Q"] = card["OTIF_Recent_Q"].round(2)
        card["OTIF_Prior"] = card["OTIF_Prior"].round(2)
        return card.sort_values("OTIF_Pct_Overall").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, orders: pd.DataFrame,
            supplier_master: Optional[pd.DataFrame] = None) -> KPIResult:
        supplier_master = (supplier_master if supplier_master is not None
                           else pd.DataFrame(columns=[Keys.SUPPLIER]))
        detail = self.prepare_orders(orders)

        by_supplier = self._otif_by(detail, Keys.SUPPLIER)
        by_category = self._otif_by(detail, Keys.CATEGORY)
        by_channel = self._otif_by(detail, Keys.CHANNEL)
        by_month = self._otif_by(detail, "Order_Month").sort_values("Order_Month")
        by_sku = self._otif_by(detail, Keys.SKU)

        chart, violations = self.build_control_chart(by_month)
        root_causes = self.analyse_root_causes(detail)
        scorecard = self.build_supplier_scorecard(detail, supplier_master)

        overall = otif_rate(detail["On_Time"], detail["In_Full"])
        worst_month = chart.loc[chart["OTIF_Pct"].idxmin()] if not chart.empty else None

        diagnostics = {
            "total_orders": int(len(detail)),
            "overall_otif_pct": round(float(overall), 2),
            "on_time_pct": round(float(detail["On_Time"].mean() * 100), 2),
            "in_full_pct": round(float(detail["In_Full"].mean() * 100), 2),
            "volume_fill_rate_pct": round(float(
                detail["Delivered_Qty"].sum() / max(detail["Ordered_Qty"].sum(), 1)
                * 100), 2),
            "benchmark": ("World Class" if overall >= self.cfg.otif_world_class
                          else "Acceptable" if overall >= self.cfg.otif_acceptable
                          else "At Risk"),
            "months_analysed": int(len(chart)),
            "center_line": float(chart["Center_Line"].iloc[0]) if not chart.empty else None,
            "sigma": float(chart["Sigma"].iloc[0]) if not chart.empty else None,
            "ucl": float(chart["UCL"].iloc[0]) if not chart.empty else None,
            "lcl": float(chart["LCL"].iloc[0]) if not chart.empty else None,
            "special_cause_months": int(chart["Special_Cause"].sum()) if not chart.empty else 0,
            "below_2sigma_months": int(chart["Below_2Sigma_Alert"].sum())
            if not chart.empty else 0,
            "we_rule_violations": int(len(violations)),
            "worst_month": str(worst_month["Order_Month"]) if worst_month is not None else None,
            "worst_month_otif": float(worst_month["OTIF_Pct"]) if worst_month is not None else None,
            "top_root_cause": (root_causes.iloc[0]["Root_Cause"]
                               if not root_causes.empty else None),
            "top_root_cause_share": (float(root_causes.iloc[0]["Share_Pct"])
                                     if not root_causes.empty else None),
            "suppliers_at_risk": int((by_supplier["Benchmark"] == "At Risk").sum()),
            "failure_mode_mix": detail["Failure_Mode"].value_counts().to_dict(),
        }

        logger.info("OTIF: %.2f%% (%s) | on-time %.2f%% | in-full %.2f%% | "
                    "control limits [%.2f, %.2f] | %d special-cause month(s) | top cause: %s",
                    diagnostics["overall_otif_pct"], diagnostics["benchmark"],
                    diagnostics["on_time_pct"], diagnostics["in_full_pct"],
                    diagnostics["lcl"] or 0, diagnostics["ucl"] or 0,
                    diagnostics["special_cause_months"], diagnostics["top_root_cause"])

        return KPIResult(
            order_detail=detail,
            otif_by_supplier=by_supplier,
            otif_by_category=by_category,
            otif_by_channel=by_channel,
            otif_by_month=by_month,
            otif_by_sku=by_sku,
            spc_chart=chart,
            spc_violations=violations,
            root_cause_analysis=root_causes,
            supplier_scorecard=scorecard,
            diagnostics=diagnostics,
        )


def run_otif_analysis(config: AppConfig, orders: pd.DataFrame,
                      supplier_master: Optional[pd.DataFrame] = None) -> KPIResult:
    """Functional entry point used by the pipeline."""
    return OTIFTracker(config).run(orders, supplier_master)
