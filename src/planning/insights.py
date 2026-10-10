"""
Automated Executive Insight Engine.

Converts calculated results into the sentences an S&OP chair would actually say
in the monthly review.

Every insight in this module is *derived*. There is no template bank of
plausible-sounding statements waiting to be filled in with whatever number is
handy: each generator inspects a real result frame, decides whether the
condition it tests is genuinely present, and returns nothing when it is not.
If the supply plan is clean, the supply section is empty -- which is itself the
finding.

Insights carry three pieces of metadata that make them usable rather than
decorative:

* **Severity** -- Critical / High / Medium / Info, driving sort order.
* **Impact value** -- the rupee number at stake, so a reader can triage.
* **Owner** -- the function that must act. An insight with no owner is a fact,
  not an insight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, Keys
from src.shared.utils import get_logger, human_number, safe_div

logger = get_logger("insights")

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3}


@dataclass
class Insight:
    """A single management-ready finding."""

    domain: str
    severity: str
    headline: str
    detail: str
    metric: str = ""
    value: float = 0.0
    impact_value: float = 0.0
    owner: str = "Supply Chain"
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class InsightEngine:
    """Generates the executive insight set from pipeline results."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.currency = config.report.currency_symbol

    # ------------------------------------------------------------------ #
    # Demand
    # ------------------------------------------------------------------ #
    def demand_insights(self, forecast_result) -> List[Insight]:
        out: List[Insight] = []
        diag = forecast_result.diagnostics
        hierarchy = forecast_result.hierarchy_accuracy

        # -- accuracy against the benchmark, at the level that matters ---- #
        if not hierarchy.empty:
            review_level = hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
            if not review_level.empty:
                row = review_level.iloc[0]
                wape = float(row["WAPE"])
                target = self.config.forecast.portfolio_mape_target
                meets = wape < target
                out.append(Insight(
                    domain="Demand",
                    severity="Info" if meets else "Medium",
                    headline=(f"Forecast accuracy at the S&OP review level is "
                              f"{wape:.1f}% WAPE, {'inside' if meets else 'outside'} "
                              f"the {target:.0f}% planning benchmark."),
                    detail=(f"Measured on an untouched {self.config.forecast.horizon_weeks}-week "
                            f"holdout at Category/Month, the level at which demand is "
                            f"reviewed and signed off. SKU-week accuracy is materially "
                            f"weaker, as expected -- error cancels on aggregation."),
                    metric="WAPE (Category/Month)", value=round(wape, 2),
                    owner="Demand Planning",
                    recommended_action=("Maintain current model governance."
                                        if meets else
                                        "Review high-error categories in the demand review.")))

            fva_row = hierarchy[hierarchy["Aggregation_Level"] == "SKU / Week"]
            if not fva_row.empty and np.isfinite(fva_row.iloc[0].get("FVA_vs_Naive", np.nan)):
                fva = float(fva_row.iloc[0]["FVA_vs_Naive"])
                positive = fva > 0
                out.append(Insight(
                    domain="Demand",
                    severity="Info" if positive else "High",
                    headline=(f"The modelling pipeline adds {fva:+.1f} percentage points of "
                              f"Forecast Value Added over a seasonal-naive benchmark."),
                    detail=("Forecast Value Added compares the deployed models against "
                            "repeating the same week last year. A negative value would mean "
                            "the analytics are destroying value and should be retired."),
                    metric="FVA vs Seasonal Naive", value=round(fva, 2),
                    owner="Demand Planning",
                    recommended_action=("Retain the model portfolio." if positive else
                                        "Revert to the naive benchmark and rebuild features.")))

        # -- systematic bias by category ---------------------------------- #
        category = forecast_result.category_accuracy
        if not category.empty and "Weighted_Bias_Pct" in category.columns:
            for _, row in category.iterrows():
                bias = float(row["Weighted_Bias_Pct"]) if np.isfinite(row["Weighted_Bias_Pct"]) else 0.0
                if abs(bias) < 8.0:
                    continue
                direction = "over-forecasting" if bias > 0 else "under-forecasting"
                consequence = ("excess stock and write-off risk" if bias > 0
                               else "stockout and lost-sales risk")
                out.append(Insight(
                    domain="Demand",
                    severity="High" if abs(bias) > 20 else "Medium",
                    headline=(f"Category {row[Keys.CATEGORY]} shows a forecast bias of "
                              f"{bias:+.1f}%, indicating systematic {direction}."),
                    detail=(f"Volume-weighted signed error across the holdout. A persistent "
                            f"one-sided error is a process problem, not a model problem, and "
                            f"drives {consequence}. Median MAPE in this category is "
                            f"{row.get('Median_MAPE', float('nan')):.1f}%."),
                    metric="Forecast Bias %", value=round(bias, 2),
                    owner="Demand Planning",
                    recommended_action=("Apply a bias correction factor and challenge "
                                        "commercial overrides in this category.")))

        # -- interval calibration ----------------------------------------- #
        coverage = diag.get("interval_coverage", {})
        cov80 = coverage.get("coverage_80")
        if cov80 is not None and np.isfinite(cov80):
            gap = abs(cov80 - 80.0)
            if gap > 7.0:
                direction = "wider" if cov80 > 80 else "narrower"
                out.append(Insight(
                    domain="Demand",
                    severity="Medium",
                    headline=(f"The 80% prediction interval achieved {cov80:.1f}% empirical "
                              f"coverage -- the bands are {direction} than nominal."),
                    detail=("Interval width feeds safety-stock sizing downstream. "
                            "Over-wide intervals inflate buffer stock; over-narrow intervals "
                            "under-protect service."),
                    metric="PI80 Coverage %", value=round(cov80, 2),
                    owner="Demand Planning",
                    recommended_action="Recalibrate the horizon variance growth parameter."))

        # -- model mix ----------------------------------------------------- #
        accuracy = forecast_result.sku_accuracy
        if not accuracy.empty and "Selected_Model" in accuracy.columns:
            mix = accuracy["Selected_Model"].value_counts()
            if len(mix):
                top_model = mix.index[0]
                share = mix.iloc[0] / mix.sum() * 100
                out.append(Insight(
                    domain="Demand",
                    severity="Info",
                    headline=(f"{top_model} was selected as champion model for "
                              f"{share:.0f}% of the portfolio."),
                    detail=("Champion selection is made per SKU on cross-validation "
                            "performance only, before the holdout is opened. Model mix: "
                            + ", ".join(f"{m} {c}" for m, c in mix.items()) + "."),
                    metric="Champion model share %", value=round(share, 1),
                    owner="Demand Planning"))

            poor = accuracy[accuracy["MAPE"] > 50] if "MAPE" in accuracy.columns else pd.DataFrame()
            if len(poor) > 0:
                share = len(poor) / len(accuracy) * 100
                out.append(Insight(
                    domain="Demand",
                    severity="Medium" if share < 25 else "High",
                    headline=(f"{len(poor)} SKUs ({share:.0f}% of the portfolio) forecast "
                              f"above 50% MAPE and are not safely plannable statistically."),
                    detail=("These are typically new, intermittent or heavily promoted items. "
                            "They should be planned on commercial input with wider buffers "
                            "rather than trusted to the statistical baseline."),
                    metric="SKUs above 50% MAPE", value=float(len(poor)),
                    owner="Demand Planning",
                    recommended_action="Move to attribute-based or judgemental planning."))
        return out

    # ------------------------------------------------------------------ #
    # Consensus
    # ------------------------------------------------------------------ #
    def consensus_insights(self, consensus_result) -> List[Insight]:
        out: List[Insight] = []
        diag = consensus_result.diagnostics
        plan = consensus_result.consensus_plan
        scenario = diag["scenario"]
        primary = plan[plan["Scenario"] == scenario]

        statistical = primary["Forecast_Units"].sum()
        consensus = primary["Consensus_Units"].sum()
        uplift = safe_div(consensus - statistical, statistical) * 100

        if abs(uplift) > 2:
            out.append(Insight(
                domain="Consensus",
                severity="Medium" if abs(uplift) > 12 else "Info",
                headline=(f"The {scenario} consensus plan sits {uplift:+.1f}% against the "
                          f"statistical baseline."),
                detail=(f"Statistical baseline {statistical:,.0f} units versus consensus "
                        f"{consensus:,.0f} units. A persistent positive gap that does not "
                        f"materialise in actuals is the classic S&OP failure mode and shows "
                        f"up later as excess stock."),
                metric="Consensus uplift %", value=round(uplift, 2),
                owner="S&OP / Demand Planning",
                recommended_action=("Track override realisation rate in next month's review.")))

        alerts = diag["override_alerts"]
        if alerts:
            threshold = self.config.consensus.override_alert_threshold_pct
            escalations = diag["board_escalations"]
            out.append(Insight(
                domain="Consensus",
                severity="High" if escalations > 0 else "Medium",
                headline=(f"{alerts:,} commercial overrides exceed the {threshold:.0f}% "
                          f"governance threshold, of which {escalations:,} require S&OP "
                          f"board sign-off."),
                detail=("Every flagged override carries a mandatory justification and is "
                        "written to the audit log with a timestamp. The statistical baseline "
                        "is preserved alongside so post-mortem attribution is possible."),
                metric="Override alerts", value=float(alerts),
                owner="S&OP Governance",
                recommended_action="Review flagged overrides in the demand review."))

        blocked = diag.get("blocked_overrides", 0)
        if blocked:
            out.append(Insight(
                domain="Consensus", severity="Critical",
                headline=(f"{blocked:,} flagged overrides have no valid justification and are "
                          f"BLOCKED from entering the consensus plan."),
                detail=("Governance rule: any deviation beyond the alert threshold requires a "
                        "documented business rationale before it can be accepted."),
                metric="Blocked overrides", value=float(blocked),
                owner="S&OP Governance",
                recommended_action="Obtain justification or revert to statistical baseline."))

        constrained_value = diag["constrained_out_value"]
        if constrained_value > 0:
            constrained_skus = int(primary.loc[primary["Is_Constrained"], Keys.SKU].nunique())
            out.append(Insight(
                domain="Consensus",
                severity="High",
                headline=(f"Operational constraints remove "
                          f"{human_number(constrained_value, self.currency)} of demand across "
                          f"{constrained_skus} SKUs from the achievable plan."),
                detail=(f"{diag['constrained_out_units']:,.0f} units of commercially supported "
                        f"demand cannot be supplied within supplier and warehouse capacity in "
                        f"the horizon. This is revenue the business is choosing not to chase "
                        f"and should be an explicit board decision, not a planning by-product."),
                metric="Constrained-out value", value=round(constrained_value, 0),
                impact_value=round(constrained_value, 0),
                owner="Supply Planning / Procurement",
                recommended_action=("Secure incremental capacity, re-phase demand, or "
                                    "formally accept the revenue reduction.")))

        summary = consensus_result.scenario_summary
        if not summary.empty and len(summary) > 1:
            spread = summary["Consensus_Value"].max() - summary["Consensus_Value"].min()
            out.append(Insight(
                domain="Consensus", severity="Info",
                headline=(f"Scenario range spans {human_number(spread, self.currency)} of "
                          f"planned demand value between Conservative and Aggressive."),
                detail=("The spread quantifies the commercial risk appetite decision. "
                        "Choosing a scenario is choosing a working-capital and service "
                        "position, and the range makes that trade-off explicit."),
                metric="Scenario value spread", value=round(float(spread), 0),
                impact_value=round(float(spread), 0),
                owner="S&OP Board"))
        return out

    # ------------------------------------------------------------------ #
    # Supply
    # ------------------------------------------------------------------ #
    def supply_insights(self, supply_result) -> List[Insight]:
        out: List[Insight] = []
        diag = supply_result.diagnostics
        summary = supply_result.sku_summary
        capacity = supply_result.capacity_alerts
        stockouts = supply_result.stockout_risks

        # -- capacity breaches, worst supplier first ---------------------- #
        breaches = capacity[(capacity["Severity"] == "BREACH")
                            & (capacity["Resource_Type"] == "Supplier")]
        if not breaches.empty:
            worst = breaches.sort_values("Utilisation_Pct", ascending=False).iloc[0]
            out.append(Insight(
                domain="Supply", severity="Critical",
                headline=(f"Supplier {worst['Resource_ID']} is loaded to "
                          f"{worst['Utilisation_Pct']:.0f}% of weekly capacity in week "
                          f"{int(worst['horizon'])}, affecting "
                          f"{int(worst['SKUs_Affected'])} SKUs."),
                detail=(f"Planned orders require {worst['Required_Units']:,.0f} units against "
                        f"available capacity of {worst['Available_Capacity']:,.0f} -- an "
                        f"overload of {worst['Overload_Units']:,.0f} units. "
                        f"{breaches['Resource_ID'].nunique()} supplier(s) breach capacity "
                        f"across {len(breaches)} supplier-weeks in total."),
                metric="Peak supplier utilisation %",
                value=round(float(worst["Utilisation_Pct"]), 1),
                owner="Procurement",
                recommended_action=("Re-phase orders, activate alternate supply, or "
                                    "negotiate incremental capacity.")))

        warnings = capacity[(capacity["Severity"] == "WARNING")
                            & (capacity["Resource_Type"] == "Supplier")]
        if not warnings.empty and breaches.empty:
            out.append(Insight(
                domain="Supply", severity="High",
                headline=(f"{warnings['Resource_ID'].nunique()} suppliers exceed "
                          f"{self.config.supply.capacity_alert_threshold * 100:.0f}% "
                          f"capacity utilisation without breaching."),
                detail="Utilisation at this level leaves no headroom for demand upside.",
                metric="Suppliers near capacity",
                value=float(warnings["Resource_ID"].nunique()),
                owner="Procurement",
                recommended_action="Pre-book capacity ahead of the peak."))

        warehouse_breaches = capacity[(capacity["Severity"] == "BREACH")
                                      & (capacity["Resource_Type"] == "Warehouse")]
        if not warehouse_breaches.empty:
            worst_wh = warehouse_breaches.sort_values(
                "Utilisation_Pct", ascending=False).iloc[0]
            out.append(Insight(
                domain="Supply", severity="High",
                headline=(f"Warehouse {worst_wh['Resource_ID']} inbound capacity is exceeded "
                          f"in week {int(worst_wh['horizon'])} at "
                          f"{worst_wh['Utilisation_Pct']:.0f}% utilisation."),
                detail=("Inbound receiving is a hard physical constraint. Orders arriving "
                        "beyond dock capacity queue and become late receipts regardless of "
                        "supplier performance."),
                metric="Warehouse utilisation %",
                value=round(float(worst_wh["Utilisation_Pct"]), 1),
                owner="Warehouse Operations",
                recommended_action="Re-phase inbound scheduling or add receiving shifts."))

        # -- stockouts inside the lead time: the expedite list ------------ #
        expedite = summary[summary["First_Stockout_Week"].notna()
                           & ~summary["Can_Recover_In_Lead_Time"]]
        if not expedite.empty:
            worst = expedite.sort_values("Supply_Risk_Index", ascending=False).iloc[0]
            out.append(Insight(
                domain="Supply", severity="Critical",
                headline=(f"{len(expedite)} SKUs will stock out inside their supplier lead "
                          f"time and cannot be recovered by a normal replenishment order."),
                detail=(f"Worst case: {worst[Keys.SKU]} is projected to stock out in week "
                        f"{int(worst['First_Stockout_Week'])} against a lead time of "
                        f"{worst['Lead_Time_Weeks']:.1f} weeks "
                        f"(Supply Risk Index {worst['Supply_Risk_Index']:.0f}, "
                        f"{worst['Risk_Band']}). A standard order placed today arrives "
                        f"after the shortage begins."),
                metric="SKUs requiring expedite", value=float(len(expedite)),
                owner="Supply Planning",
                recommended_action=("Air-freight, partial-ship from an alternate supplier, "
                                    "or reallocate stock between warehouses.")))

        recoverable = summary[summary["First_Stockout_Week"].notna()
                              & summary["Can_Recover_In_Lead_Time"]]
        if not recoverable.empty:
            earliest = int(recoverable["First_Stockout_Week"].min())
            out.append(Insight(
                domain="Supply", severity="High",
                headline=(f"{len(recoverable)} SKUs show projected stockouts that remain "
                          f"recoverable, the earliest in week {earliest}."),
                detail=(f"These were identified {earliest} week(s) ahead of the shortage and "
                        f"inside the supplier lead time window, which is the entire point of "
                        f"running a forward supply plan rather than reacting to an empty bin."),
                metric="Recoverable stockout SKUs", value=float(len(recoverable)),
                owner="Supply Planning",
                recommended_action="Release planned orders now."))

        # -- individual high-value stockout ------------------------------- #
        if not stockouts.empty and "Revenue_At_Risk" in stockouts.columns:
            projected = stockouts[stockouts["Event_Type"] == "Projected Stockout"]
            if not projected.empty:
                top = projected.sort_values("Revenue_At_Risk", ascending=False).iloc[0]
                if top["Revenue_At_Risk"] > 0:
                    out.append(Insight(
                        domain="Supply", severity="High",
                        headline=(f"{top[Keys.SKU]} is projected to stock out in week "
                                  f"{int(top['horizon'])}, putting "
                                  f"{human_number(top['Revenue_At_Risk'], self.currency)} of "
                                  f"revenue at risk."),
                        detail=(f"Shortage of {top['Shortage_Units']:,.0f} units against "
                                f"demand of {top['Demand_Units']:,.0f}. "
                                f"Category {top.get(Keys.CATEGORY, 'n/a')}, "
                                f"ABC class {top.get('ABC_Class', 'n/a')}, "
                                f"supplier {top.get(Keys.SUPPLIER, 'n/a')}."),
                        metric="Revenue at risk",
                        value=round(float(top["Revenue_At_Risk"]), 0),
                        impact_value=round(float(top["Revenue_At_Risk"]), 0),
                        owner="Supply Planning",
                        recommended_action=str(top.get("Recommended_Action", ""))))

                total_at_risk = float(projected["Revenue_At_Risk"].sum())
                if total_at_risk > 0:
                    out.append(Insight(
                        domain="Supply", severity="High",
                        headline=(f"Total revenue at risk from projected stockouts is "
                                  f"{human_number(total_at_risk, self.currency)} across "
                                  f"{projected[Keys.SKU].nunique()} SKUs."),
                        detail=("Computed as shortage units multiplied by unit selling price, "
                                "summed across the horizon. This assumes lost sales rather "
                                "than deferred demand, so it is an upper bound."),
                        metric="Total revenue at risk", value=round(total_at_risk, 0),
                        impact_value=round(total_at_risk, 0),
                        owner="S&OP Board"))

        # -- risk distribution -------------------------------------------- #
        red = int((summary["Risk_Band"] == "Red").sum())
        if red:
            out.append(Insight(
                domain="Supply", severity="High" if red > 20 else "Medium",
                headline=(f"{red} SKUs carry a Red Supply Risk Index "
                          f"(>= {self.config.supply.risk_red_threshold:.0f}/100)."),
                detail=("The index blends stockout proximity, capacity utilisation, cover "
                        "shortfall, lead-time exposure and forecast uncertainty using "
                        "documented weights. Every component is inspectable per SKU."),
                metric="Red-risk SKUs", value=float(red),
                owner="Supply Planning",
                recommended_action="Work the Red list in the supply review."))

        # -- excess stock -------------------------------------------------- #
        excess = summary[summary["Min_Cover"] > self.config.supply.excess_cover_weeks]
        if not excess.empty:
            capital = float(excess["Avg_Working_Capital"].sum())
            out.append(Insight(
                domain="Supply", severity="Medium",
                headline=(f"{len(excess)} SKUs hold more than "
                          f"{self.config.supply.excess_cover_weeks:.0f} weeks of cover, "
                          f"tying up {human_number(capital, self.currency)} of working "
                          f"capital."),
                detail=("Excess cover is the quiet cost of a service-first planning culture. "
                        "Releasing it is usually the fastest cash lever available to a "
                        "supply chain function."),
                metric="Excess-cover working capital", value=round(capital, 0),
                impact_value=round(capital, 0),
                owner="Supply Planning / Finance",
                recommended_action="Reduce order quantities and re-phase open POs."))
        return out

    # ------------------------------------------------------------------ #
    # Service
    # ------------------------------------------------------------------ #
    def service_insights(self, kpi_result) -> List[Insight]:
        out: List[Insight] = []
        diag = kpi_result.diagnostics
        otif = diag["overall_otif_pct"]
        benchmark = diag["benchmark"]
        top_cause = diag.get("top_root_cause")
        share = diag.get("top_root_cause_share")

        severity = ("Critical" if benchmark == "At Risk"
                    else "Medium" if benchmark == "Acceptable" else "Info")
        cause_clause = (f", primarily driven by {top_cause.lower()} ({share:.0f}% of failures)"
                        if top_cause and share else "")
        out.append(Insight(
            domain="Service", severity=severity,
            headline=(f"OTIF is {otif:.1f}%, classified {benchmark} against the "
                      f"{self.config.kpi.otif_acceptable:.0f}% acceptable threshold"
                      f"{cause_clause}."),
            detail=(f"On-time {diag['on_time_pct']:.1f}%, in-full {diag['in_full_pct']:.1f}%, "
                    f"volume fill rate {diag['volume_fill_rate_pct']:.1f}% across "
                    f"{diag['total_orders']:,} orders over {diag['months_analysed']} months. "
                    f"The gap between order-level OTIF and volume fill rate shows failures "
                    f"are concentrated in a minority of lines rather than spread thinly."),
            metric="OTIF %", value=otif, owner="Supply Chain Operations",
            recommended_action=("Drive the vital-few root causes through supplier development."
                                if benchmark != "World Class" else "Sustain current controls."))
        )

        # -- SPC: the finding that a threshold chart cannot produce -------- #
        special = diag.get("special_cause_months", 0)
        chart = kpi_result.spc_chart
        if special and not chart.empty:
            flagged = chart[chart["Special_Cause"] | chart["Below_2Sigma_Alert"]]
            low = flagged[flagged["Sigma_Deviation"] < 0]
            if not low.empty:
                months = ", ".join(low["Order_Month"].astype(str).tolist())
                worst = low.loc[low["OTIF_Pct"].idxmin()]
                out.append(Insight(
                    domain="Service", severity="Critical",
                    headline=(f"Statistical process control identifies {len(low)} month(s) of "
                              f"special-cause variation below the control limit: {months}."),
                    detail=(f"Control limits are {chart['LCL'].iloc[0]:.1f}% to "
                            f"{chart['UCL'].iloc[0]:.1f}% around a centre line of "
                            f"{chart['Center_Line'].iloc[0]:.1f}%, with sigma estimated from "
                            f"the average moving range. {worst['Order_Month']} at "
                            f"{worst['OTIF_Pct']:.1f}% sits "
                            f"{abs(worst['Sigma_Deviation']):.1f} sigma below centre -- this "
                            f"is a genuine process shift, not routine variation, and warrants "
                            f"a corrective action."),
                    metric="Special-cause months", value=float(len(low)),
                    owner="Supply Chain Operations",
                    recommended_action="Run a structured root-cause investigation."))

            high = flagged[flagged["Sigma_Deviation"] > 0]
            if not high.empty:
                out.append(Insight(
                    domain="Service", severity="Info",
                    headline=(f"Control chart also flags {len(high)} month(s) of favourable "
                              f"special-cause variation -- the process has shifted upward."),
                    detail=("A sustained run above the centre line means the current control "
                            "limits are stale and understate present capability. Recalculate "
                            "limits on the post-shift period and lock in whatever changed."),
                    metric="Favourable special-cause months", value=float(len(high)),
                    owner="Supply Chain Operations",
                    recommended_action="Recompute control limits and standardise the improvement."))

        # -- Pareto vital few ---------------------------------------------- #
        causes = kpi_result.root_cause_analysis
        if not causes.empty and "Vital_Few" in causes.columns:
            vital = causes[causes["Vital_Few"]]
            if not vital.empty:
                value = float(vital["Value_At_Risk"].sum())
                total = float(causes["Value_At_Risk"].sum())
                names = ", ".join(vital["Root_Cause"].tolist())
                out.append(Insight(
                    domain="Service", severity="High",
                    headline=(f"{len(vital)} of {len(causes)} root causes account for "
                              f"{safe_div(value, total) * 100:.0f}% of service-failure value: "
                              f"{names}."),
                    detail=(f"Ranked by value at risk rather than frequency, because the most "
                            f"common failure is rarely the most expensive. Total exposure "
                            f"{human_number(total, self.currency)}. Owners: "
                            + "; ".join(f"{r.Root_Cause} -> {r.Owner}"
                                        for r in vital.itertuples()) + "."),
                    metric="Vital-few value at risk", value=round(value, 0),
                    impact_value=round(value, 0),
                    owner="Supply Chain Operations",
                    recommended_action=str(vital.iloc[0]["Corrective_Action"])))

        # -- worst suppliers ------------------------------------------------ #
        card = kpi_result.supplier_scorecard
        if not card.empty:
            reliable = card[card.get("Statistically_Reliable", True)]
            at_risk = reliable[reliable["Benchmark"] == "At Risk"]
            if not at_risk.empty:
                worst = at_risk.iloc[0]
                out.append(Insight(
                    domain="Service", severity="High",
                    headline=(f"{len(at_risk)} suppliers with a statistically meaningful order "
                              f"count sit below the 90% OTIF threshold; worst is "
                              f"{worst[Keys.SUPPLIER]} at {worst['OTIF_Pct_Overall']:.1f}%."),
                    detail=(f"{worst[Keys.SUPPLIER]}: {int(worst['Orders'])} orders, average "
                            f"delay {worst['Avg_Delay_Days']:.1f} days, trend "
                            f"{worst.get('Trend', 'n/a')}. Flag: "
                            f"{worst.get('Risk_Flag', 'n/a')}."),
                    metric="Suppliers below threshold", value=float(len(at_risk)),
                    owner="Procurement",
                    recommended_action="Escalate through supplier performance reviews."))

            deteriorating = card[card.get("Trend", "") == "Deteriorating"]
            if not deteriorating.empty:
                out.append(Insight(
                    domain="Service", severity="Medium",
                    headline=(f"{len(deteriorating)} suppliers show a deteriorating OTIF trend "
                              f"in the most recent quarter."),
                    detail=("Trend is measured as the most recent three months against the "
                            "prior period. A deteriorating supplier at an acceptable level is "
                            "a leading indicator; the level alone is a lagging one."),
                    metric="Deteriorating suppliers", value=float(len(deteriorating)),
                    owner="Procurement",
                    recommended_action="Open early dialogue before performance breaches target."))
        return out

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #
    def generate(self, forecast_result=None, consensus_result=None,
                 supply_result=None, kpi_result=None) -> pd.DataFrame:
        """Build the full, severity-ranked insight register."""
        insights: List[Insight] = []
        if forecast_result is not None:
            insights += self.demand_insights(forecast_result)
        if consensus_result is not None:
            insights += self.consensus_insights(consensus_result)
        if supply_result is not None:
            insights += self.supply_insights(supply_result)
        if kpi_result is not None:
            insights += self.service_insights(kpi_result)

        if not insights:
            logger.warning("No insights generated -- check that pipeline results were passed.")
            return pd.DataFrame(columns=["Domain", "Severity", "Headline", "Detail"])

        frame = pd.DataFrame([i.to_dict() for i in insights])
        frame["severity_rank"] = frame["severity"].map(SEVERITY_ORDER).fillna(9)
        frame = (frame.sort_values(["severity_rank", "impact_value"],
                                   ascending=[True, False])
                 .drop(columns=["severity_rank"]).reset_index(drop=True))
        frame.insert(0, "Rank", range(1, len(frame) + 1))
        frame.columns = [c if c == "Rank" else c.replace("_", " ").title().replace(" ", "_")
                         for c in frame.columns]

        counts = frame["Severity"].value_counts().to_dict()
        logger.info("Generated %d executive insights (%s).", len(frame), counts)
        return frame


def generate_insights(config: AppConfig, **results) -> pd.DataFrame:
    """Functional entry point used by the pipeline."""
    return InsightEngine(config).generate(**results)
