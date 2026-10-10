"""
Module 2 -- Consensus S&OP Planning.

Reconciles three inputs that, in every real S&OP process, disagree:

* **Statistical baseline** -- what the models say demand will be.
* **Commercial intelligence** -- what Sales and Marketing know that the history
  cannot: a listing win, an NPD launch, a competitor exit, a customer
  commitment, cannibalisation from a sister SKU.
* **Operational constraints** -- what Supply can actually deliver: supplier
  weekly capacity, warehouse inbound capacity, MOQ, lead time.

The governance rules encoded here are the difference between a consensus
process and an argument:

1. **The statistical baseline is never overwritten.** It is carried through
   every stage as its own column. When the plan misses, you can tell whether
   the model was wrong or the override was.
2. **Large overrides must be justified.** Any deviation beyond the configured
   threshold (default 20%) is flagged, requires a justification string, and is
   written to an immutable audit log with a timestamp.
3. **Constraints bind last.** Commercial ambition is recorded in full, then the
   operational ceiling is applied on top. The gap between the two -- the
   *constrained-out* volume -- is itself a headline S&OP number, because it is
   the revenue the business is choosing not to chase.

The waterfall the dashboard renders is a direct read-out of this sequence::

    Statistical -> Commercial -> Promo/NPD -> Constraint -> Final Consensus
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, ConsensusConfig, Keys
from src.shared.utils import get_logger, safe_div

logger = get_logger("consensus")


@dataclass
class ConsensusResult:
    """Outputs of the consensus workflow."""

    consensus_plan: pd.DataFrame       # SKU x week plan, all stages retained
    override_log: pd.DataFrame         # auditable register of every override
    scenario_summary: pd.DataFrame     # portfolio totals per scenario
    waterfall: pd.DataFrame            # aggregated bridge for the dashboard
    diagnostics: Dict[str, object]


class ConsensusPlanner:
    """Builds the consensus demand plan under a configurable scenario."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: ConsensusConfig = config.consensus
        self.rng = np.random.default_rng(config.seed + 303)

    # ------------------------------------------------------------------ #
    # Commercial intelligence layer
    # ------------------------------------------------------------------ #
    def generate_commercial_inputs(self, forecast: pd.DataFrame,
                                   sku_master: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate the commercial overrides that Sales and Marketing submit.

        Overrides are not random noise on every SKU. They concentrate where
        commercial teams actually have information the model does not:

        * **New products** -- almost always overridden upward; the model has
          little or no history to work from and knows nothing about the
          listing plan.
        * **Promoted SKUs** -- trade marketing knows the display support and
          the retailer's forward-buy intent.
        * **Large SKUs** -- get management attention in proportion to revenue.

        Roughly half of overrides are small refinements; the tail is large and
        is exactly what the >20% governance rule exists to catch.
        """
        rng = self.rng
        base = forecast[[Keys.SKU, "horizon", Keys.WEEK, "Forecast_Units",
                         Keys.CATEGORY, Keys.CHANNEL, "Unit_Price"]].copy()

        lifecycle = sku_master.set_index(Keys.SKU)["Lifecycle_Status"]
        base["Lifecycle_Status"] = base[Keys.SKU].map(lifecycle).fillna("Mature")

        sku_volume = base.groupby(Keys.SKU)["Forecast_Units"].sum()
        volume_rank = sku_volume.rank(pct=True)

        # Probability a SKU attracts any commercial input at all.
        prob = (base[Keys.SKU].map(volume_rank).fillna(0.5) * 0.55
                + base["Lifecycle_Status"].eq("New Product") * 0.40
                + 0.12)
        prob = np.clip(prob * self.cfg.share_of_skus_with_commercial_input * 2.0, 0, 0.92)

        selected_skus = sku_volume.index[
            rng.random(len(sku_volume)) <
            np.clip(prob.groupby(base[Keys.SKU]).first().reindex(sku_volume.index).fillna(0.2), 0, 1)
        ]
        base["Has_Commercial_Input"] = base[Keys.SKU].isin(selected_skus)

        n = len(base)
        drivers = np.array(self.cfg.commercial_drivers)
        driver_probs = {
            "New Product": [0.10, 0.14, 0.52, 0.10, 0.12, 0.02],
            "Phase-Out":   [0.34, 0.06, 0.02, 0.10, 0.06, 0.42],
            "Mature":      [0.30, 0.24, 0.03, 0.18, 0.16, 0.09],
        }
        driver = np.empty(n, dtype=object)
        for status, probs in driver_probs.items():
            mask = (base["Lifecycle_Status"] == status).to_numpy()
            if mask.any():
                driver[mask] = rng.choice(drivers, int(mask.sum()), p=probs)
        base["Commercial_Driver"] = driver

        # Uplift magnitude by driver -- NPD skews high, cannibalisation negative.
        centre = {
            "Sales Override": 0.04, "Promotion Uplift": 0.13,
            "New Product Introduction": 0.34, "Customer Commitment": 0.11,
            "Market Opportunity": 0.16, "Cannibalisation Adjustment": -0.17,
        }
        spread = {
            "Sales Override": 0.16, "Promotion Uplift": 0.14,
            "New Product Introduction": 0.28, "Customer Commitment": 0.09,
            "Market Opportunity": 0.19, "Cannibalisation Adjustment": 0.12,
        }
        mu = base["Commercial_Driver"].map(centre).fillna(0.0).to_numpy()
        sd = base["Commercial_Driver"].map(spread).fillna(0.1).to_numpy()
        raw_delta = rng.normal(mu, sd)

        # Overrides are submitted at SKU level and applied uniformly across the
        # horizon, with a mild taper: commercial confidence decays with distance.
        sku_delta = pd.Series(raw_delta, index=base.index).groupby(base[Keys.SKU]).transform("mean")
        taper = 1.0 - 0.03 * (base["horizon"] - 1)
        delta = np.where(base["Has_Commercial_Input"], sku_delta * taper, 0.0)

        base["Commercial_Delta_Pct"] = np.round(delta * 100.0, 2)
        base["Commercial_Forecast"] = np.round(
            np.clip(base["Forecast_Units"] * (1.0 + delta), 0.0, None), 1)
        base.loc[~base["Has_Commercial_Input"], "Commercial_Driver"] = "No Override"
        base.loc[~base["Has_Commercial_Input"], "Commercial_Forecast"] = \
            base.loc[~base["Has_Commercial_Input"], "Forecast_Units"]

        base["Justification"] = self._justifications(base)
        base["Submitted_By"] = np.where(
            base["Has_Commercial_Input"],
            rng.choice(["Sales Director", "Category Manager", "Key Account Manager",
                        "Trade Marketing Lead", "Channel Head"], n),
            "-")
        return base

    @staticmethod
    def _justifications(frame: pd.DataFrame) -> np.ndarray:
        """Attach a driver-appropriate business justification to each override."""
        templates = {
            "Sales Override": "Field intelligence indicates demand shift not yet visible in history.",
            "Promotion Uplift": "Confirmed trade promotion with display support; retailer forward-buy expected.",
            "New Product Introduction": "NPD launch plan with confirmed listings; statistical history insufficient.",
            "Customer Commitment": "Signed customer volume commitment for the planning horizon.",
            "Market Opportunity": "Competitor supply gap creates addressable share opportunity.",
            "Cannibalisation Adjustment": "Volume transfer expected to replacement SKU in same sub-category.",
            "No Override": "-",
        }
        return frame["Commercial_Driver"].map(templates).fillna("-").to_numpy()

    # ------------------------------------------------------------------ #
    # Operational constraint layer
    # ------------------------------------------------------------------ #
    def build_operational_constraints(self, plan: pd.DataFrame,
                                      sku_master: pd.DataFrame,
                                      supplier_master: pd.DataFrame,
                                      warehouse_master: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the deliverable ceiling for every SKU-week.

        Supplier weekly capacity is a *shared* resource: it is allocated across
        that supplier's SKUs in proportion to each SKU's commercial demand
        (fair-share allocation). Applying the full supplier capacity to every
        SKU independently -- the common shortcut -- would silently overstate
        what the network can deliver by a factor equal to the number of SKUs
        per supplier.
        """
        work = plan.merge(
            sku_master[[Keys.SKU, Keys.SUPPLIER, Keys.WAREHOUSE, "Unit_Cost"]],
            on=Keys.SKU, how="left")

        supplier = supplier_master.set_index(Keys.SUPPLIER)
        work["Supplier_Weekly_Capacity"] = work[Keys.SUPPLIER].map(
            supplier["Weekly_Capacity_Units"]).fillna(1e9)
        work["Supplier_Lead_Time_Weeks"] = work[Keys.SUPPLIER].map(
            supplier["Lead_Time_Weeks"]).fillna(2.0)
        work["Supplier_MOQ"] = work[Keys.SUPPLIER].map(supplier["MOQ"]).fillna(0)

        warehouse = warehouse_master.set_index(Keys.WAREHOUSE)
        work["Warehouse_Weekly_Capacity"] = work[Keys.WAREHOUSE].map(
            warehouse["Weekly_Inbound_Capacity_Units"]).fillna(1e9)

        # --- fair-share allocation of the shared supplier capacity --------- #
        supplier_week_demand = (work.groupby([Keys.SUPPLIER, "horizon"])
                                ["Commercial_Forecast"].transform("sum"))
        work["Supplier_Demand_Share"] = np.where(
            supplier_week_demand > 0,
            work["Commercial_Forecast"] / supplier_week_demand, 0.0)
        work["Supplier_Allocated_Capacity"] = (
            work["Supplier_Weekly_Capacity"] * work["Supplier_Demand_Share"])

        warehouse_week_demand = (work.groupby([Keys.WAREHOUSE, "horizon"])
                                 ["Commercial_Forecast"].transform("sum"))
        work["Warehouse_Demand_Share"] = np.where(
            warehouse_week_demand > 0,
            work["Commercial_Forecast"] / warehouse_week_demand, 0.0)
        work["Warehouse_Allocated_Capacity"] = (
            work["Warehouse_Weekly_Capacity"] * work["Warehouse_Demand_Share"])

        work["Supply_Ceiling"] = np.minimum(work["Supplier_Allocated_Capacity"],
                                            work["Warehouse_Allocated_Capacity"])

        work["Supplier_Utilisation_Pct"] = np.round(
            np.where(work["Supplier_Weekly_Capacity"] > 0,
                     supplier_week_demand / work["Supplier_Weekly_Capacity"] * 100, 0.0), 2)
        work["Warehouse_Utilisation_Pct"] = np.round(
            np.where(work["Warehouse_Weekly_Capacity"] > 0,
                     warehouse_week_demand / work["Warehouse_Weekly_Capacity"] * 100, 0.0), 2)
        work["Is_Constrained"] = work["Commercial_Forecast"] > work["Supply_Ceiling"] + 1e-6
        return work

    # ------------------------------------------------------------------ #
    # Scenario application
    # ------------------------------------------------------------------ #
    def apply_scenario(self, plan: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Blend the statistical and commercial views under a named risk posture.

            consensus = (statistical + w * (commercial - statistical)) * bias

        `w` is how much of the commercial delta the scenario accepts and `bias`
        is the scenario's overall risk appetite. Conservative accepts about a
        third of the commercial ambition and shades the result down;
        Aggressive takes the commercial number at face value and adds headroom.
        The operational ceiling is then applied to all three -- you cannot ship
        what you cannot make.
        """
        if scenario not in self.cfg.scenario_commercial_weight:
            raise ValueError(f"Unknown scenario '{scenario}'. "
                             f"Valid: {list(self.cfg.scenario_commercial_weight)}")

        work = plan.copy()
        weight = self.cfg.scenario_commercial_weight[scenario]
        bias = self.cfg.scenario_bias[scenario]
        respect = self.cfg.scenario_respect_constraints[scenario]

        delta = work["Commercial_Forecast"] - work["Forecast_Units"]
        unconstrained = (work["Forecast_Units"] + weight * delta) * bias
        unconstrained = np.clip(unconstrained, 0.0, None)

        work["Scenario"] = scenario
        work["Unconstrained_Plan"] = np.round(unconstrained, 1)
        # Rounding is applied BEFORE the ceiling is re-imposed. Rounding a
        # constrained quantity up -- even by 0.05 of a unit -- publishes a plan
        # the supply base cannot meet, which is precisely the failure this
        # engine exists to prevent.
        if respect:
            capped = np.minimum(unconstrained, work["Supply_Ceiling"])
            work["Consensus_Units"] = np.minimum(np.round(capped, 1),
                                                 work["Supply_Ceiling"])
        else:
            work["Consensus_Units"] = np.round(unconstrained, 1)
        work["Constrained_Out_Units"] = np.round(
            np.clip(work["Unconstrained_Plan"] - work["Consensus_Units"], 0.0, None), 1)
        work["Consensus_Value"] = np.round(
            work["Consensus_Units"] * work["Unit_Price"], 2)
        work["Constrained_Out_Value"] = np.round(
            work["Constrained_Out_Units"] * work["Unit_Price"], 2)

        # Governance fields
        work["Override_Abs_Deviation"] = np.round(delta, 1)
        work["Override_Deviation_Pct"] = np.round(
            np.where(work["Forecast_Units"] > 1e-6,
                     delta / work["Forecast_Units"] * 100.0, 0.0), 2)
        work["Override_Alert"] = (work["Override_Deviation_Pct"].abs()
                                  > self.cfg.override_alert_threshold_pct)
        work["Requires_Board_Signoff"] = (work["Override_Deviation_Pct"].abs()
                                          > self.cfg.override_escalation_pct)
        work["Planner_Decision"] = np.select(
            [work["Requires_Board_Signoff"],
             work["Override_Alert"],
             work["Has_Commercial_Input"]],
            ["Escalate to S&OP Board", "Review - Justification Required", "Accepted"],
            default="No Override")
        work["Audit_Timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return work

    # ------------------------------------------------------------------ #
    # Audit and reporting
    # ------------------------------------------------------------------ #
    def build_override_log(self, plan: pd.DataFrame) -> pd.DataFrame:
        """
        The immutable override register.

        One row per SKU-week where a commercial input changed the number. This
        is the artefact an S&OP chair reads in the demand review and the one an
        auditor asks for when the plan misses.
        """
        log = plan[plan["Has_Commercial_Input"]].copy()
        if log.empty:
            return pd.DataFrame(columns=[
                Keys.SKU, "Statistical_Forecast", "Commercial_Override"])

        log = log.rename(columns={
            "Forecast_Units": "Statistical_Forecast",
            "Commercial_Forecast": "Commercial_Override",
        })
        columns = [Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, "horizon", Keys.WEEK,
                   "Statistical_Forecast", "Commercial_Override",
                   "Override_Abs_Deviation", "Override_Deviation_Pct",
                   "Commercial_Driver", "Justification", "Submitted_By",
                   "Override_Alert", "Requires_Board_Signoff", "Planner_Decision",
                   "Supply_Ceiling", "Consensus_Units", "Scenario",
                   "Audit_Timestamp"]
        log = log[[c for c in columns if c in log.columns]].copy()

        # A flagged override without a justification is a governance failure,
        # not a data quality nicety -- surface it explicitly.
        log["Justification_Provided"] = (log["Justification"].notna()
                                         & (log["Justification"].str.strip() != "-")
                                         & (log["Justification"].str.len() > 10))
        log["Governance_Status"] = np.select(
            [~log["Override_Alert"],
             log["Override_Alert"] & log["Justification_Provided"],
             log["Override_Alert"] & ~log["Justification_Provided"]],
            ["Within Tolerance", "Flagged - Justified", "BLOCKED - Justification Missing"],
            default="Within Tolerance")
        return log.sort_values("Override_Deviation_Pct",
                               key=abs, ascending=False).reset_index(drop=True)

    def build_scenario_summary(self, plan_by_scenario: Dict[str, pd.DataFrame]
                               ) -> pd.DataFrame:
        """Portfolio-level comparison of the three planning postures."""
        rows: List[Dict[str, object]] = []
        for scenario, frame in plan_by_scenario.items():
            statistical = frame["Forecast_Units"].sum()
            commercial = frame["Commercial_Forecast"].sum()
            rows.append({
                "Scenario": scenario,
                "Statistical_Units": round(statistical, 0),
                "Commercial_Units": round(commercial, 0),
                "Unconstrained_Units": round(frame["Unconstrained_Plan"].sum(), 0),
                "Consensus_Units": round(frame["Consensus_Units"].sum(), 0),
                "Constrained_Out_Units": round(frame["Constrained_Out_Units"].sum(), 0),
                "Consensus_Value": round(frame["Consensus_Value"].sum(), 0),
                "Constrained_Out_Value": round(frame["Constrained_Out_Value"].sum(), 0),
                "Uplift_vs_Statistical_Pct": round(
                    safe_div(frame["Consensus_Units"].sum() - statistical,
                             statistical) * 100, 2),
                "Constrained_SKU_Weeks": int(frame["Is_Constrained"].sum()),
                "Constrained_SKUs": int(
                    frame.loc[frame["Is_Constrained"], Keys.SKU].nunique()),
                "Override_Alerts": int(frame["Override_Alert"].sum()),
                "Board_Escalations": int(frame["Requires_Board_Signoff"].sum()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_waterfall(plan: pd.DataFrame) -> pd.DataFrame:
        """
        The consensus bridge the dashboard renders.

        Each step is a delta, and the steps sum exactly to the final consensus.
        The commercial delta is decomposed so that promotion/NPD-driven volume
        is visible separately from pure sales judgement -- these have very
        different credibility and the S&OP chair should be able to see which
        is which.
        """
        statistical = plan["Forecast_Units"].sum()

        promo_npd_drivers = {"Promotion Uplift", "New Product Introduction"}
        is_promo_npd = plan["Commercial_Driver"].isin(promo_npd_drivers)
        commercial_delta_all = plan["Commercial_Forecast"] - plan["Forecast_Units"]

        promo_npd_delta = commercial_delta_all[is_promo_npd].sum()
        other_commercial_delta = commercial_delta_all[~is_promo_npd].sum()

        # The scenario weighting scales the accepted portion of both deltas.
        accepted_total = plan["Unconstrained_Plan"].sum() - statistical
        raw_total = commercial_delta_all.sum()
        scale = safe_div(accepted_total, raw_total, default=0.0) if abs(raw_total) > 1e-9 else 0.0

        constraint_delta = -(plan["Unconstrained_Plan"].sum()
                             - plan["Consensus_Units"].sum())

        steps = [
            ("Statistical Baseline", statistical, "absolute"),
            ("Commercial Override", other_commercial_delta * scale, "relative"),
            ("Promotion / NPD", promo_npd_delta * scale, "relative"),
            ("Operational Constraint", constraint_delta, "relative"),
            ("Final Consensus", plan["Consensus_Units"].sum(), "total"),
        ]
        return pd.DataFrame(
            [{"Step": s, "Units": round(float(v), 0), "Measure": m} for s, v, m in steps])

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, forecast: pd.DataFrame, sku_master: pd.DataFrame,
            supplier_master: pd.DataFrame, warehouse_master: pd.DataFrame,
            scenario: Optional[str] = None) -> ConsensusResult:
        scenario = scenario or self.cfg.default_scenario

        commercial = self.generate_commercial_inputs(forecast, sku_master)
        constrained = self.build_operational_constraints(
            commercial, sku_master, supplier_master, warehouse_master)

        by_scenario = {name: self.apply_scenario(constrained, name)
                       for name in self.cfg.scenario_commercial_weight}
        primary = by_scenario[scenario]

        override_log = self.build_override_log(primary)
        summary = self.build_scenario_summary(by_scenario)
        waterfall = self.build_waterfall(primary)

        blocked = int((override_log["Governance_Status"]
                       == "BLOCKED - Justification Missing").sum()) if not override_log.empty else 0

        diagnostics = {
            "scenario": scenario,
            "sku_weeks": int(len(primary)),
            "skus_with_commercial_input": int(
                primary.loc[primary["Has_Commercial_Input"], Keys.SKU].nunique()),
            "override_alerts": int(primary["Override_Alert"].sum()),
            "board_escalations": int(primary["Requires_Board_Signoff"].sum()),
            "blocked_overrides": blocked,
            "constrained_sku_weeks": int(primary["Is_Constrained"].sum()),
            "constrained_out_units": float(primary["Constrained_Out_Units"].sum()),
            "constrained_out_value": float(primary["Constrained_Out_Value"].sum()),
            "consensus_units": float(primary["Consensus_Units"].sum()),
            "consensus_value": float(primary["Consensus_Value"].sum()),
            "scenarios": list(by_scenario),
        }

        logger.info("Consensus [%s]: %s units | %d override alerts | %d escalations | "
                    "%d constrained SKU-weeks (%.0f units constrained out)",
                    scenario, f"{diagnostics['consensus_units']:,.0f}",
                    diagnostics["override_alerts"], diagnostics["board_escalations"],
                    diagnostics["constrained_sku_weeks"],
                    diagnostics["constrained_out_units"])

        # Retain all scenarios in the plan so the dashboard can switch without
        # re-running the engine.
        all_scenarios = pd.concat(by_scenario.values(), ignore_index=True)
        return ConsensusResult(
            consensus_plan=all_scenarios,
            override_log=override_log,
            scenario_summary=summary,
            waterfall=waterfall,
            diagnostics=diagnostics,
        )


def run_consensus_planning(config: AppConfig, forecast: pd.DataFrame,
                           sku_master: pd.DataFrame, supplier_master: pd.DataFrame,
                           warehouse_master: pd.DataFrame,
                           scenario: Optional[str] = None) -> ConsensusResult:
    """Functional entry point used by the pipeline."""
    return ConsensusPlanner(config).run(
        forecast, sku_master, supplier_master, warehouse_master, scenario)
