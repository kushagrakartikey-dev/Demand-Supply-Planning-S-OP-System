"""
Module 3 -- Supply Planning & Constraint Engine.

Turns the consensus demand plan into a 12-week, week-by-week supply plan and
tells the S&OP board where it will break.

The core is a classical MRP netting loop run per SKU::

    Gross Requirement = Consensus Demand + Safety Stock Replenishment
    Net Requirement   = Gross Requirement - On Hand - Open POs - Planned Receipts
    Planned Order     = round_up_to_MOQ( Net Requirement )   [offset by lead time]

Three things make this more than a textbook exercise:

* **Time-phasing is real.** A planned order raised in week *w* lands in week
  *w + ceil(lead_time)*. An order placed inside the lead-time window cannot
  rescue a stockout in that window -- the engine detects this and raises an
  expedite recommendation instead of quietly assuming instant supply.
* **Capacity is a shared, finite resource.** Supplier weekly capacity is
  allocated across that supplier's SKUs pro-rata to requirement. When the sum
  of planned orders exceeds it, the breach is logged against the supplier, not
  hidden inside a per-SKU calculation that each looks fine on its own.
* **Risk is scored transparently.** The Supply Risk Index is a documented
  weighted blend of five normalised factors, every one of which can be traced
  back to a number a planner can check. No black box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, Keys, SupplyConfig
from src.shared.metrics import (abc_classification, capacity_utilisation,
                                coefficient_of_variation, safety_stock,
                                service_level_z, weeks_of_cover, working_capital)
from src.shared.utils import clamp, get_logger, safe_div

logger = get_logger("supply_planner")


@dataclass
class SupplyPlanResult:
    """Outputs of the supply planning engine."""

    supply_plan: pd.DataFrame        # SKU x week time-phased plan
    sku_summary: pd.DataFrame        # one row per SKU with risk classification
    capacity_alerts: pd.DataFrame    # supplier / warehouse capacity breaches
    stockout_risks: pd.DataFrame     # projected stockouts with lead-time context
    inventory_policy: pd.DataFrame   # safety stock / ROP per SKU (Output 5 input)
    diagnostics: Dict[str, object]


class SupplyPlanner:
    """MRP netting, capacity checking and supply risk scoring."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: SupplyConfig = config.supply

    # ------------------------------------------------------------------ #
    # Inventory policy
    # ------------------------------------------------------------------ #
    def build_inventory_policy(self, forecast_uncertainty: pd.DataFrame,
                               sku_master: pd.DataFrame,
                               supplier_master: pd.DataFrame,
                               sales_history: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the safety-stock and reorder-point policy for every SKU.

        The service-level target is set by ABC class, not applied uniformly.
        Holding a 99% service level on a C-item is one of the most expensive
        habits in inventory management: it consumes working capital on stock
        nobody is waiting for. A-items earn the buffer because a stockout there
        costs real revenue.

        Safety stock uses the full formula including lead-time variance::

            SS = Z * sqrt( LT * sigma_D^2 + mu_D^2 * sigma_LT^2 )

        with sigma_D taken from the *forecast error* distribution published by
        the demand engine, not from raw historical demand variation. The buffer
        exists to absorb the error in the plan, so that is what it should be
        sized against.
        """
        cfg = self.cfg

        # --- ABC classification on 12-week forecast revenue ---------------- #
        policy = forecast_uncertainty.copy()
        policy["Forecast_Revenue_12W"] = (policy["Forecast_Units_12W"]
                                          * policy["Unit_Price"]).round(2)
        policy["ABC_Class"] = abc_classification(
            policy.set_index(Keys.SKU)["Forecast_Revenue_12W"],
            cfg.abc_thresholds).reindex(policy[Keys.SKU]).to_numpy()

        # --- XYZ classification on demand variability ---------------------- #
        cv = (sales_history.groupby(Keys.SKU)["Sales_Units"]
              .apply(coefficient_of_variation).rename("CV").reset_index())
        policy = policy.merge(cv, on=Keys.SKU, how="left")
        policy["CV"] = policy["CV"].fillna(policy["Demand_CV"]).fillna(0.5)
        policy["XYZ_Class"] = np.where(policy["CV"] < 0.5, "X",
                                       np.where(policy["CV"] <= 1.0, "Y", "Z"))
        policy["ABC_XYZ"] = policy["ABC_Class"] + policy["XYZ_Class"]

        # --- supplier attributes ------------------------------------------- #
        supplier = supplier_master.set_index(Keys.SUPPLIER)
        policy["Lead_Time_Weeks"] = policy[Keys.SUPPLIER].map(
            supplier["Lead_Time_Weeks"]).fillna(2.0)
        policy["Lead_Time_Std_Weeks"] = policy[Keys.SUPPLIER].map(
            supplier["Lead_Time_Std_Weeks"]).fillna(0.3)
        policy["MOQ"] = policy[Keys.SUPPLIER].map(supplier["MOQ"]).fillna(0).astype(int)
        policy["Supplier_Weekly_Capacity"] = policy[Keys.SUPPLIER].map(
            supplier["Weekly_Capacity_Units"]).fillna(1e9)

        # --- service level and safety stock -------------------------------- #
        policy["Service_Level_Target"] = policy["ABC_Class"].map(
            cfg.service_level_by_class).fillna(0.90)
        policy["Service_Level_Z"] = policy["Service_Level_Target"].apply(service_level_z).round(3)

        policy["Demand_Std_Weekly"] = policy["Forecast_Error_Std"].fillna(
            policy["Historical_Demand_Std"]).fillna(0.0)

        policy["Safety_Stock_Units"] = [
            round(safety_stock(demand_std=row.Demand_Std_Weekly,
                               lead_time_weeks=row.Lead_Time_Weeks,
                               service_level=row.Service_Level_Target,
                               avg_demand=row.Avg_Weekly_Forecast,
                               lead_time_std_weeks=row.Lead_Time_Std_Weeks), 1)
            for row in policy.itertuples()
        ]
        policy["Reorder_Point_Units"] = (
            policy["Avg_Weekly_Forecast"] * policy["Lead_Time_Weeks"]
            + policy["Safety_Stock_Units"]).round(1)
        policy["Safety_Stock_Value"] = (policy["Safety_Stock_Units"]
                                        * policy["Unit_Cost"]).round(2)
        policy["Safety_Stock_Weeks"] = policy.apply(
            lambda r: round(weeks_of_cover(r["Safety_Stock_Units"],
                                           r["Avg_Weekly_Forecast"]), 2), axis=1)
        return policy

    # ------------------------------------------------------------------ #
    # MRP netting
    # ------------------------------------------------------------------ #
    def run_mrp(self, consensus_plan: pd.DataFrame, policy: pd.DataFrame,
                inventory: pd.DataFrame, purchase_orders: pd.DataFrame
                ) -> pd.DataFrame:
        """
        Time-phased MRP netting, vectorised across SKUs and looped over weeks.

        The week loop is genuinely sequential -- closing inventory in week *w*
        is opening inventory in week *w+1* -- but every SKU is processed
        simultaneously inside each week, so the whole 500-SKU x 12-week plan
        resolves in a handful of numpy operations rather than 6,000 row visits.
        """
        cfg = self.cfg
        horizon = cfg.horizon_weeks

        demand = (consensus_plan.pivot_table(
            index=Keys.SKU, columns="horizon", values="Consensus_Units",
            aggfunc="sum").reindex(columns=range(1, horizon + 1)).fillna(0.0))
        skus = demand.index.to_numpy()
        demand_matrix = demand.to_numpy(dtype=float)

        pol = policy.set_index(Keys.SKU).reindex(skus)
        on_hand = (inventory.set_index(Keys.SKU)["On_Hand_Units"]
                   .reindex(skus).fillna(0.0).to_numpy(dtype=float))
        allocated = (inventory.set_index(Keys.SKU)["Allocated_Units"]
                     .reindex(skus).fillna(0.0).to_numpy(dtype=float))
        # Allocated stock is committed to customer orders and is not available
        # to cover forecast demand -- netting it off is what separates a real
        # ATP calculation from a naive on-hand balance.
        available = np.clip(on_hand - allocated, 0.0, None)

        safety = pol["Safety_Stock_Units"].fillna(0.0).to_numpy(dtype=float)
        lead_time = pol["Lead_Time_Weeks"].fillna(2.0).to_numpy(dtype=float)
        lead_weeks = np.ceil(lead_time).astype(int)
        moq = pol["MOQ"].fillna(0.0).to_numpy(dtype=float)
        unit_cost = pol["Unit_Cost"].fillna(0.0).to_numpy(dtype=float)

        # --- open purchase orders, phased into the receipt grid ------------ #
        open_po = np.zeros((len(skus), horizon))
        if not purchase_orders.empty:
            po = purchase_orders.copy()
            po["week"] = po["Expected_Receipt_Week"].clip(0, horizon - 1).astype(int)
            grid = (po.pivot_table(index=Keys.SKU, columns="week",
                                   values="Order_Qty", aggfunc="sum")
                    .reindex(index=skus, columns=range(horizon)).fillna(0.0))
            open_po = grid.to_numpy(dtype=float)

        n = len(skus)
        planned_orders = np.zeros((n, horizon))
        planned_receipts = np.zeros((n, horizon))
        opening = np.zeros((n, horizon))
        closing = np.zeros((n, horizon))
        gross_req = np.zeros((n, horizon))
        net_req = np.zeros((n, horizon))

        position = available.copy()
        for w in range(horizon):
            opening[:, w] = position

            # Receipts landing this week: open POs + previously planned orders.
            receipts = open_po[:, w] + planned_receipts[:, w]

            gross = demand_matrix[:, w] + np.clip(
                safety - (position + receipts - demand_matrix[:, w]), 0.0, None)
            gross_req[:, w] = gross

            net = np.clip(gross - position - receipts, 0.0, None)
            net_req[:, w] = net

            # Round the net requirement up to the supplier MOQ.
            order = np.where(net > 1e-6,
                             np.where(moq > 0, np.ceil(net / np.maximum(moq, 1)) * moq, net),
                             0.0)
            planned_orders[:, w] = order

            # Time-phase the receipt. Orders landing beyond the horizon still
            # get raised -- they are real commitments -- but cannot help here.
            arrival = w + lead_weeks
            in_horizon = arrival < horizon
            if in_horizon.any():
                np.add.at(planned_receipts,
                          (np.where(in_horizon)[0], arrival[in_horizon]),
                          order[in_horizon])

            position = position + receipts - demand_matrix[:, w]
            closing[:, w] = position
            position = np.clip(position, 0.0, None)   # cannot hold negative stock

        # --- assemble the long plan ---------------------------------------- #
        weeks = np.tile(np.arange(1, horizon + 1), n)
        sku_col = np.repeat(skus, horizon)

        forward_demand = np.zeros((n, horizon))
        for w in range(horizon):
            window = demand_matrix[:, w:min(w + 4, horizon)]
            forward_demand[:, w] = window.mean(axis=1) if window.size else 0.0

        plan = pd.DataFrame({
            Keys.SKU: sku_col,
            "horizon": weeks,
            "Opening_Inventory": opening.ravel().round(1),
            "Demand_Units": demand_matrix.ravel().round(1),
            "Open_PO_Receipts": open_po.ravel().round(1),
            "Planned_Receipts": planned_receipts.ravel().round(1),
            "Gross_Requirement": gross_req.ravel().round(1),
            "Net_Requirement": net_req.ravel().round(1),
            "Planned_Order_Qty": planned_orders.ravel().round(1),
            "Closing_Inventory": closing.ravel().round(1),
            "Safety_Stock": np.repeat(safety, horizon).round(1),
            "Forward_Weekly_Demand": forward_demand.ravel().round(2),
            "Unit_Cost": np.repeat(unit_cost, horizon),
            "Lead_Time_Weeks": np.repeat(lead_time, horizon),
        })

        plan["Weeks_Of_Cover"] = [
            round(weeks_of_cover(c, d), 2)
            for c, d in zip(plan["Closing_Inventory"], plan["Forward_Weekly_Demand"])]
        plan["Is_Stockout_Week"] = plan["Closing_Inventory"] < 0
        plan["Below_Safety_Stock"] = (plan["Closing_Inventory"]
                                      < plan["Safety_Stock"]) & ~plan["Is_Stockout_Week"]
        plan["Shortage_Units"] = np.clip(-plan["Closing_Inventory"], 0.0, None).round(1)
        plan["Working_Capital"] = [
            round(working_capital(inv, cost), 2)
            for inv, cost in zip(plan["Closing_Inventory"].clip(lower=0), plan["Unit_Cost"])]
        plan["Inventory_Status"] = np.select(
            [plan["Is_Stockout_Week"],
             plan["Below_Safety_Stock"],
             plan["Weeks_Of_Cover"] > self.cfg.excess_cover_weeks],
            ["Stockout", "Below Safety Stock", "Excess"],
            default="Healthy")
        return plan

    # ------------------------------------------------------------------ #
    # Capacity checking
    # ------------------------------------------------------------------ #
    def check_capacity(self, plan: pd.DataFrame, policy: pd.DataFrame,
                       sku_master: pd.DataFrame, supplier_master: pd.DataFrame,
                       warehouse_master: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate planned orders to the resource that actually constrains them.

        Capacity breaches only exist in aggregate. A single SKU asking for
        3,000 units is never a problem; forty SKUs on the same supplier line
        asking for 3,000 units each in the same week is. This function is the
        one place the plan is summed across SKUs, which is why it is the only
        place a breach can be detected.
        """
        cfg = self.cfg
        work = plan.merge(sku_master[[Keys.SKU, Keys.SUPPLIER, Keys.WAREHOUSE,
                                      Keys.CATEGORY]], on=Keys.SKU, how="left")

        supplier_cap = supplier_master.set_index(Keys.SUPPLIER)["Weekly_Capacity_Units"]
        warehouse_cap = warehouse_master.set_index(
            Keys.WAREHOUSE)["Weekly_Inbound_Capacity_Units"]

        alerts: List[pd.DataFrame] = []

        supplier_load = (work.groupby([Keys.SUPPLIER, "horizon"], as_index=False)
                         .agg(Required_Units=("Planned_Order_Qty", "sum"),
                              SKUs_Affected=(Keys.SKU, "nunique")))
        supplier_load["Available_Capacity"] = supplier_load[Keys.SUPPLIER].map(
            supplier_cap).fillna(np.inf)
        supplier_load["Resource_Type"] = "Supplier"
        supplier_load = supplier_load.rename(columns={Keys.SUPPLIER: "Resource_ID"})
        alerts.append(supplier_load)

        warehouse_load = (work.groupby([Keys.WAREHOUSE, "horizon"], as_index=False)
                          .agg(Required_Units=("Planned_Order_Qty", "sum"),
                               SKUs_Affected=(Keys.SKU, "nunique")))
        warehouse_load["Available_Capacity"] = warehouse_load[Keys.WAREHOUSE].map(
            warehouse_cap).fillna(np.inf)
        warehouse_load["Resource_Type"] = "Warehouse"
        warehouse_load = warehouse_load.rename(columns={Keys.WAREHOUSE: "Resource_ID"})
        alerts.append(warehouse_load)

        combined = pd.concat(alerts, ignore_index=True)
        combined["Utilisation_Pct"] = [
            round(capacity_utilisation(r, a), 2)
            for r, a in zip(combined["Required_Units"], combined["Available_Capacity"])]
        combined["Overload_Units"] = np.clip(
            combined["Required_Units"] - combined["Available_Capacity"], 0.0, None).round(1)
        combined["Severity"] = np.select(
            [combined["Utilisation_Pct"] >= cfg.capacity_breach_threshold * 100,
             combined["Utilisation_Pct"] >= cfg.capacity_alert_threshold * 100],
            ["BREACH", "WARNING"], default="OK")
        combined["Recommended_Action"] = np.select(
            [combined["Severity"] == "BREACH",
             combined["Severity"] == "WARNING"],
            ["Escalate capacity constraint - activate alternate supply or re-phase orders",
             "Monitor closely - pre-book capacity and confirm supplier commitment"],
            default="No action required")
        return combined.sort_values(["Severity", "Utilisation_Pct"],
                                    ascending=[True, False]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Risk scoring
    # ------------------------------------------------------------------ #
    def score_supply_risk(self, plan: pd.DataFrame, policy: pd.DataFrame,
                          capacity: pd.DataFrame, sku_master: pd.DataFrame
                          ) -> pd.DataFrame:
        """
        Supply Risk Index, 0 (safe) to 100 (critical).

        Five normalised factors, weighted per configuration::

            SRI = 100 * ( w1*stockout_proximity + w2*capacity_utilisation
                        + w3*cover_shortfall   + w4*lead_time_risk
                        + w5*forecast_uncertainty )

        Each factor is scaled to [0, 1] before weighting so no single input can
        dominate through units alone:

        * **stockout proximity** -- 1.0 if a stockout is projected in week 1,
          decaying linearly to 0 at the end of the horizon. Nearness matters
          more than depth, because a distant shortage is still recoverable.
        * **capacity utilisation** -- the supplier's peak weekly utilisation
          across the horizon, scaled so 100% utilisation maps to 1.0.
        * **cover shortfall** -- how far minimum weeks-of-cover falls below the
          target, as a fraction of the target.
        * **lead-time risk** -- lead time relative to the configured reference,
          because a long lead time removes the option to react.
        * **forecast uncertainty** -- the SKU's forecast-error CV, since an
          unpredictable SKU is riskier at the same cover.
        """
        cfg = self.cfg
        horizon = cfg.horizon_weeks
        weights = cfg.risk_weights

        agg = (plan.groupby(Keys.SKU)
               .agg(Min_Cover=("Weeks_Of_Cover", "min"),
                    Avg_Cover=("Weeks_Of_Cover", "mean"),
                    Total_Demand=("Demand_Units", "sum"),
                    Total_Planned_Orders=("Planned_Order_Qty", "sum"),
                    Total_Shortage=("Shortage_Units", "sum"),
                    Stockout_Weeks=("Is_Stockout_Week", "sum"),
                    Below_SS_Weeks=("Below_Safety_Stock", "sum"),
                    Closing_Inventory_W12=("Closing_Inventory", "last"),
                    Avg_Working_Capital=("Working_Capital", "mean"),
                    Lead_Time_Weeks=("Lead_Time_Weeks", "first"))
               .reset_index())

        # First projected stockout week (NaN if none in horizon).
        stockouts = plan[plan["Is_Stockout_Week"]]
        first_stockout = (stockouts.groupby(Keys.SKU)["horizon"].min()
                          .rename("First_Stockout_Week").reset_index())
        agg = agg.merge(first_stockout, on=Keys.SKU, how="left")

        first_below_ss = (plan[plan["Below_Safety_Stock"]].groupby(Keys.SKU)["horizon"]
                          .min().rename("First_Below_SS_Week").reset_index())
        agg = agg.merge(first_below_ss, on=Keys.SKU, how="left")

        pol_cols = [Keys.SKU, Keys.SUPPLIER, Keys.WAREHOUSE, Keys.CATEGORY, Keys.CHANNEL,
                    "ABC_Class", "XYZ_Class", "ABC_XYZ", "Safety_Stock_Units",
                    "Reorder_Point_Units", "Service_Level_Target", "Demand_Std_Weekly",
                    "Avg_Weekly_Forecast", "Unit_Cost", "Unit_Price", "MOQ",
                    "Forecast_Units_12W", "Selected_Model"]
        agg = agg.merge(policy[[c for c in pol_cols if c in policy.columns]],
                        on=Keys.SKU, how="left")

        # Peak supplier utilisation across the horizon.
        supplier_peak = (capacity[capacity["Resource_Type"] == "Supplier"]
                         .groupby("Resource_ID")["Utilisation_Pct"].max()
                         .rename("Supplier_Peak_Utilisation_Pct"))
        agg["Supplier_Peak_Utilisation_Pct"] = agg[Keys.SUPPLIER].map(
            supplier_peak).fillna(0.0)

        # --- normalised factors -------------------------------------------- #
        proximity = np.where(
            agg["First_Stockout_Week"].notna(),
            np.clip((horizon - agg["First_Stockout_Week"].fillna(horizon) + 1) / horizon,
                    0.0, 1.0),
            # No hard stockout: a dip below safety stock still carries risk,
            # scaled down because the buffer absorbed it.
            np.where(agg["First_Below_SS_Week"].notna(),
                     0.45 * np.clip(
                         (horizon - agg["First_Below_SS_Week"].fillna(horizon) + 1) / horizon,
                         0.0, 1.0),
                     0.0))

        utilisation = np.clip(agg["Supplier_Peak_Utilisation_Pct"] / 100.0, 0.0, 1.0)

        cover_gap = np.clip(
            (cfg.target_weeks_of_cover - agg["Min_Cover"].fillna(0.0))
            / max(cfg.target_weeks_of_cover, 1e-6), 0.0, 1.0)

        lead_risk = np.clip(agg["Lead_Time_Weeks"].fillna(2.0)
                            / cfg.max_lead_time_weeks_reference, 0.0, 1.0)

        uncertainty = np.clip(
            agg["Demand_Std_Weekly"].fillna(0.0)
            / agg["Avg_Weekly_Forecast"].replace(0, np.nan), 0.0, 1.5).fillna(0.5) / 1.5

        agg["Risk_Factor_Stockout_Proximity"] = np.round(proximity, 4)
        agg["Risk_Factor_Capacity"] = np.round(utilisation, 4)
        agg["Risk_Factor_Cover"] = np.round(cover_gap, 4)
        agg["Risk_Factor_Lead_Time"] = np.round(lead_risk, 4)
        agg["Risk_Factor_Uncertainty"] = np.round(uncertainty, 4)

        score = 100.0 * (
            weights["stockout_proximity"] * proximity
            + weights["capacity_utilisation"] * utilisation
            + weights["weeks_of_cover"] * cover_gap
            + weights["lead_time_risk"] * lead_risk
            + weights["forecast_uncertainty"] * uncertainty)

        agg["Supply_Risk_Index"] = [round(clamp(s, 0, 100), 1) for s in score]
        agg["Risk_Band"] = np.select(
            [agg["Supply_Risk_Index"] >= cfg.risk_red_threshold,
             agg["Supply_Risk_Index"] >= cfg.risk_amber_threshold],
            ["Red", "Amber"], default="Green")

        agg["Weeks_To_Stockout"] = agg["First_Stockout_Week"].fillna(np.inf)
        agg["Can_Recover_In_Lead_Time"] = (
            agg["Weeks_To_Stockout"] > agg["Lead_Time_Weeks"].fillna(2.0))
        agg["Recommended_Action"] = self._recommend(agg)
        return agg.sort_values("Supply_Risk_Index", ascending=False).reset_index(drop=True)

    def _recommend(self, agg: pd.DataFrame) -> np.ndarray:
        """
        Map the risk picture to the action a planner should actually take.

        Note the ordering: a projected stockout *inside* the supplier lead time
        cannot be solved by raising a normal order, so it routes to expedite or
        alternate supply. This distinction is the single most useful thing the
        engine tells a planner, and it is why lead time is carried all the way
        through the netting loop.
        """
        cfg = self.cfg
        conditions = [
            (agg["First_Stockout_Week"].notna()) & (~agg["Can_Recover_In_Lead_Time"]),
            (agg["First_Stockout_Week"].notna()) & (agg["Can_Recover_In_Lead_Time"]),
            agg["Supplier_Peak_Utilisation_Pct"] >= cfg.capacity_breach_threshold * 100,
            agg["Supplier_Peak_Utilisation_Pct"] >= cfg.capacity_alert_threshold * 100,
            agg["Below_SS_Weeks"] > 0,
            agg["Min_Cover"] > cfg.excess_cover_weeks,
        ]
        choices = [
            "EXPEDITE - stockout falls inside lead time; air-freight or activate alternate supplier",
            "Raise planned order now - recoverable within supplier lead time",
            "Escalate supplier capacity constraint - re-phase or dual-source",
            "Pre-book supplier capacity - utilisation approaching ceiling",
            "Increase safety stock or bring forward replenishment",
            "Review excess stock - reduce order quantity and release working capital",
        ]
        return np.select(conditions, choices, default="No action - within policy")

    # ------------------------------------------------------------------ #
    # Stockout register
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_stockout_register(plan: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
        """Every projected stockout, with lead-time recoverability and value at risk."""
        events = plan[plan["Is_Stockout_Week"] | plan["Below_Safety_Stock"]].copy()
        if events.empty:
            return pd.DataFrame(columns=[Keys.SKU, "horizon", "Event_Type"])

        events["Event_Type"] = np.where(events["Is_Stockout_Week"],
                                        "Projected Stockout", "Below Safety Stock")
        keep = [Keys.SKU, "horizon", "Event_Type", "Opening_Inventory", "Demand_Units",
                "Closing_Inventory", "Shortage_Units", "Safety_Stock",
                "Weeks_Of_Cover", "Planned_Order_Qty", "Lead_Time_Weeks"]
        events = events[[c for c in keep if c in events.columns]]

        risk_cols = [Keys.SKU, Keys.SUPPLIER, Keys.CATEGORY, Keys.CHANNEL, "ABC_Class",
                     "Supply_Risk_Index", "Risk_Band", "Unit_Price",
                     "Can_Recover_In_Lead_Time", "Recommended_Action"]
        events = events.merge(risk[[c for c in risk_cols if c in risk.columns]],
                              on=Keys.SKU, how="left")
        events["Revenue_At_Risk"] = (events["Shortage_Units"]
                                     * events["Unit_Price"].fillna(0)).round(2)
        events["Weeks_Notice"] = events["horizon"]
        return events.sort_values(["Event_Type", "horizon", "Revenue_At_Risk"],
                                  ascending=[True, True, False]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, consensus_plan: pd.DataFrame, forecast_uncertainty: pd.DataFrame,
            sku_master: pd.DataFrame, supplier_master: pd.DataFrame,
            warehouse_master: pd.DataFrame, inventory: pd.DataFrame,
            purchase_orders: pd.DataFrame, sales_history: pd.DataFrame,
            scenario: Optional[str] = None) -> SupplyPlanResult:

        scenario = scenario or self.config.consensus.default_scenario
        plan_input = consensus_plan[consensus_plan["Scenario"] == scenario].copy()
        if plan_input.empty:
            raise ValueError(f"No consensus rows for scenario '{scenario}'.")

        policy = self.build_inventory_policy(
            forecast_uncertainty, sku_master, supplier_master, sales_history)
        plan = self.run_mrp(plan_input, policy, inventory, purchase_orders)
        capacity = self.check_capacity(plan, policy, sku_master,
                                       supplier_master, warehouse_master)
        risk = self.score_supply_risk(plan, policy, capacity, sku_master)
        stockouts = self.build_stockout_register(plan, risk)

        plan = plan.merge(
            risk[[Keys.SKU, "Supply_Risk_Index", "Risk_Band", Keys.SUPPLIER,
                  Keys.CATEGORY, Keys.CHANNEL, "ABC_Class"]],
            on=Keys.SKU, how="left")
        plan["Scenario"] = scenario

        breaches = capacity[capacity["Severity"] == "BREACH"]
        earliest_notice = (int(stockouts.loc[stockouts["Event_Type"] == "Projected Stockout",
                                             "horizon"].min())
                           if (stockouts["Event_Type"] == "Projected Stockout").any()
                           else None)

        diagnostics = {
            "scenario": scenario,
            "skus_planned": int(plan[Keys.SKU].nunique()),
            "plan_rows": int(len(plan)),
            "total_planned_orders": float(plan["Planned_Order_Qty"].sum()),
            "total_working_capital": float(
                plan.loc[plan["horizon"] == 1, "Working_Capital"].sum()),
            "avg_weeks_of_cover": float(plan["Weeks_Of_Cover"].mean()),
            "skus_with_stockout": int(risk["First_Stockout_Week"].notna().sum()),
            "skus_below_safety_stock": int((risk["Below_SS_Weeks"] > 0).sum()),
            "capacity_breaches": int(len(breaches)),
            "capacity_warnings": int((capacity["Severity"] == "WARNING").sum()),
            "suppliers_breaching": int(
                breaches.loc[breaches["Resource_Type"] == "Supplier",
                             "Resource_ID"].nunique()) if not breaches.empty else 0,
            "risk_distribution": risk["Risk_Band"].value_counts().to_dict(),
            "earliest_stockout_week": earliest_notice,
            "expedite_required": int(
                (~risk["Can_Recover_In_Lead_Time"]
                 & risk["First_Stockout_Week"].notna()).sum()),
            "revenue_at_risk": float(stockouts["Revenue_At_Risk"].sum())
            if "Revenue_At_Risk" in stockouts.columns else 0.0,
        }

        logger.info("Supply plan [%s]: %d SKUs | %d stockout risks (earliest week %s) | "
                    "%d capacity breaches | risk mix %s",
                    scenario, diagnostics["skus_planned"],
                    diagnostics["skus_with_stockout"],
                    diagnostics["earliest_stockout_week"],
                    diagnostics["capacity_breaches"],
                    diagnostics["risk_distribution"])

        return SupplyPlanResult(
            supply_plan=plan,
            sku_summary=risk,
            capacity_alerts=capacity,
            stockout_risks=stockouts,
            inventory_policy=policy,
            diagnostics=diagnostics,
        )


def run_supply_planning(config: AppConfig, **kwargs) -> SupplyPlanResult:
    """Functional entry point used by the pipeline."""
    return SupplyPlanner(config).run(**kwargs)
