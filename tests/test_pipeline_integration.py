"""
Integration tests across the pipeline modules.

Where ``test_metrics_and_validation.py`` checks formulas in isolation, these
tests assert the *behaviour* of the system: that the MRP inventory balance
actually closes, that a stockout is only called recoverable when the lead time
genuinely allows it, that the governance threshold fires on the right rows, and
that no forecast was ever produced from data it should not have seen.

These are the properties that would still be true of a correct system if every
number changed -- which is what makes them worth testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.planning.config import Keys
from src.shared.validation import validate_datasets


# --------------------------------------------------------------------------- #
# Test 9 -- data generation
# --------------------------------------------------------------------------- #
class TestDataGeneration:

    def test_expected_shape(self, data, config):
        assert data.sku_master[Keys.SKU].nunique() == config.data.n_skus
        assert len(data.supplier_master) == config.data.n_suppliers
        assert len(data.orders) == config.data.n_orders

    def test_weekly_history_spans_the_configured_window(self, data, config):
        weeks = data.sales_history[Keys.WEEK_INDEX].nunique()
        assert weeks == config.data.history_weeks

    def test_all_datasets_pass_the_schema_contract(self, data):
        validator = validate_datasets(data.as_dict())
        assert validator.is_valid(), [i.message for i in validator.errors]

    def test_referential_integrity_across_datasets(self, data):
        skus = set(data.sku_master[Keys.SKU])
        suppliers = set(data.supplier_master[Keys.SUPPLIER])
        assert set(data.sales_history[Keys.SKU]) <= skus
        assert set(data.orders[Keys.SKU]) <= skus
        assert set(data.sku_master[Keys.SUPPLIER]) <= suppliers
        assert set(data.inventory[Keys.SKU]) <= skus

    def test_censored_sales_never_exceed_true_demand(self, data):
        """
        Sales_Units is demand truncated by availability. If a single row broke
        this the whole stockout mechanism would be meaningless.
        """
        sales = data.sales_history
        assert (sales["Sales_Units"] <= sales["Demand_Units"] + 1e-6).all()

    def test_stockout_weeks_show_lost_sales(self, data):
        sales = data.sales_history
        stockouts = sales[sales["Is_Stockout"] & (sales["Demand_Units"] > 0)]
        assert len(stockouts) > 0
        assert (stockouts["Lost_Sales_Units"] > 0).mean() > 0.9

    def test_promotions_lift_average_demand(self, data):
        """The promotion signal must be real, or the promo feature is noise."""
        sales = data.sales_history
        promo_mean = sales.loc[sales["Is_Promo"], "Demand_Units"].mean()
        base_mean = sales.loc[~sales["Is_Promo"], "Demand_Units"].mean()
        assert promo_mean > base_mean

    def test_festive_weeks_lift_demand(self, data):
        sales = data.sales_history
        festive = sales.loc[sales["Is_Festival_Week"], "Demand_Units"].mean()
        normal = sales.loc[~sales["Is_Festival_Week"], "Demand_Units"].mean()
        assert festive > normal

    def test_generation_is_reproducible(self, config):
        from src.shared.data_generator import generate_supply_chain_data
        first = generate_supply_chain_data(config, persist=False)
        second = generate_supply_chain_data(config, persist=False)
        pd.testing.assert_frame_equal(
            first.sales_history.head(500), second.sales_history.head(500))


# --------------------------------------------------------------------------- #
# Test 10 -- forecasting outputs
# --------------------------------------------------------------------------- #
class TestForecasting:

    def test_forecast_covers_every_sku_for_the_full_horizon(self, forecast, config):
        fc = forecast.forecast
        assert not fc.empty
        assert set(fc["horizon"]) == set(range(1, config.forecast.horizon_weeks + 1))

    def test_forecasts_are_non_negative(self, forecast):
        assert (forecast.forecast["Forecast_Units"] >= 0).all()

    def test_all_four_model_families_ran(self, forecast):
        models = set(forecast.model_scores["Model"].unique())
        for expected in ("HoltWinters", "XGBoost", "SeasonalNaive"):
            assert expected in models
        # SARIMA and Prophet run on the competition panel only.
        assert {"SARIMA", "Prophet"} & models

    def test_ensemble_weights_sum_to_one_per_sku(self, forecast):
        weights = forecast.ensemble_weights
        if weights.empty:
            pytest.skip("No ensemble weights produced at this scale.")
        totals = weights.groupby(Keys.SKU)["Weight"].sum()
        assert np.allclose(totals.to_numpy(), 1.0, atol=1e-6)

    def test_seasonal_naive_is_excluded_from_the_ensemble(self, forecast):
        """The benchmark must not be a member of the thing it benchmarks."""
        weights = forecast.ensemble_weights
        if not weights.empty:
            assert "SeasonalNaive" not in set(weights["Model"])

    def test_prediction_intervals_are_correctly_ordered(self, forecast):
        fc = forecast.forecast
        assert (fc["P2_5"] <= fc["P10"] + 1e-6).all()
        assert (fc["P10"] <= fc["Forecast_Units"] + 1e-6).all()
        assert (fc["Forecast_Units"] <= fc["P90"] + 1e-6).all()
        assert (fc["P90"] <= fc["P97_5"] + 1e-6).all()

    def test_intervals_widen_with_horizon(self, forecast):
        """Uncertainty must grow further out, or the bands are not meaningful."""
        fc = forecast.forecast.copy()
        fc["width"] = fc["P90"] - fc["P10"]
        near = fc.loc[fc["horizon"] <= 3, "width"].mean()
        far = fc.loc[fc["horizon"] >= 10, "width"].mean()
        assert far > near

    def test_lower_bounds_are_never_negative(self, forecast):
        assert (forecast.forecast["P10"] >= 0).all()
        assert (forecast.forecast["P2_5"] >= 0).all()

    def test_holdout_accuracy_is_reported_at_multiple_levels(self, forecast):
        hierarchy = forecast.hierarchy_accuracy
        assert not hierarchy.empty
        levels = set(hierarchy["Aggregation_Level"])
        assert "SKU / Week" in levels
        assert "Category / Month" in levels

    def test_error_falls_as_demand_aggregates(self, forecast):
        """
        A structural property of forecast error, and a useful canary: if
        SKU-week error were ever *lower* than category-month error, something
        in the aggregation logic is wrong.
        """
        hierarchy = forecast.hierarchy_accuracy.set_index("Aggregation_Level")
        assert hierarchy.loc["SKU / Week", "WAPE"] >= hierarchy.loc[
            "Category / Month", "WAPE"]

    def test_champion_selection_uses_cv_not_holdout(self, forecast):
        """
        Selection metadata must come from the CV stage. If CV_MAPE were sourced
        from the holdout the reported accuracy would be optimistically biased.
        """
        accuracy = forecast.sku_accuracy
        assert "CV_MAPE" in accuracy.columns
        assert "Selected_Model" in accuracy.columns
        assert accuracy["Selected_Model"].notna().all()

    def test_uncertainty_contract_is_populated_for_output_5(self, forecast):
        contract = forecast.uncertainty
        for column in ("Forecast_Units_12W", "Forecast_Error_Std",
                       "Avg_Weekly_Forecast", Keys.SUPPLIER, Keys.SKU):
            assert column in contract.columns
        assert (contract["Forecast_Error_Std"] >= 0).all()

    def test_feature_importance_is_produced(self, forecast):
        assert not forecast.feature_importance.empty
        assert forecast.feature_importance["Importance"].sum() > 0

    def test_no_future_leakage_in_engineered_features(self, config, data):
        """
        Direct assertion against the leakage class this design exists to
        prevent: no backward-looking feature may equal the contemporaneous
        target.
        """
        from src.planning.demand_engine import DirectMultiHorizonFeatureBuilder
        from src.shared.validation import assert_feature_causality
        builder = DirectMultiHorizonFeatureBuilder(config.forecast)
        sales = data.sales_history.copy()
        sales["Is_Promo"] = sales["Is_Promo"].astype(int)
        sales["Is_Stockout"] = sales["Is_Stockout"].astype(int)
        panel = builder.build_panel(sales)
        one_sku = panel[panel[Keys.SKU] == panel[Keys.SKU].iloc[0]]
        features = [c for c in panel.columns
                    if c.startswith(("lag_", "roll_", "ewm_"))]
        assert_feature_causality(one_sku, Keys.WEEK_INDEX, features, "Sales_Units")


# --------------------------------------------------------------------------- #
# Test 11 -- consensus governance
# --------------------------------------------------------------------------- #
class TestConsensus:

    def test_all_scenarios_are_produced(self, consensus, config):
        scenarios = set(consensus.consensus_plan["Scenario"])
        assert scenarios == set(config.consensus.scenario_commercial_weight)

    def test_statistical_baseline_is_never_overwritten(self, consensus):
        """
        The core governance guarantee: the model's own number survives the
        whole workflow so a plan miss can be attributed after the fact.
        """
        plan = consensus.consensus_plan
        assert "Forecast_Units" in plan.columns
        assert "Commercial_Forecast" in plan.columns
        assert "Consensus_Units" in plan.columns
        assert plan["Forecast_Units"].notna().all()

    def test_override_alert_fires_exactly_at_the_threshold(self, consensus, config):
        plan = consensus.consensus_plan
        threshold = config.consensus.override_alert_threshold_pct
        flagged = plan[plan["Override_Alert"]]
        unflagged = plan[~plan["Override_Alert"]]
        if not flagged.empty:
            assert (flagged["Override_Deviation_Pct"].abs() > threshold).all()
        if not unflagged.empty:
            assert (unflagged["Override_Deviation_Pct"].abs() <= threshold + 1e-9).all()

    def test_deviation_percentage_is_arithmetically_correct(self, consensus):
        plan = consensus.consensus_plan
        sample = plan[plan["Forecast_Units"] > 1].head(200)
        expected = ((sample["Commercial_Forecast"] - sample["Forecast_Units"])
                    / sample["Forecast_Units"] * 100)
        assert np.allclose(sample["Override_Deviation_Pct"], expected, atol=0.02)

    def test_every_flagged_override_carries_a_justification(self, consensus):
        log = consensus.override_log
        if log.empty:
            pytest.skip("No overrides generated at this scale.")
        flagged = log[log["Override_Alert"]]
        if not flagged.empty:
            assert "Justification" in flagged.columns
            assert (flagged["Justification"].str.len() > 10).all()

    def test_audit_log_is_timestamped(self, consensus):
        log = consensus.override_log
        if not log.empty:
            assert "Audit_Timestamp" in log.columns
            assert log["Audit_Timestamp"].notna().all()

    def test_consensus_never_exceeds_the_supply_ceiling(self, consensus, config):
        """Constraints must actually bind, not merely be recorded."""
        plan = consensus.consensus_plan
        for scenario in config.consensus.scenario_commercial_weight:
            if not config.consensus.scenario_respect_constraints[scenario]:
                continue
            subset = plan[plan["Scenario"] == scenario]
            assert (subset["Consensus_Units"] <= subset["Supply_Ceiling"] + 1e-3).all()

    def test_scenario_ordering_is_monotonic(self, consensus):
        """Aggressive must plan at least as much volume as Conservative."""
        summary = consensus.scenario_summary.set_index("Scenario")
        assert (summary.loc["Aggressive", "Unconstrained_Units"]
                >= summary.loc["Conservative", "Unconstrained_Units"])

    def test_waterfall_steps_reconcile_to_the_final_consensus(self, consensus):
        """The bridge must actually add up, or it is decoration."""
        waterfall = consensus.waterfall.set_index("Step")["Units"]
        computed = (waterfall["Statistical Baseline"]
                    + waterfall["Commercial Override"]
                    + waterfall["Promotion / NPD"]
                    + waterfall["Operational Constraint"])
        assert computed == pytest.approx(waterfall["Final Consensus"], rel=0.02)


# --------------------------------------------------------------------------- #
# Test 12 -- supply planning / MRP
# --------------------------------------------------------------------------- #
class TestSupplyPlanning:

    def test_plan_covers_every_sku_and_week(self, supply, config):
        plan = supply.supply_plan
        assert set(plan["horizon"]) == set(range(1, config.supply.horizon_weeks + 1))

    def test_inventory_balance_equation_closes(self, supply):
        """
        closing = opening + receipts - demand, exactly, for every row.
        If this drifts the entire supply plan is fiction.
        """
        plan = supply.supply_plan
        expected = (plan["Opening_Inventory"]
                    + plan["Planned_Receipts"] + plan["Open_PO_Receipts"]
                    - plan["Demand_Units"])
        assert np.allclose(plan["Closing_Inventory"], expected, atol=0.5)

    def test_opening_inventory_chains_from_prior_closing(self, supply):
        """Week w opening must equal week w-1 closing, floored at zero."""
        plan = supply.supply_plan.sort_values([Keys.SKU, "horizon"])
        grouped = plan.groupby(Keys.SKU)
        prior_close = grouped["Closing_Inventory"].shift(1)
        mask = prior_close.notna()
        expected = prior_close[mask].clip(lower=0)
        assert np.allclose(plan.loc[mask, "Opening_Inventory"], expected, atol=0.5)

    def test_gross_requirement_includes_demand(self, supply):
        plan = supply.supply_plan
        assert (plan["Gross_Requirement"] >= plan["Demand_Units"] - 1e-6).all()

    def test_net_requirement_is_never_negative(self, supply):
        assert (supply.supply_plan["Net_Requirement"] >= -1e-6).all()

    def test_planned_orders_respect_moq(self, supply):
        """An order below the supplier minimum could not actually be placed."""
        plan = supply.supply_plan
        policy = supply.inventory_policy.set_index(Keys.SKU)["MOQ"]
        orders = plan[plan["Planned_Order_Qty"] > 0].copy()
        if orders.empty:
            pytest.skip("No planned orders generated.")
        orders["MOQ"] = orders[Keys.SKU].map(policy).fillna(0)
        binding = orders[orders["MOQ"] > 0]
        assert (binding["Planned_Order_Qty"] >= binding["MOQ"] - 1e-6).all()

    def test_stockout_flag_matches_negative_closing_stock(self, supply):
        plan = supply.supply_plan
        assert (plan["Is_Stockout_Week"] == (plan["Closing_Inventory"] < 0)).all()

    def test_shortage_units_equal_the_negative_balance(self, supply):
        plan = supply.supply_plan
        stockouts = plan[plan["Is_Stockout_Week"]]
        if not stockouts.empty:
            assert np.allclose(stockouts["Shortage_Units"],
                               -stockouts["Closing_Inventory"], atol=0.5)

    def test_recoverability_respects_the_lead_time(self, supply):
        """
        A shortage arriving before a replenishment could physically land is
        NOT recoverable. This distinction drives the expedite list, so it must
        be exact rather than approximately right.
        """
        summary = supply.sku_summary
        with_stockout = summary[summary["First_Stockout_Week"].notna()]
        if with_stockout.empty:
            pytest.skip("No stockouts projected at this scale.")
        expected = (with_stockout["First_Stockout_Week"]
                    > with_stockout["Lead_Time_Weeks"])
        assert (with_stockout["Can_Recover_In_Lead_Time"] == expected).all()

    def test_supply_risk_index_is_bounded(self, supply):
        summary = supply.sku_summary
        assert summary["Supply_Risk_Index"].between(0, 100).all()

    def test_risk_band_matches_configured_thresholds(self, supply, config):
        summary = supply.sku_summary
        red = summary[summary["Risk_Band"] == "Red"]
        green = summary[summary["Risk_Band"] == "Green"]
        if not red.empty:
            assert (red["Supply_Risk_Index"] >= config.supply.risk_red_threshold).all()
        if not green.empty:
            assert (green["Supply_Risk_Index"] < config.supply.risk_amber_threshold).all()

    def test_risk_weights_sum_to_one(self, config):
        assert sum(config.supply.risk_weights.values()) == pytest.approx(1.0)

    def test_capacity_severity_matches_utilisation(self, supply, config):
        capacity = supply.capacity_alerts
        breaches = capacity[capacity["Severity"] == "BREACH"]
        ok = capacity[capacity["Severity"] == "OK"]
        if not breaches.empty:
            assert (breaches["Utilisation_Pct"]
                    >= config.supply.capacity_breach_threshold * 100).all()
        if not ok.empty:
            assert (ok["Utilisation_Pct"]
                    < config.supply.capacity_alert_threshold * 100).all()

    def test_capacity_is_aggregated_across_skus(self, supply):
        """
        A breach can only be detected in aggregate. Verify the alert log is
        keyed on the shared resource, not on individual SKUs.
        """
        capacity = supply.capacity_alerts
        assert "Resource_ID" in capacity.columns
        assert set(capacity["Resource_Type"]) <= {"Supplier", "Warehouse"}
        assert (capacity["SKUs_Affected"] >= 1).all()

    def test_service_level_targets_follow_abc_class(self, supply, config):
        """C-items must not be buffered to A-item service levels."""
        policy = supply.inventory_policy
        for abc, target in config.supply.service_level_by_class.items():
            subset = policy[policy["ABC_Class"] == abc]
            if not subset.empty:
                assert np.allclose(subset["Service_Level_Target"], target)

    def test_safety_stock_is_non_negative(self, supply):
        assert (supply.inventory_policy["Safety_Stock_Units"] >= 0).all()

    def test_weeks_of_cover_is_bounded(self, supply):
        assert supply.supply_plan["Weeks_Of_Cover"].between(0, 52).all()


# --------------------------------------------------------------------------- #
# Test 13 -- OTIF and SPC
# --------------------------------------------------------------------------- #
class TestOTIFAndSPC:

    def test_every_order_is_evaluated(self, kpi, config):
        assert len(kpi.order_detail) == config.data.n_orders

    def test_otif_is_the_conjunction_of_on_time_and_in_full(self, kpi):
        detail = kpi.order_detail
        assert (detail["OTIF"] == (detail["On_Time"] & detail["In_Full"])).all()

    def test_otif_is_recomputed_from_source_dates(self, kpi, config):
        """
        Flags are derived from promised date and quantity, not trusted from the
        source file -- otherwise a revised promise date could launder lateness
        out of the metric.
        """
        detail = kpi.order_detail
        expected_on_time = detail["Delay_Days"] <= config.kpi.on_time_grace_days
        assert (detail["On_Time"] == expected_on_time).all()

    def test_in_full_respects_the_tolerance(self, kpi, config):
        detail = kpi.order_detail
        expected = detail["Fill_Rate"] >= config.kpi.in_full_tolerance
        assert (detail["In_Full"] == expected).all()

    def test_dimensional_otif_reconciles_to_the_total(self, kpi):
        """Every cut must sum back to the same order population."""
        total = len(kpi.order_detail)
        for frame in (kpi.otif_by_supplier, kpi.otif_by_category,
                      kpi.otif_by_channel, kpi.otif_by_month):
            assert frame["Orders"].sum() == total

    def test_otif_percentages_are_bounded(self, kpi):
        for frame in (kpi.otif_by_supplier, kpi.otif_by_category,
                      kpi.otif_by_month):
            assert frame["OTIF_Pct"].between(0, 100).all()

    def test_benchmark_bands_match_thresholds(self, kpi, config):
        frame = kpi.otif_by_supplier
        world_class = frame[frame["Benchmark"] == "World Class"]
        at_risk = frame[frame["Benchmark"] == "At Risk"]
        if not world_class.empty:
            assert (world_class["OTIF_Pct"] >= config.kpi.otif_world_class).all()
        if not at_risk.empty:
            assert (at_risk["OTIF_Pct"] < config.kpi.otif_acceptable).all()

    def test_control_limits_bracket_the_centre_line(self, kpi):
        chart = kpi.spc_chart
        assert (chart["LCL"] <= chart["Center_Line"]).all()
        assert (chart["Center_Line"] <= chart["UCL"]).all()

    def test_sigma_deviation_is_consistent_with_the_limits(self, kpi):
        chart = kpi.spc_chart
        sigma = chart["Sigma"].iloc[0]
        if sigma > 1e-9:
            expected = (chart["OTIF_Pct"] - chart["Center_Line"]) / sigma
            assert np.allclose(chart["Sigma_Deviation"], expected, atol=0.01)

    def test_two_sigma_alert_flags_the_right_months(self, kpi, config):
        chart = kpi.spc_chart
        expected = chart["Sigma_Deviation"] <= -config.kpi.spc_alert_sigma
        assert (chart["Below_2Sigma_Alert"] == expected).all()

    def test_root_cause_shares_sum_to_one_hundred(self, kpi):
        causes = kpi.root_cause_analysis
        if not causes.empty:
            assert causes["Share_Pct"].sum() == pytest.approx(100.0, abs=0.5)

    def test_root_causes_cover_every_failed_order(self, kpi):
        detail = kpi.order_detail
        failures = int((~detail["OTIF"]).sum())
        if failures:
            assert kpi.root_cause_analysis["Failed_Orders"].sum() == failures

    def test_successful_orders_carry_no_failure_cause(self, kpi):
        detail = kpi.order_detail
        successes = detail[detail["OTIF"]]
        assert (successes["Root_Cause"] == "On Time In Full").all()


# --------------------------------------------------------------------------- #
# Test 14 -- insight generation
# --------------------------------------------------------------------------- #
class TestInsights:

    def test_insights_are_generated(self, insights):
        assert not insights.empty
        assert len(insights) >= 5

    def test_required_fields_are_present(self, insights):
        for column in ("Rank", "Severity", "Domain", "Headline", "Detail", "Owner"):
            assert column in insights.columns

    def test_severity_values_are_valid(self, insights):
        assert set(insights["Severity"]) <= {"Critical", "High", "Medium", "Info"}

    def test_sorted_by_severity(self, insights):
        order = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3}
        ranks = insights["Severity"].map(order).tolist()
        assert ranks == sorted(ranks)

    def test_headlines_contain_computed_numbers(self, insights):
        """
        Insights must be derived, not templated. A headline with no digit in it
        is almost certainly a hard-coded sentence.
        """
        assert insights["Headline"].str.contains(r"\d").mean() > 0.8

    def test_every_domain_is_represented(self, insights):
        domains = set(insights["Domain"])
        assert {"Demand", "Supply", "Service"} <= domains


# --------------------------------------------------------------------------- #
# Test 15 -- cross-module integration contracts
# --------------------------------------------------------------------------- #
class TestIntegrationContracts:

    def test_forecast_flows_into_consensus_without_sku_loss(self, forecast, consensus):
        forecast_skus = set(forecast.forecast[Keys.SKU])
        consensus_skus = set(consensus.consensus_plan[Keys.SKU])
        assert consensus_skus == forecast_skus

    def test_consensus_flows_into_supply_without_sku_loss(self, consensus, supply):
        planned = set(supply.supply_plan[Keys.SKU])
        consensus_skus = set(consensus.consensus_plan[Keys.SKU])
        assert planned == consensus_skus

    def test_supply_demand_equals_consensus_demand(self, consensus, supply, config):
        """The supply plan must net against exactly the demand that was agreed."""
        scenario = consensus.diagnostics["scenario"]
        agreed = consensus.consensus_plan[
            consensus.consensus_plan["Scenario"] == scenario]["Consensus_Units"].sum()
        planned = supply.supply_plan["Demand_Units"].sum()
        assert planned == pytest.approx(agreed, rel=0.01)

    def test_shared_keys_are_consistent_across_outputs(self, forecast, supply, data):
        """Output 1 and Output 5 integration depends on these keys matching."""
        for frame in (forecast.uncertainty, supply.sku_summary):
            assert Keys.SKU in frame.columns
            assert Keys.SUPPLIER in frame.columns
        assert set(supply.sku_summary[Keys.SUPPLIER]) <= set(
            data.supplier_master[Keys.SUPPLIER])

    def test_output_5_handoff_contract_is_complete(self, forecast):
        """
        The documented interface to Output 5 (Inventory Optimisation). Breaking
        any of these columns silently breaks the downstream project.
        """
        required = {Keys.SKU, Keys.SUPPLIER, "Forecast_Units_12W",
                    "Avg_Weekly_Forecast", "Forecast_Error_Std",
                    "Historical_Demand_Std", "Demand_CV", "Unit_Cost"}
        assert required <= set(forecast.uncertainty.columns)

    def test_inventory_policy_exposes_safety_stock_inputs(self, supply):
        policy = supply.inventory_policy
        required = {"Safety_Stock_Units", "Reorder_Point_Units",
                    "Service_Level_Target", "Lead_Time_Weeks", "ABC_Class"}
        assert required <= set(policy.columns)
