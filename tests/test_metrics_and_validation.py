"""
Tests for the metric library and the temporal validation framework.

These are the highest-value tests in the suite. Every downstream KPI, every
Excel cell and every dashboard figure resolves to a function in
``src.shared.metrics``, so an error here propagates silently into results that
still *look* plausible. Each metric is therefore checked against a
hand-computed expected value rather than against its own implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.shared.metrics import (abc_classification, accuracy_bundle, bias,
                                bias_pct, capacity_utilisation,
                                coefficient_of_variation, control_limits,
                                economic_order_quantity, forecast_value_added,
                                inventory_turns, inverse_error_weights, mae,
                                mape, otif_rate, reorder_point, rmse,
                                safety_stock, service_level_z, smape,
                                weeks_of_cover, wape, working_capital,
                                xyz_classification)
from src.shared.validation import (DataValidator, assert_no_leakage,
                                   holdout_split, rolling_origin_splits,
                                   validate_datasets)


# --------------------------------------------------------------------------- #
# Test 1 -- forecast error metrics
# --------------------------------------------------------------------------- #
class TestForecastMetrics:

    def test_mape_matches_hand_calculation(self):
        actual = [100, 200, 400]
        forecast = [110, 180, 440]
        # errors: 10/100=0.10, 20/200=0.10, 40/400=0.10 -> 10%
        assert mape(actual, forecast) == pytest.approx(10.0)

    def test_mape_excludes_zero_actuals(self):
        """Zero-demand periods are undefined for MAPE and must be dropped."""
        assert mape([0, 100], [50, 110]) == pytest.approx(10.0)

    def test_perfect_forecast_scores_zero_error(self):
        actual = [12, 45, 78, 3]
        assert mape(actual, actual) == pytest.approx(0.0)
        assert wape(actual, actual) == pytest.approx(0.0)
        assert mae(actual, actual) == pytest.approx(0.0)
        assert rmse(actual, actual) == pytest.approx(0.0)
        assert bias(actual, actual) == pytest.approx(0.0)

    def test_wape_is_volume_weighted_and_defined_at_zero(self):
        actual = [0, 100]
        forecast = [10, 110]
        # sum|error| = 10 + 10 = 20; sum|actual| = 100 -> 20%
        assert wape(actual, forecast) == pytest.approx(20.0)

    def test_wape_differs_from_mape_when_volumes_are_uneven(self):
        """
        The distinction that matters in practice: a large relative error on a
        tiny SKU barely moves WAPE but dominates MAPE.
        """
        actual = [10, 1000]
        forecast = [20, 1000]      # 100% error on the small item, 0% on the large
        assert mape(actual, forecast) == pytest.approx(50.0)
        assert wape(actual, forecast) == pytest.approx(10 / 1010 * 100)

    def test_mae_and_rmse(self):
        actual = [10, 20, 30]
        forecast = [12, 18, 33]      # errors 2, -2, 3
        assert mae(actual, forecast) == pytest.approx(7 / 3)
        assert rmse(actual, forecast) == pytest.approx(np.sqrt(17 / 3))

    def test_rmse_penalises_large_errors_more_than_mae(self):
        even = rmse([100, 100], [90, 110])       # two errors of 10
        spiky = rmse([100, 100], [100, 120])     # one error of 20
        assert spiky > even
        assert mae([100, 100], [90, 110]) == mae([100, 100], [100, 120])

    def test_bias_sign_indicates_direction(self):
        assert bias([100, 100], [110, 110]) == pytest.approx(10.0)   # over-forecast
        assert bias([100, 100], [90, 90]) == pytest.approx(-10.0)    # under-forecast
        assert bias_pct([100, 100], [110, 110]) == pytest.approx(10.0)

    def test_smape_is_bounded_and_defined_at_zero_actual(self):
        assert smape([0], [50]) == pytest.approx(200.0)
        assert 0 <= smape([10, 20], [12, 18]) <= 200

    def test_accuracy_bundle_returns_full_panel(self):
        bundle = accuracy_bundle([10, 20, 30], [11, 19, 32], naive=[10, 10, 10])
        for key in ("MAPE", "sMAPE", "WAPE", "MAE", "RMSE", "Bias",
                    "Bias_Pct", "Tracking_Signal", "FVA_vs_Naive", "N_Obs"):
            assert key in bundle
        assert bundle["N_Obs"] == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            mape([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------- #
# Test 2 -- Forecast Value Added
# --------------------------------------------------------------------------- #
class TestForecastValueAdded:

    def test_positive_fva_when_model_beats_naive(self):
        actual = [100, 100, 100]
        model = [102, 98, 101]
        naive = [130, 70, 120]
        assert forecast_value_added(actual, model, naive) > 0

    def test_negative_fva_is_reported_not_hidden(self):
        """
        A pipeline that underperforms the naive benchmark must say so. Clamping
        FVA at zero would conceal exactly the finding the metric exists for.
        """
        actual = [100, 100, 100]
        model = [150, 50, 140]
        naive = [101, 99, 100]
        assert forecast_value_added(actual, model, naive) < 0


# --------------------------------------------------------------------------- #
# Test 3 -- ensemble weighting
# --------------------------------------------------------------------------- #
class TestEnsembleWeights:

    def test_weights_sum_to_one(self):
        weights = inverse_error_weights({"A": 10.0, "B": 20.0, "C": 40.0})
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_lower_error_receives_higher_weight(self):
        weights = inverse_error_weights({"good": 10.0, "bad": 40.0})
        assert weights["good"] > weights["bad"]

    def test_inverse_mape_formula_is_exact(self):
        """w_i = (1/e_i) / sum_j(1/e_j) -- verified against the closed form."""
        errors = {"A": 10.0, "B": 30.0}
        weights = inverse_error_weights(errors, floor=1.0, min_weight=0.0)
        total = 1 / 10 + 1 / 30
        assert weights["A"] == pytest.approx((1 / 10) / total)
        assert weights["B"] == pytest.approx((1 / 30) / total)

    def test_floor_prevents_single_model_domination(self):
        """A near-zero error must not collapse the ensemble onto one model."""
        weights = inverse_error_weights({"A": 0.0001, "B": 20.0}, floor=1.0)
        assert weights["A"] < 0.99
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_min_weight_drops_and_renormalises(self):
        weights = inverse_error_weights({"A": 5.0, "B": 500.0}, min_weight=0.05)
        assert "B" not in weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_non_finite_errors_are_excluded(self):
        weights = inverse_error_weights({"A": 10.0, "B": float("nan"),
                                         "C": float("inf")})
        assert set(weights) == {"A"}


# --------------------------------------------------------------------------- #
# Test 4 -- inventory formulas
# --------------------------------------------------------------------------- #
class TestInventoryMetrics:

    def test_safety_stock_matches_closed_form(self):
        z = service_level_z(0.95)
        expected = z * np.sqrt(4 * 20 ** 2)          # LT=4, sigma_D=20, sigma_LT=0
        assert safety_stock(20, 4, 0.95) == pytest.approx(expected)

    def test_lead_time_variability_increases_safety_stock(self):
        """
        The term most spreadsheets omit. Ignoring lead-time variance
        under-buffers exactly the unreliable suppliers that need it most.
        """
        without = safety_stock(20, 4, 0.95, avg_demand=100, lead_time_std_weeks=0.0)
        with_lt = safety_stock(20, 4, 0.95, avg_demand=100, lead_time_std_weeks=1.5)
        assert with_lt > without

    def test_higher_service_level_requires_more_stock(self):
        assert safety_stock(20, 4, 0.99) > safety_stock(20, 4, 0.90)

    def test_zero_variability_yields_zero_safety_stock(self):
        assert safety_stock(0, 4, 0.95, avg_demand=100,
                            lead_time_std_weeks=0) == pytest.approx(0.0)

    def test_safety_stock_never_negative(self):
        assert safety_stock(-5, -1, 0.95) >= 0

    def test_service_level_z_values(self):
        assert service_level_z(0.95) == pytest.approx(1.645, abs=0.01)
        assert service_level_z(0.99) == pytest.approx(2.326, abs=0.01)
        assert service_level_z(0.50) == pytest.approx(0.0, abs=0.01)

    def test_reorder_point_formula(self):
        assert reorder_point(100, 3, 250) == pytest.approx(550.0)

    def test_weeks_of_cover(self):
        assert weeks_of_cover(400, 100) == pytest.approx(4.0)

    def test_weeks_of_cover_capped_when_demand_is_zero(self):
        """Near-zero demand must not produce an infinite cover that breaks averages."""
        assert weeks_of_cover(500, 0) == 52.0
        assert weeks_of_cover(0, 0) == 0.0

    def test_eoq_formula(self):
        # Q* = sqrt(2DS/H), H = 0.25 * 40 = 10
        expected = np.sqrt(2 * 10_000 * 500 / 10)
        assert economic_order_quantity(10_000, 500, 40, 0.25) == pytest.approx(expected)

    def test_working_capital_and_turns(self):
        assert working_capital(1000, 25) == pytest.approx(25_000)
        assert inventory_turns(1_200_000, 100_000) == pytest.approx(12.0)

    def test_capacity_utilisation(self):
        assert capacity_utilisation(1200, 1000) == pytest.approx(120.0)
        assert capacity_utilisation(0, 0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Test 5 -- classification
# --------------------------------------------------------------------------- #
class TestClassification:

    def test_abc_puts_high_value_items_in_class_a(self):
        values = pd.Series({"hero": 700, "mid": 200, "tail1": 50, "tail2": 50})
        classes = abc_classification(values)
        assert classes["hero"] == "A"
        assert classes["tail2"] == "C"

    def test_abc_covers_every_item(self):
        values = pd.Series(np.random.default_rng(1).lognormal(5, 1, 50))
        classes = abc_classification(values)
        assert len(classes) == 50
        assert classes.notna().all()

    def test_xyz_thresholds(self):
        cv = pd.Series({"steady": 0.2, "variable": 0.8, "erratic": 1.6})
        classes = xyz_classification(cv)
        assert classes["steady"] == "X"
        assert classes["variable"] == "Y"
        assert classes["erratic"] == "Z"

    def test_coefficient_of_variation(self):
        assert coefficient_of_variation([10, 10, 10]) == pytest.approx(0.0)
        assert coefficient_of_variation([1, 2, 3, 4, 5]) > 0


# --------------------------------------------------------------------------- #
# Test 6 -- OTIF and control limits
# --------------------------------------------------------------------------- #
class TestServiceMetrics:

    def test_otif_requires_both_conditions(self):
        """On-time-but-short and in-full-but-late are both service failures."""
        on_time = [True, True, False, False]
        in_full = [True, False, True, False]
        assert otif_rate(on_time, in_full) == pytest.approx(25.0)

    def test_otif_bounds(self):
        assert otif_rate([True] * 5, [True] * 5) == pytest.approx(100.0)
        assert otif_rate([False] * 5, [False] * 5) == pytest.approx(0.0)

    def test_control_limits_are_symmetric_about_the_mean(self):
        limits = control_limits([90, 91, 89, 92, 88, 90, 91, 89])
        assert limits["UCL"] - limits["mean"] == pytest.approx(
            limits["mean"] - limits["LCL"])

    def test_sigma_uses_moving_range_not_sample_std(self):
        """
        The moving-range estimator must not be inflated by a special cause.
        With one large outlier, MRbar/d2 should stay below the raw sample
        standard deviation -- that is the whole reason it is used.
        """
        series = [90, 90, 90, 90, 90, 90, 60, 90, 90, 90]
        limits = control_limits(series)
        assert limits["sigma"] < float(np.std(series, ddof=1))

    def test_constant_series_yields_zero_sigma(self):
        limits = control_limits([90.0] * 10)
        assert limits["sigma"] == pytest.approx(0.0)
        assert limits["UCL"] == pytest.approx(90.0)

    def test_empty_series_is_handled(self):
        limits = control_limits([])
        assert limits["n"] == 0
        assert np.isnan(limits["mean"])


# --------------------------------------------------------------------------- #
# Test 7 -- temporal validation (leakage prevention)
# --------------------------------------------------------------------------- #
class TestTemporalValidation:

    def test_folds_never_overlap_their_own_test_window(self):
        splits = rolling_origin_splits(n_obs=104, horizon=12, n_folds=2,
                                       min_train=60, step=12, holdout=12)
        assert splits
        assert_no_leakage(splits)
        for split in splits:
            assert split.train_end <= split.test_start

    def test_holdout_block_is_never_used_as_a_fold(self):
        """
        The single most important guarantee in the system: model selection must
        not be able to see the window that accuracy is reported from.
        """
        n_obs, horizon = 104, 12
        splits = rolling_origin_splits(n_obs, horizon, n_folds=2,
                                       min_train=60, step=12, holdout=horizon)
        holdout_start = n_obs - horizon
        for split in splits:
            assert split.test_end <= holdout_start

    def test_training_windows_expand_over_folds(self):
        splits = rolling_origin_splits(104, 12, n_folds=2, min_train=60,
                                       step=12, holdout=12)
        train_sizes = [s.train_end for s in splits]
        assert train_sizes == sorted(train_sizes)

    def test_short_series_returns_no_folds(self):
        assert rolling_origin_splits(30, 12, n_folds=2, min_train=60,
                                     holdout=12) == []

    def test_holdout_split_is_the_final_block(self):
        hold = holdout_split(104, 12)
        assert hold.test_start == 92
        assert hold.test_end == 104
        assert hold.horizon == 12

    def test_leakage_assertion_actually_fires(self):
        from src.shared.validation import TimeSplit
        bad = [TimeSplit(fold=1, train_end=50, test_start=40, test_end=52,
                         label="BAD")]
        with pytest.raises(AssertionError):
            assert_no_leakage(bad)


# --------------------------------------------------------------------------- #
# Test 8 -- schema validation
# --------------------------------------------------------------------------- #
class TestDataValidator:

    def test_missing_columns_are_errors(self):
        v = DataValidator()
        v.require_columns(pd.DataFrame({"A": [1]}), "test", ["A", "B"])
        assert not v.is_valid()

    def test_duplicate_keys_are_errors(self):
        v = DataValidator()
        v.require_unique(pd.DataFrame({"id": [1, 1, 2]}), "test", "id")
        assert not v.is_valid()

    def test_negative_values_are_errors(self):
        v = DataValidator()
        v.require_non_negative(pd.DataFrame({"qty": [5, -3]}), "test", ["qty"])
        assert not v.is_valid()

    def test_orphan_foreign_keys_are_detected(self):
        v = DataValidator()
        child = pd.DataFrame({"Supplier_ID": ["S1", "S9"]})
        v.require_referential_integrity(child, "child", "Supplier_ID", ["S1", "S2"])
        assert not v.is_valid()

    def test_clean_frame_passes(self):
        v = DataValidator()
        frame = pd.DataFrame({"id": [1, 2], "qty": [5, 10]})
        v.require_columns(frame, "test", ["id", "qty"])
        v.require_unique(frame, "test", "id")
        v.require_non_negative(frame, "test", ["qty"])
        assert v.is_valid()

    def test_raise_for_errors_raises(self):
        v = DataValidator()
        v.require_columns(pd.DataFrame(), "test", ["missing"])
        with pytest.raises(ValueError):
            v.raise_for_errors()
