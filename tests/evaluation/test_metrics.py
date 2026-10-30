"""Tests for the accuracy metrics.

Every expected value here is computed by hand in the comment above the
assertion. A metric test that recomputes the metric with the same expression
as the implementation proves nothing, so the numbers below are literal.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.evaluation.metrics import (
    MetricSet,
    directional_accuracy,
    evaluate,
    interval_coverage,
    mae,
    mase,
    mase_scale,
    rmse,
    smape,
)

# -- MAE -------------------------------------------------------------------


def test_mae_is_the_mean_absolute_error() -> None:
    # |0| + |2| + |1| = 3, over 3 points -> 1.0
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 2.0]) == pytest.approx(1.0)


def test_mae_of_a_perfect_forecast_is_zero() -> None:
    assert mae([5.0, 6.0], [5.0, 6.0]) == 0.0


def test_mae_ignores_the_sign_of_the_error() -> None:
    # Both forecasts are off by 2, in opposite directions.
    assert mae([10.0], [12.0]) == pytest.approx(mae([10.0], [8.0]))


# -- RMSE ------------------------------------------------------------------


def test_rmse_is_the_root_mean_squared_error() -> None:
    # Errors 0, 2, 1 -> mean square 5/3 -> sqrt(5/3) = 1.290994...
    assert rmse([1.0, 2.0, 3.0], [1.0, 4.0, 2.0]) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_rmse_punishes_a_single_large_error_more_than_mae() -> None:
    # MAE = 10/3, RMSE = sqrt(100/3) = 5.7735...
    actual = [0.0, 0.0, 0.0]
    predicted = [10.0, 0.0, 0.0]
    assert mae(actual, predicted) == pytest.approx(10.0 / 3.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(100.0 / 3.0))


# -- sMAPE -----------------------------------------------------------------


def test_smape_matches_the_hand_computed_percentage() -> None:
    # Terms: 2*2/(10+12) = 2/11 and 2*4/(20+16) = 2/9.
    # Mean = (2/11 + 2/9) / 2 = (18 + 22) / 198 = 20/99 -> 20.2020...%
    assert smape([10.0, 20.0], [12.0, 16.0]) == pytest.approx(2000.0 / 99.0)


def test_smape_reaches_two_hundred_for_an_inverted_forecast() -> None:
    # 2*|10 - (-10)| / (10 + 10) = 2, the ceiling of this convention.
    assert smape([10.0, -10.0], [-10.0, 10.0]) == pytest.approx(200.0)


def test_smape_of_a_perfect_forecast_is_zero() -> None:
    assert smape([3.0, 4.0], [3.0, 4.0]) == 0.0


def test_smape_defines_the_all_zero_term_as_zero() -> None:
    # First pair is 0/0 -> 0 by convention; second is 2*10/30 = 2/3.
    # Mean of (0, 2/3) is 1/3 -> 33.333...%
    assert smape([0.0, 10.0], [0.0, 20.0]) == pytest.approx(100.0 / 3.0)


# -- MASE scale ------------------------------------------------------------


def test_mase_scale_is_the_mean_naive_error() -> None:
    # One-step differences: |2-1|, |4-2|, |8-4| = 1, 2, 4 -> mean 7/3.
    assert mase_scale([1.0, 2.0, 4.0, 8.0]) == pytest.approx(7.0 / 3.0)


def test_mase_scale_uses_the_seasonal_lag() -> None:
    # Lag 2: |4-1| and |8-2| -> mean 9/2.
    assert mase_scale([1.0, 2.0, 4.0, 8.0], seasonal_period=2) == pytest.approx(4.5)


def test_mase_scale_rejects_a_window_no_longer_than_the_season() -> None:
    with pytest.raises(DataValidationError, match="more than seasonal_period"):
        mase_scale([1.0, 2.0], seasonal_period=2)


def test_mase_scale_rejects_a_non_positive_season() -> None:
    with pytest.raises(DataValidationError, match="seasonal_period must be positive"):
        mase_scale([1.0, 2.0], seasonal_period=0)


def test_mase_scale_rejects_a_constant_series() -> None:
    with pytest.raises(DataValidationError, match="constant"):
        mase_scale([5.0, 5.0, 5.0])


# -- MASE ------------------------------------------------------------------


def test_mase_is_the_mae_over_the_naive_scale() -> None:
    # MAE = 1, scale = 7/3 -> 3/7.
    value = mase([10.0, 10.0], [11.0, 9.0], [1.0, 2.0, 4.0, 8.0])
    assert value == pytest.approx(3.0 / 7.0)


def test_a_naive_forecast_scores_exactly_one() -> None:
    # The training window 1, 3, 5, 7 has a one-step naive error of 2 at every
    # point, so the scale is 2. A naive forecast of 7 against an outcome of 9
    # is off by exactly 2 -> MASE 2 / 2 = 1, "as good as doing nothing".
    assert mase([9.0], [7.0], [1.0, 3.0, 5.0, 7.0]) == pytest.approx(1.0)


def test_mase_uses_the_seasonal_scale_when_asked() -> None:
    # MAE = 2, lag-2 scale = 4.5 -> 4/9.
    value = mase([10.0], [12.0], [1.0, 2.0, 4.0, 8.0], seasonal_period=2)
    assert value == pytest.approx(4.0 / 9.0)


# -- directional accuracy --------------------------------------------------


def test_directional_accuracy_counts_correctly_signed_changes() -> None:
    # Outcomes: +1, -1, 0. Predictions: +1, +1, 0. Two of three match.
    value = directional_accuracy([2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0])
    assert value == pytest.approx(200.0 / 3.0)


def test_a_flat_prediction_only_scores_on_a_flat_outcome() -> None:
    # Predicting "no change" every time: only the unchanged outcome counts.
    value = directional_accuracy([2.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert value == pytest.approx(100.0 / 3.0)


def test_directional_accuracy_is_one_hundred_for_a_perfect_forecast() -> None:
    assert directional_accuracy([2.0, 0.0], [2.0, 0.0], [1.0, 1.0]) == 100.0


def test_directional_accuracy_rejects_a_mismatched_reference() -> None:
    with pytest.raises(DataValidationError, match="previous must have the same length"):
        directional_accuracy([1.0, 2.0], [1.0, 2.0], [1.0])


# -- interval coverage -----------------------------------------------------


def test_interval_coverage_counts_outcomes_inside_the_bounds() -> None:
    # 1 is inside [0, 2]; 2 is outside [3, 4]; 3 is inside [2, 4].
    assert interval_coverage([1.0, 2.0, 3.0], [0.0, 3.0, 2.0], [2.0, 4.0, 4.0]) == pytest.approx(
        200.0 / 3.0
    )


def test_interval_coverage_includes_the_bounds() -> None:
    assert interval_coverage([0.0, 5.0], [0.0, 5.0], [0.0, 5.0]) == 100.0


def test_interval_coverage_rejects_inverted_bounds() -> None:
    with pytest.raises(DataValidationError, match="lower bound exceeds upper bound"):
        interval_coverage([1.0], [2.0], [0.0])


def test_interval_coverage_rejects_a_mismatched_upper_bound() -> None:
    with pytest.raises(DataValidationError, match="upper must have the same length"):
        interval_coverage([1.0, 2.0], [0.0, 0.0], [3.0])


# -- shared validation -----------------------------------------------------


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(DataValidationError, match="y_true is empty"):
        mae([], [])


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(DataValidationError, match="must have the same length"):
        rmse([1.0, 2.0], [1.0])


# -- MetricSet -------------------------------------------------------------


def test_evaluate_agrees_with_the_individual_functions() -> None:
    y_true = [10.0, 12.0, 11.0]
    y_pred = [11.0, 11.0, 13.0]
    previous = [9.0, 10.0, 12.0]
    y_train = [7.0, 8.0, 9.0, 10.0]

    result = evaluate(y_true, y_pred, previous=previous, y_train=y_train)

    assert isinstance(result, MetricSet)
    assert result.mae == pytest.approx(mae(y_true, y_pred))
    assert result.rmse == pytest.approx(rmse(y_true, y_pred))
    assert result.smape == pytest.approx(smape(y_true, y_pred))
    assert result.mase == pytest.approx(mase(y_true, y_pred, y_train))
    assert result.directional_accuracy == pytest.approx(
        directional_accuracy(y_true, y_pred, previous)
    )
    assert result.n_observations == 3


def test_metric_set_as_dict_drops_the_sample_count() -> None:
    result = evaluate([1.0], [1.0], previous=[1.0], y_train=[1.0, 2.0])
    as_dict = result.as_dict()
    assert "n_observations" not in as_dict
    assert as_dict["mae"] == 0.0


def test_metric_set_is_frozen() -> None:
    result = evaluate([1.0], [1.0], previous=[1.0], y_train=[1.0, 2.0])
    with pytest.raises(AttributeError):
        result.mae = 5.0  # type: ignore[misc]
