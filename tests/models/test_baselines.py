"""Value tests for the baseline models.

Each baseline is checked against a hand-computed expectation on a short,
literal series, so the tests prove the arithmetic rather than restating the
implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.models import Drift, Forecaster, MovingAverage, Naive, SeasonalNaive


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series([float(value) for value in values], index=index)


# -- Naive -----------------------------------------------------------------


def test_naive_repeats_the_last_value() -> None:
    values = Naive(horizon=3).fit(_series([1.0, 2.0, 5.0])).predict()
    np.testing.assert_array_equal(values, [5.0, 5.0, 5.0])


def test_naive_ignores_earlier_values() -> None:
    flat = Naive().fit(_series([0.0, 0.0, 7.0])).predict()
    noisy = Naive().fit(_series([100.0, -4.0, 7.0])).predict()
    np.testing.assert_array_equal(flat, noisy)


# -- SeasonalNaive ---------------------------------------------------------


def test_seasonal_naive_repeats_one_season() -> None:
    # Season 2, so the forecast alternates 10, 20, 10, 20.
    values = SeasonalNaive(season=2, horizon=4).fit(_series([10.0, 20.0])).predict()
    np.testing.assert_array_equal(values, [10.0, 20.0, 10.0, 20.0])


def test_seasonal_naive_uses_its_own_forecasts_beyond_one_season() -> None:
    # History [1, 2, 3]; season 3 -> step 1 copies 1, then steps 2..4 copy the
    # forecasts just produced rather than any observation.
    values = SeasonalNaive(season=3, horizon=4).fit(_series([1.0, 2.0, 3.0])).predict()
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 1.0])


def test_seasonal_naive_rejects_a_short_history() -> None:
    with pytest.raises(DataValidationError, match="needs at least 7 observations"):
        SeasonalNaive(season=7).fit(_series([1.0, 2.0, 3.0]))


def test_seasonal_naive_rejects_a_non_positive_season() -> None:
    with pytest.raises(DataValidationError, match="season must be positive"):
        SeasonalNaive(season=0)


# -- MovingAverage ---------------------------------------------------------


def test_moving_average_is_the_trailing_mean() -> None:
    values = MovingAverage(window=3, horizon=2).fit(_series([1.0, 2.0, 3.0, 6.0])).predict()
    # mean(2, 3, 6) = 11/3
    np.testing.assert_allclose(values, [11.0 / 3.0, 11.0 / 3.0])


def test_moving_average_ignores_rows_outside_the_window() -> None:
    values = MovingAverage(window=2).fit(_series([1000.0, 4.0, 6.0])).predict()
    assert values[0] == pytest.approx(5.0)


def test_moving_average_rejects_a_short_history() -> None:
    with pytest.raises(DataValidationError, match="needs at least 5 observations"):
        MovingAverage(window=5).fit(_series([1.0, 2.0]))


def test_moving_average_rejects_a_non_positive_window() -> None:
    with pytest.raises(DataValidationError, match="window must be positive"):
        MovingAverage(window=0)


# -- Drift -----------------------------------------------------------------


def test_drift_extrapolates_the_average_change() -> None:
    # 1 -> 3 -> 5 is a slope of 2 per period.
    values = Drift(horizon=3).fit(_series([1.0, 3.0, 5.0])).predict()
    np.testing.assert_allclose(values, [7.0, 9.0, 11.0])


def test_drift_window_restricts_the_slope_estimate() -> None:
    # Whole history slopes at (101 - 1) / 3, but the last two points slope at 1.
    values = Drift(window=2, horizon=2).fit(_series([1.0, 1.0, 100.0, 101.0])).predict()
    np.testing.assert_allclose(values, [102.0, 103.0])


def test_drift_rejects_a_single_observation() -> None:
    with pytest.raises(DataValidationError, match="at least 2 observations"):
        Drift().fit(_series([5.0]))


def test_drift_rejects_a_window_below_two() -> None:
    with pytest.raises(DataValidationError, match="window must be at least 2"):
        Drift(window=1)


def test_drift_rejects_a_window_longer_than_the_history() -> None:
    with pytest.raises(DataValidationError, match="needs at least 10 observations"):
        Drift(window=10).fit(_series([1.0, 2.0, 3.0]))


# -- shared behaviour ------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [Naive(horizon=2), MovingAverage(window=2, horizon=2)],
    ids=["naive", "moving-average"],
)
def test_flat_models_repeat_one_value(model: Forecaster) -> None:
    values = model.fit(_series([1.0, 3.0])).predict()
    assert values.shape == (2,)
    assert values[0] == pytest.approx(values[1])


def test_seasonal_naive_is_not_flat_across_a_season() -> None:
    # Two observations, season 2: the two steps copy different observations.
    values = SeasonalNaive(season=2, horizon=2).fit(_series([1.0, 3.0])).predict()
    np.testing.assert_array_equal(values, [1.0, 3.0])
