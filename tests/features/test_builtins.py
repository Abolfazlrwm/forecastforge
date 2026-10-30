"""Value and contract tests for the built-in transforms.

The value tests use :func:`counting_series` (target ``1.0 .. 12.0``) so the
expected numbers can be written down by hand instead of being copied from the
implementation, which would only prove the code agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.features import (
    MACD,
    RSI,
    CalendarFeatures,
    FeatureSet,
    Lag,
    Returns,
    RollingStat,
    Volatility,
)
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata


def _series(values: list[float]) -> TimeSeries:
    """Build a daily single-column series from literal values."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    frame = pd.DataFrame({"close": [float(value) for value in values]}, index=index)
    return TimeSeries(
        frame=frame,
        target="close",
        metadata=TimeSeriesMetadata(symbol="TEST", source="fixture", frequency="D"),
    )


def _column(transform: object, series: TimeSeries, index: int = 0) -> np.ndarray:
    """Apply ``transform`` and return the ``index``-th feature column as floats."""
    features = transform.apply(FeatureSet.from_time_series(series))  # type: ignore[attr-defined]
    return features.matrix.iloc[:, index].to_numpy(dtype="float64")


# -- Lag -------------------------------------------------------------------


def test_lag_shifts_values_forward(counting_series: TimeSeries) -> None:
    values = _column(Lag(2), counting_series)
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    np.testing.assert_array_equal(values[2:], np.arange(1.0, 11.0))


def test_lag_sorts_and_deduplicates_periods(counting_series: TimeSeries) -> None:
    transform = Lag([3, 1, 3, 1])
    assert transform.periods == (1, 3)
    features = transform.apply(FeatureSet.from_time_series(counting_series))
    assert features.features == ("close_lag1", "close_lag3")


def test_lag_warmup_equals_the_longest_period(counting_series: TimeSeries) -> None:
    features = Lag([1, 4]).apply(FeatureSet.from_time_series(counting_series))
    assert features.warmup == 4


@pytest.mark.parametrize("periods", [0, -1, [2, 0], []], ids=["zero", "negative", "mixed", "empty"])
def test_lag_rejects_bad_periods(periods: object) -> None:
    with pytest.raises(DataValidationError):
        Lag(periods)  # type: ignore[arg-type]


def test_lag_can_target_another_column(synthetic_series: TimeSeries) -> None:
    features = Lag(1, column="volume").apply(FeatureSet.from_time_series(synthetic_series))
    assert features.features == ("volume_lag1",)


def test_unknown_column_is_rejected(synthetic_series: TimeSeries) -> None:
    with pytest.raises(DataValidationError, match="not present"):
        Lag(1, column="nope").apply(FeatureSet.from_time_series(synthetic_series))


# -- RollingStat -----------------------------------------------------------


def test_rolling_mean_matches_hand_computation(counting_series: TimeSeries) -> None:
    values = _column(RollingStat(3, "mean"), counting_series)
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    np.testing.assert_allclose(values[2:], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])


def test_rolling_std_uses_the_sample_estimator(counting_series: TimeSeries) -> None:
    values = _column(RollingStat(3, "std"), counting_series)
    # Three consecutive integers have sample standard deviation exactly 1.
    np.testing.assert_allclose(values[2:], np.ones(10))


def test_rolling_min_and_max(counting_series: TimeSeries) -> None:
    low = _column(RollingStat(3, "min"), counting_series)
    high = _column(RollingStat(3, "max"), counting_series)
    np.testing.assert_array_equal(low[2:], np.arange(1.0, 11.0))
    np.testing.assert_array_equal(high[2:], np.arange(3.0, 13.0))


def test_rolling_warmup_is_window_minus_one(counting_series: TimeSeries) -> None:
    features = RollingStat(4).apply(FeatureSet.from_time_series(counting_series))
    assert features.warmup == 3


def test_rolling_min_periods_shrinks_the_warmup(counting_series: TimeSeries) -> None:
    features = RollingStat(4, "mean", min_periods=1).apply(
        FeatureSet.from_time_series(counting_series)
    )
    assert features.warmup == 0
    # With min_periods=1 the first value is the single observation itself.
    assert features.matrix.iloc[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 0}, "window must be positive"),
        ({"window": 3, "stat": "median"}, "unknown rolling statistic"),
        ({"window": 3, "min_periods": 0}, "min_periods"),
        ({"window": 3, "min_periods": 9}, "min_periods"),
    ],
    ids=["zero-window", "bad-stat", "zero-min-periods", "min-periods-too-large"],
)
def test_rolling_stat_rejects_bad_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        RollingStat(**kwargs)  # type: ignore[arg-type]


# -- Returns ---------------------------------------------------------------


def test_simple_returns_match_hand_computation(counting_series: TimeSeries) -> None:
    values = _column(Returns(1), counting_series)
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], 1.0 / np.arange(1.0, 12.0))


def test_log_returns_match_hand_computation(counting_series: TimeSeries) -> None:
    values = _column(Returns(1, log=True), counting_series)
    expected = np.log(np.arange(2.0, 13.0) / np.arange(1.0, 12.0))
    np.testing.assert_allclose(values[1:], expected)


def test_multi_period_returns(counting_series: TimeSeries) -> None:
    values = _column(Returns(3), counting_series)
    assert np.isnan(values[:3]).all()
    expected = np.arange(4.0, 13.0) / np.arange(1.0, 10.0) - 1.0
    np.testing.assert_allclose(values[3:], expected)


def test_log_returns_reject_non_positive_prices() -> None:
    series = _series([1.0, 0.0, 2.0])
    with pytest.raises(DataValidationError, match="strictly positive"):
        Returns(1, log=True).apply(FeatureSet.from_time_series(series))


def test_returns_reject_non_positive_periods() -> None:
    with pytest.raises(DataValidationError, match="periods must be positive"):
        Returns(0)


def test_log_return_column_name_differs_from_simple(counting_series: TimeSeries) -> None:
    simple = Returns(1).apply(FeatureSet.from_time_series(counting_series))
    logged = Returns(1, log=True).apply(FeatureSet.from_time_series(counting_series))
    assert simple.features == ("close_ret1",)
    assert logged.features == ("close_logret1",)


# -- CalendarFeatures ------------------------------------------------------


def test_calendar_dayofweek_matches_the_index(counting_series: TimeSeries) -> None:
    # 2024-01-01 is a Monday, so the first row is day 0.
    values = _column(CalendarFeatures(["dayofweek"]), counting_series)
    np.testing.assert_array_equal(values, counting_series.index.dayofweek.to_numpy())


def test_calendar_cyclic_encoding_is_bounded_and_wraps(counting_series: TimeSeries) -> None:
    features = CalendarFeatures(["dow_cyclic"]).apply(FeatureSet.from_time_series(counting_series))
    assert features.features == ("dt_dow_sin", "dt_dow_cos")
    sine = features.matrix["dt_dow_sin"].to_numpy()
    cosine = features.matrix["dt_dow_cos"].to_numpy()
    np.testing.assert_allclose(sine**2 + cosine**2, np.ones(len(sine)))
    # Monday sits at angle 0.
    assert sine[0] == pytest.approx(0.0)
    assert cosine[0] == pytest.approx(1.0)


def test_calendar_month_component(counting_series: TimeSeries) -> None:
    values = _column(CalendarFeatures(["month"]), counting_series)
    np.testing.assert_array_equal(values, np.ones(len(counting_series)))


def test_calendar_has_no_warmup(counting_series: TimeSeries) -> None:
    features = CalendarFeatures(["dayofweek", "month", "quarter"]).apply(
        FeatureSet.from_time_series(counting_series)
    )
    assert features.warmup == 0
    assert features.is_complete


def test_calendar_rejects_unknown_components() -> None:
    with pytest.raises(DataValidationError, match="unknown calendar component"):
        CalendarFeatures(["dayofweek", "moonphase"])


def test_calendar_rejects_an_empty_component_list() -> None:
    with pytest.raises(DataValidationError, match="at least one component"):
        CalendarFeatures([])


# -- RSI -------------------------------------------------------------------


def test_rsi_is_one_hundred_when_every_step_is_a_gain() -> None:
    series = _series([float(value) for value in range(1, 31)])
    values = _column(RSI(5), series)
    np.testing.assert_allclose(values[5:], np.full(25, 100.0))


def test_rsi_is_zero_when_every_step_is_a_loss() -> None:
    series = _series([float(value) for value in range(30, 0, -1)])
    values = _column(RSI(5), series)
    np.testing.assert_allclose(values[5:], np.zeros(25))


def test_rsi_is_neutral_for_a_flat_series() -> None:
    series = _series([5.0] * 30)
    values = _column(RSI(5), series)
    np.testing.assert_allclose(values[5:], np.full(25, 50.0))


def test_rsi_stays_within_bounds(synthetic_series: TimeSeries) -> None:
    values = _column(RSI(14), synthetic_series)
    valid = values[~np.isnan(values)]
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0


def test_rsi_warmup_equals_the_window(synthetic_series: TimeSeries) -> None:
    features = RSI(14).apply(FeatureSet.from_time_series(synthetic_series))
    assert features.warmup == 14


def test_rsi_rejects_a_non_positive_window() -> None:
    with pytest.raises(DataValidationError, match="window must be positive"):
        RSI(0)


# -- MACD ------------------------------------------------------------------


def test_macd_histogram_is_macd_minus_signal(counting_series: TimeSeries) -> None:
    features = MACD(fast=2, slow=4, signal=2).apply(FeatureSet.from_time_series(counting_series))
    macd = features.matrix["close_macd"].to_numpy()
    signal = features.matrix["close_macd_signal"].to_numpy()
    histogram = features.matrix["close_macd_hist"].to_numpy()
    np.testing.assert_allclose(histogram, macd - signal, equal_nan=True)


def test_macd_line_is_the_difference_of_two_emas(counting_series: TimeSeries) -> None:
    features = MACD(fast=2, slow=4, signal=2).apply(FeatureSet.from_time_series(counting_series))
    close = counting_series.target_series
    fast = close.ewm(span=2, adjust=False, min_periods=2).mean()
    slow = close.ewm(span=4, adjust=False, min_periods=4).mean()
    np.testing.assert_allclose(
        features.matrix["close_macd"].to_numpy(), (fast - slow).to_numpy(), equal_nan=True
    )


def test_macd_warmup_combines_both_smoothing_stages(synthetic_series: TimeSeries) -> None:
    # slow - 1 rows for the MACD line, then signal - 1 more for its signal.
    features = MACD().apply(FeatureSet.from_time_series(synthetic_series))
    assert features.warmup == (26 - 1) + (9 - 1)


def test_macd_rejects_slow_not_exceeding_fast() -> None:
    with pytest.raises(DataValidationError, match="slow span must exceed"):
        MACD(fast=10, slow=10)


@pytest.mark.parametrize(
    "kwargs", [{"fast": 0}, {"slow": -1}, {"signal": 0}], ids=["fast", "slow", "signal"]
)
def test_macd_rejects_non_positive_spans(kwargs: dict[str, int]) -> None:
    with pytest.raises(DataValidationError, match="spans must be positive"):
        MACD(**kwargs)


# -- Volatility ------------------------------------------------------------


def test_volatility_matches_the_rolling_std_of_log_returns() -> None:
    series = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    values = _column(Volatility(2, annualize=False), series)
    log_returns = np.log(series.target_series).diff().to_numpy()
    expected = np.array(
        [
            np.nan,
            np.nan,
            np.std([log_returns[1], log_returns[2]], ddof=1),
            np.std([log_returns[2], log_returns[3]], ddof=1),
            np.std([log_returns[3], log_returns[4]], ddof=1),
        ]
    )
    np.testing.assert_allclose(values, expected, equal_nan=True)


def test_annualised_volatility_scales_by_root_periods(synthetic_series: TimeSeries) -> None:
    raw = _column(Volatility(10, annualize=False), synthetic_series)
    annualised = _column(Volatility(10), synthetic_series)
    np.testing.assert_allclose(annualised, raw * np.sqrt(252.0), equal_nan=True)


def test_volatility_warmup_equals_the_window(synthetic_series: TimeSeries) -> None:
    features = Volatility(10).apply(FeatureSet.from_time_series(synthetic_series))
    assert features.warmup == 10


def test_volatility_column_name_marks_the_raw_variant(synthetic_series: TimeSeries) -> None:
    raw = Volatility(5, annualize=False).apply(FeatureSet.from_time_series(synthetic_series))
    assert raw.features == ("close_vol5_raw",)


def test_volatility_rejects_non_positive_prices() -> None:
    series = _series([1.0, 2.0, -1.0, 4.0])
    with pytest.raises(DataValidationError, match="strictly positive"):
        Volatility(2).apply(FeatureSet.from_time_series(series))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"window": 0}, "window must be positive"), ({"periods_per_year": 0}, "periods_per_year")],
    ids=["window", "periods-per-year"],
)
def test_volatility_rejects_bad_configuration(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        Volatility(**kwargs)
