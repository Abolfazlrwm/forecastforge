"""Tests for :class:`FeaturePipeline` composition."""

from __future__ import annotations

import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.features import (
    CalendarFeatures,
    FeaturePipeline,
    FeatureSet,
    Lag,
    Returns,
    RollingStat,
)
from forecastforge.schemas.time_series import TimeSeries


def test_empty_pipeline_returns_the_input_unchanged(counting_series: TimeSeries) -> None:
    features = FeaturePipeline([]).apply(counting_series)
    assert features.features == ()
    assert features.warmup == 0


def test_steps_run_in_the_declared_order(counting_series: TimeSeries) -> None:
    pipeline = FeaturePipeline([Lag(1), Lag(2)])
    assert pipeline.feature_names(counting_series) == ("close_lag1", "close_lag2")


def test_a_later_step_can_consume_an_earlier_step_s_column(
    synthetic_series: TimeSeries,
) -> None:
    pipeline = FeaturePipeline(
        [
            Returns(periods=1, log=True),
            RollingStat(window=5, stat="std", column="close_logret1"),
        ]
    )
    features = pipeline.apply(synthetic_series)
    assert features.features == ("close_logret1", "close_logret1_rolling_std_5")
    # One row for the first return, then four more to fill the window.
    assert features.warmup == 5


def test_a_step_that_needs_a_later_column_fails_clearly(
    synthetic_series: TimeSeries,
) -> None:
    pipeline = FeaturePipeline(
        [
            RollingStat(window=5, stat="std", column="close_logret1"),
            Returns(periods=1, log=True),
        ]
    )
    with pytest.raises(DataValidationError, match="not present"):
        pipeline.apply(synthetic_series)


def test_pipeline_accepts_an_existing_feature_set(counting_series: TimeSeries) -> None:
    base = Lag(2).apply(FeatureSet.from_time_series(counting_series))
    extended = FeaturePipeline([Lag(3)]).apply(base)
    assert extended.features == ("close_lag2", "close_lag3")


def test_pipeline_composes_with_calendar_features(counting_series: TimeSeries) -> None:
    pipeline = FeaturePipeline(
        [Lag(1), CalendarFeatures(["dayofweek"]), Returns(1)],
    )
    features = pipeline.apply(counting_series)
    assert features.features == ("close_lag1", "dt_dayofweek", "close_ret1")
    # The lag dominates the warm-up; calendar features add none.
    assert features.warmup == 1


def test_pipeline_length_and_iteration(counting_series: TimeSeries) -> None:
    steps = [Lag(1), Returns(1)]
    pipeline = FeaturePipeline(steps)
    assert len(pipeline) == 2
    assert list(pipeline) == steps


def test_pipeline_repr_lists_every_step() -> None:
    text = repr(FeaturePipeline([Lag(2), Returns(1, log=True)]))
    assert "Lag(periods=(2,)" in text
    assert "Returns(periods=1, log=True" in text


def test_pipeline_does_not_mutate_the_input_series(counting_series: TimeSeries) -> None:
    before = list(counting_series.frame.columns)
    FeaturePipeline([Lag(1), Returns(1)]).apply(counting_series)
    assert list(counting_series.frame.columns) == before
