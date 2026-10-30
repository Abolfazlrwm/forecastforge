"""Tests for the :class:`FeatureSet` container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.features import FeatureSet, Lag
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata


def _metadata() -> TimeSeriesMetadata:
    """Return throwaway provenance for hand-built frames."""
    return TimeSeriesMetadata(symbol="TEST", source="fixture", frequency="D")


def _frame(values: list[float], *, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a single-column ``close`` frame on a daily index."""
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=index)


def test_from_time_series_carries_target_and_metadata(counting_series: TimeSeries) -> None:
    features = FeatureSet.from_time_series(counting_series)
    assert features.target == counting_series.target
    assert features.metadata == counting_series.metadata
    assert features.features == ()
    assert len(features) == len(counting_series)


def test_empty_feature_set_is_complete_and_has_no_warmup() -> None:
    features = FeatureSet(
        frame=_frame([1.0, 2.0, 3.0]), target="close", features=(), metadata=_metadata()
    )
    assert features.warmup == 0
    assert features.is_complete
    assert features.n_incomplete_rows == 0


def test_warmup_counts_leading_incomplete_rows(counting_series: TimeSeries) -> None:
    features = Lag(3).apply(FeatureSet.from_time_series(counting_series))
    assert features.warmup == 3
    assert features.n_incomplete_rows == 3
    assert not features.is_complete


def test_warmup_is_zero_when_nothing_is_missing(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series)).drop_warmup()
    assert features.warmup == 0
    assert features.is_complete


def test_warmup_counts_every_row_when_all_are_incomplete() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    frame["bad"] = [np.nan, np.nan, np.nan]
    features = FeatureSet(frame=frame, target="close", features=("bad",), metadata=_metadata())
    assert features.warmup == 3
    assert not features.is_complete


def test_nan_after_a_complete_row_is_reported_but_not_as_warmup() -> None:
    frame = _frame([1.0, 2.0, 3.0, 4.0])
    frame["gappy"] = [np.nan, 1.0, np.nan, 2.0]
    features = FeatureSet(frame=frame, target="close", features=("gappy",), metadata=_metadata())
    assert features.warmup == 1
    assert features.n_incomplete_rows == 2
    assert not features.is_complete


def test_drop_warmup_removes_exactly_the_prefix(counting_series: TimeSeries) -> None:
    features = Lag(3).apply(FeatureSet.from_time_series(counting_series))
    trimmed = features.drop_warmup()
    assert len(trimmed) == len(features) - 3
    assert trimmed.is_complete
    assert trimmed.index[0] == features.index[3]


def test_drop_warmup_is_a_noop_without_nan(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series)).drop_warmup()
    assert features.drop_warmup() is features


def test_dropna_removes_scattered_rows() -> None:
    frame = _frame([1.0, 2.0, 3.0, 4.0])
    frame["gappy"] = [np.nan, 1.0, np.nan, 2.0]
    features = FeatureSet(frame=frame, target="close", features=("gappy",), metadata=_metadata())
    dropped = features.dropna()
    assert len(dropped) == 2
    assert dropped.is_complete


def test_dropna_is_a_noop_without_features() -> None:
    features = FeatureSet(
        frame=_frame([1.0, 2.0]), target="close", features=(), metadata=_metadata()
    )
    assert features.dropna() is features


def test_to_time_series_round_trips_when_complete(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series)).drop_warmup()
    series = features.to_time_series()
    assert isinstance(series, TimeSeries)
    assert series.covariates == features.features
    assert len(series) == len(features)


def test_to_time_series_rejects_remaining_nan(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series))
    with pytest.raises(DataValidationError, match="NaN"):
        features.to_time_series()


def test_extend_appends_columns_in_order(counting_series: TimeSeries) -> None:
    base = FeatureSet.from_time_series(counting_series)
    added = pd.DataFrame({"a": np.arange(len(base), dtype=float)}, index=base.index)
    extended = base.extend(added)
    assert extended.features == ("a",)


def test_extend_rejects_a_misaligned_index(counting_series: TimeSeries) -> None:
    base = FeatureSet.from_time_series(counting_series)
    added = pd.DataFrame({"a": [1.0, 2.0]}, index=base.index[:2])
    with pytest.raises(DataValidationError, match="index"):
        base.extend(added)


def test_extend_rejects_a_column_collision(counting_series: TimeSeries) -> None:
    base = FeatureSet.from_time_series(counting_series)
    added = pd.DataFrame({"close": np.zeros(len(base))}, index=base.index)
    with pytest.raises(DataValidationError, match="overwrite"):
        base.extend(added)


def test_applying_the_same_transform_twice_is_rejected(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series))
    with pytest.raises(DataValidationError, match="overwrite"):
        Lag(2).apply(features)


def test_duplicate_column_names_are_rejected() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    frame["shadow"] = frame["close"]
    frame.columns = ["close", "close"]
    with pytest.raises(DataValidationError, match="duplicate column"):
        FeatureSet(frame=frame, target="close", features=(), metadata=_metadata())


def test_missing_target_is_rejected() -> None:
    frame = _frame([1.0, 2.0]).rename(columns={"close": "price"})
    with pytest.raises(DataValidationError, match="target column"):
        FeatureSet(frame=frame, target="close", features=(), metadata=_metadata())


def test_nan_target_is_rejected() -> None:
    frame = _frame([1.0, np.nan, 3.0])
    with pytest.raises(DataValidationError, match="NaN"):
        FeatureSet(frame=frame, target="close", features=(), metadata=_metadata())


def test_missing_feature_column_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="feature column"):
        FeatureSet(
            frame=_frame([1.0, 2.0]), target="close", features=("nope",), metadata=_metadata()
        )


def test_non_numeric_feature_is_rejected() -> None:
    frame = _frame([1.0, 2.0])
    frame["label"] = ["a", "b"]
    with pytest.raises(DataValidationError, match="must be numeric"):
        FeatureSet(frame=frame, target="close", features=("label",), metadata=_metadata())


def test_matrix_returns_only_feature_columns(counting_series: TimeSeries) -> None:
    features = Lag([1, 2]).apply(FeatureSet.from_time_series(counting_series))
    assert list(features.matrix.columns) == list(features.features)
    assert "close" not in features.matrix.columns


def test_unsorted_index_is_rejected() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    frame = frame.iloc[[2, 0, 1]]
    with pytest.raises(DataValidationError, match="sorted"):
        FeatureSet(frame=frame, target="close", features=(), metadata=_metadata())
