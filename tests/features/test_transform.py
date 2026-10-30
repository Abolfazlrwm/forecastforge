"""Tests for the :class:`Transform` base contract."""

from __future__ import annotations

import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.features import FeatureSet, Lag, Transform
from forecastforge.schemas.time_series import TimeSeries


class _MismatchedTransform(Transform):
    """Declares two columns but produces one — the contract must reject it."""

    name = "mismatched"

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Claim two outputs that will never be produced."""
        return ("first", "second")

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Return a single column, violating the declaration."""
        return pd.DataFrame({"first": features.frame[features.target]}, index=features.frame.index)


class _MisalignedTransform(Transform):
    """Returns a frame on a different index than the input."""

    name = "misaligned"

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Declare a single output column."""
        return ("shifted",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Return rows that do not line up with the input timeline."""
        return pd.DataFrame({"shifted": [1.0, 2.0]})


class _RenameTargetTransform(Transform):
    """Tries to overwrite an existing column."""

    name = "clobber"

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Declare that it will emit a column named like the target."""
        return (features.target,)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Return a column that collides with the target."""
        return pd.DataFrame(
            {features.target: features.frame[features.target] * 2.0}, index=features.frame.index
        )


def test_apply_rejects_a_column_count_mismatch(counting_series: TimeSeries) -> None:
    features = FeatureSet.from_time_series(counting_series)
    with pytest.raises(DataValidationError, match="declared columns"):
        _MismatchedTransform().apply(features)


def test_apply_rejects_a_misaligned_index(counting_series: TimeSeries) -> None:
    features = FeatureSet.from_time_series(counting_series)
    with pytest.raises(DataValidationError, match="index"):
        _MisalignedTransform().apply(features)


def test_apply_rejects_overwriting_an_existing_column(counting_series: TimeSeries) -> None:
    features = FeatureSet.from_time_series(counting_series)
    with pytest.raises(DataValidationError, match="overwrite"):
        _RenameTargetTransform().apply(features)


def test_apply_returns_a_new_set_and_leaves_the_input_untouched(
    counting_series: TimeSeries,
) -> None:
    features = FeatureSet.from_time_series(counting_series)
    enriched = Lag(2).apply(features)
    assert features.features == ()
    assert enriched.features == ("close_lag2",)
    assert len(enriched.frame.columns) == len(features.frame.columns) + 1


def test_apply_preserves_the_original_target_and_index(counting_series: TimeSeries) -> None:
    features = Lag(2).apply(FeatureSet.from_time_series(counting_series))
    pd.testing.assert_series_equal(features.target_series, counting_series.target_series)
    pd.testing.assert_index_equal(features.index, counting_series.index)


def test_repr_reports_the_configuration() -> None:
    assert repr(Lag([5, 1])) == "Lag(periods=(1, 5), column=None)"
