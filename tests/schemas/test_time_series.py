"""Tests for forecastforge.schemas.time_series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from forecastforge.core.errors import DataValidationError
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata


def _frame(periods: int = 4) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame({"close": np.arange(periods, dtype="float64")}, index=index)


def _series(**overrides: object) -> TimeSeries:
    frame = _frame()
    metadata = TimeSeriesMetadata(symbol="TEST", source="unit-test", frequency="D")
    return TimeSeries(frame=frame, target="close", metadata=metadata, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructs_from_a_valid_frame() -> None:
    series = _series()

    assert len(series) == 4
    assert series.target == "close"
    assert series.covariates == ()
    assert series.metadata.symbol == "TEST"


def test_index_is_a_datetime_index() -> None:
    assert isinstance(_series().index, pd.DatetimeIndex)


def test_target_series_returns_the_target_column() -> None:
    series = _series()

    pd.testing.assert_series_equal(series.target_series, series.frame["close"])


def test_accepts_declared_covariates() -> None:
    frame = _frame()
    frame["volume"] = np.arange(len(frame), dtype="float64")
    metadata = TimeSeriesMetadata(symbol="TEST", frequency="D")

    series = TimeSeries(frame=frame, target="close", metadata=metadata, covariates=("volume",))

    assert series.covariates == ("volume",)


def test_frame_is_copied_so_source_mutation_cannot_corrupt_the_series() -> None:
    frame = _frame()
    series = TimeSeries(frame=frame, target="close", metadata=TimeSeriesMetadata(symbol="TEST"))

    frame.loc[:, "close"] = 0.0

    assert series.frame["close"].iloc[0] == 0.0
    assert series.frame["close"].iloc[1] == 1.0


def test_container_cannot_be_rebound() -> None:
    series = _series()

    with pytest.raises(AttributeError):
        series.target = "open"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation is enforced on construction
# ---------------------------------------------------------------------------


def test_rejects_a_frame_that_violates_the_contract() -> None:
    frame = _frame()
    frame.iloc[0, 0] = np.nan

    with pytest.raises(DataValidationError, match="NaN"):
        TimeSeries(frame=frame, target="close", metadata=TimeSeriesMetadata(symbol="TEST"))


def test_rejects_a_missing_covariate() -> None:
    with pytest.raises(DataValidationError, match="covariate"):
        TimeSeries(
            frame=_frame(),
            target="close",
            metadata=TimeSeriesMetadata(symbol="TEST"),
            covariates=("volume",),
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_defaults() -> None:
    metadata = TimeSeriesMetadata(symbol="SPY")

    assert metadata.source == "unknown"
    assert metadata.frequency is None
    assert metadata.attributes == {}


def test_metadata_is_frozen() -> None:
    metadata = TimeSeriesMetadata(symbol="SPY")

    with pytest.raises(ValidationError):
        metadata.symbol = "QQQ"  # type: ignore[misc]


def test_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TimeSeriesMetadata(symbol="SPY", ticker="SPY")  # type: ignore[call-arg]


def test_metadata_rejects_empty_symbol() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        TimeSeriesMetadata(symbol="")


def test_metadata_round_trips_through_a_dict() -> None:
    metadata = TimeSeriesMetadata(
        symbol="SPY", source="synthetic", frequency="B", attributes={"seed": 7}
    )

    assert TimeSeriesMetadata.model_validate(metadata.model_dump()) == metadata
