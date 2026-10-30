"""Shared fixtures for the feature-engineering tests."""

from __future__ import annotations

import pandas as pd
import pytest

from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata


@pytest.fixture
def synthetic_series() -> TimeSeries:
    """A 300-row deterministic series, long enough for every warm-up period."""
    return SyntheticSource(config=SyntheticConfig(seed=11, length=300)).fetch(symbol="SPY")


@pytest.fixture
def counting_series() -> TimeSeries:
    """Twelve rows whose target is ``1.0 .. 12.0``.

    Integer prices make hand-computed expectations exact rather than
    approximately right, which keeps the value tests readable.
    """
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = pd.DataFrame({"close": [float(value) for value in range(1, 13)]}, index=index)
    frame.index.name = "date"
    return TimeSeries(
        frame=frame,
        target="close",
        metadata=TimeSeriesMetadata(symbol="TEST", source="fixture", frequency="D"),
    )
