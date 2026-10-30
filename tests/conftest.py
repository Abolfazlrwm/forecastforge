"""Fixtures shared by the model and evaluation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from forecastforge.data import SyntheticConfig, SyntheticSource


@pytest.fixture
def synthetic_close() -> pd.Series:
    """A 600-row deterministic close series, long enough for several folds."""
    series = SyntheticSource(config=SyntheticConfig(seed=3, length=600)).fetch(symbol="SPY")
    return series.target_series
