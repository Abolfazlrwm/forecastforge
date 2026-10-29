"""Tests for the deterministic synthetic data source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataError
from forecastforge.data import DataSource, Regime, SyntheticConfig, SyntheticSource
from forecastforge.schemas.time_series import TimeSeries


def _close(source: SyntheticSource, symbol: str = "SPY") -> pd.Series:
    ts = source.fetch(symbol=symbol)
    assert isinstance(ts, TimeSeries)
    return ts.target_series


def test_deterministic_for_same_seed() -> None:
    a = _close(SyntheticSource())
    b = _close(SyntheticSource())
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_different_seed_changes_series() -> None:
    base = _close(SyntheticSource())
    other = _close(SyntheticSource(config=SyntheticConfig(seed=7)))
    assert not np.array_equal(base.to_numpy(), other.to_numpy())


def test_length_matches_config() -> None:
    ts = SyntheticSource(config=SyntheticConfig(length=250, seed=1)).fetch(symbol="SPY")
    assert len(ts) == 250
    assert isinstance(ts.index, pd.DatetimeIndex)


def test_default_columns_present() -> None:
    ts = SyntheticSource().fetch(symbol="SPY")
    for col in ("open", "high", "low", "close", "volume"):
        assert col in ts.frame.columns


def test_column_subset_selection() -> None:
    config = SyntheticConfig(columns=("close", "volume"))
    ts = SyntheticSource(config=config).fetch(symbol="SPY", covariates=("volume",))
    assert set(ts.frame.columns) == {"close", "volume"}


def test_unknown_column_rejected() -> None:
    config = SyntheticConfig(columns=("close", "not_a_column"))
    with pytest.raises(DataError, match="unknown synthetic columns"):
        SyntheticSource(config=config).fetch(symbol="SPY")


def test_level_shift_creates_jump() -> None:
    config = SyntheticConfig(
        length=600,
        seed=3,
        volatility=0.0,
        seasonality_amplitude=0.0,
        regimes=(Regime(at=500, level_shift=0.5),),
    )
    ts = SyntheticSource(config=config).fetch(symbol="SPY")
    close = ts.target_series.to_numpy()
    ratio = close[500] / close[499]
    assert ratio > 1.3  # exp(0.5) ~= 1.65


def test_drift_shift_changes_path() -> None:
    baseline = _close(SyntheticSource(config=SyntheticConfig(seed=11)))
    shifted = _close(
        SyntheticSource(
            config=SyntheticConfig(seed=11, regimes=(Regime(at=500, drift_shift=0.002),)),
        )
    )
    assert not np.array_equal(baseline.to_numpy(), shifted.to_numpy())


def test_fetch_slices_by_window() -> None:
    ts = SyntheticSource().fetch(symbol="SPY", start="2016-01-01", end="2016-12-31")
    assert len(ts) < 1000
    assert ts.index.min() >= pd.Timestamp("2016-01-01")
    assert ts.index.max() <= pd.Timestamp("2016-12-31")


def test_metadata_records_seed() -> None:
    config = SyntheticConfig(seed=99, regimes=(Regime(at=10),))
    ts = SyntheticSource(config=config).fetch(symbol="SPY")
    assert ts.metadata.source == "synthetic"
    assert ts.metadata.attributes["seed"] == 99
    assert ts.metadata.attributes["regimes"] == [10]


def test_satisfies_data_source_protocol() -> None:
    assert isinstance(SyntheticSource(), DataSource)


@pytest.mark.parametrize("length", [0, -5])
def test_non_positive_length_rejected(length: int) -> None:
    with pytest.raises(DataError, match="length must be positive"):
        SyntheticSource(config=SyntheticConfig(length=length)).fetch(symbol="SPY")


def test_regime_out_of_range_rejected() -> None:
    config = SyntheticConfig(length=100, regimes=(Regime(at=500),))
    with pytest.raises(DataError, match="out of range"):
        SyntheticSource(config=config).fetch(symbol="SPY")


def test_frame_is_copied_not_aliased() -> None:
    from forecastforge.schemas.time_series import TimeSeriesMetadata

    original = SyntheticSource().fetch(symbol="SPY").frame.copy()
    ts = TimeSeries(
        frame=original,
        target="close",
        metadata=TimeSeriesMetadata(symbol="SPY"),
    )
    # Mutating the caller's frame must not corrupt the stored container.
    original.iloc[0, original.columns.get_loc("close")] = -999.0
    assert ts.frame["close"].iloc[0] != -999.0
