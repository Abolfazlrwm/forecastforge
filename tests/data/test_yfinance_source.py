"""Tests for the optional Yahoo Finance connector (fully offline)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from forecastforge.core.errors import DataError, MissingOptionalDependencyError
from forecastforge.data import YFinanceSource
from forecastforge.data.loaders import write_parquet
from forecastforge.schemas.time_series import TimeSeries


def _cache_frame() -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        index=index,
    )


def test_missing_dependency_raises(tmp_path: Path) -> None:
    source = YFinanceSource(cache_dir=tmp_path)
    with patch("forecastforge.data.yfinance_source.optional_import", return_value=None):
        with pytest.raises(MissingOptionalDependencyError, match="data.*extra"):
            source.fetch(symbol="SPY")


def test_cache_is_used_offline(tmp_path: Path) -> None:
    write_parquet(_cache_frame(), tmp_path / "SPY.parquet")
    source = YFinanceSource(cache_dir=tmp_path)
    ts = source.fetch(symbol="SPY")
    assert isinstance(ts, TimeSeries)
    assert ts.metadata.source == "yfinance"
    assert "close" in ts.frame.columns


def test_force_refresh_without_dependency_raises(tmp_path: Path) -> None:
    write_parquet(_cache_frame(), tmp_path / "SPY.parquet")
    source = YFinanceSource(cache_dir=tmp_path)
    with patch("forecastforge.data.yfinance_source.optional_import", return_value=None):
        with pytest.raises(MissingOptionalDependencyError):
            source.fetch(symbol="SPY", force_refresh=True)


def _multiindex_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["SPY"]]
    )
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    data = pd.DataFrame(1.0, index=index, columns=columns)
    return data


def test_download_normalizes_multiindex_columns() -> None:
    fake_yf = SimpleNamespace(download=lambda symbol, **kwargs: _multiindex_frame())
    out = YFinanceSource()._download("SPY", fake_yf)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        assert col in out.columns
    assert out.index.name == "date"


def test_download_normalizes_flat_columns() -> None:
    flat = _multiindex_frame()
    flat.columns = [c[0] for c in flat.columns]  # drop the ticker level
    fake_yf = SimpleNamespace(download=lambda symbol, **kwargs: flat)
    out = YFinanceSource()._download("SPY", fake_yf)
    assert "close" in out.columns


def test_download_empty_raises() -> None:
    fake_yf = SimpleNamespace(download=lambda symbol, **kwargs: pd.DataFrame())
    with pytest.raises(DataError, match="no data"):
        YFinanceSource()._download("SPY", fake_yf)
