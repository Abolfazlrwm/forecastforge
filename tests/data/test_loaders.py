"""Tests for the CSV/Parquet file loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecastforge.core.errors import DataError
from forecastforge.data import read_csv, read_parquet, write_csv, write_parquet


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    index.name = "date"
    return pd.DataFrame(
        {"close": [100.0, 101.2, 99.8, 102.5, 103.0], "volume": [1, 2, 3, 4, 5]},
        index=index,
    )


def test_parquet_roundtrip(tmp_path: Path) -> None:
    frame = _sample_frame()
    path = write_parquet(frame, tmp_path / "out.parquet")
    back = read_parquet(path)
    pd.testing.assert_frame_equal(frame, back, check_freq=False)
    assert isinstance(back.index, pd.DatetimeIndex)


def test_csv_roundtrip(tmp_path: Path) -> None:
    frame = _sample_frame()
    path = write_csv(frame, tmp_path / "out.csv")
    back = read_csv(path)
    pd.testing.assert_frame_equal(frame, back, check_freq=False)
    assert isinstance(back.index, pd.DatetimeIndex)


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "out.parquet"
    write_parquet(_sample_frame(), target)
    assert target.exists()


def test_parquet_preserves_index_name(tmp_path: Path) -> None:
    frame = _sample_frame()
    frame.index.name = None
    path = write_parquet(frame, tmp_path / "no_name.parquet")
    back = read_parquet(path)
    assert back.index.name == "date"


def test_read_csv_coerces_index_name(tmp_path: Path) -> None:
    frame = _sample_frame()
    frame.index.name = None
    path = write_csv(frame, tmp_path / "csv_no_name.csv")
    back = read_csv(path)
    assert back.index.name == "date"


def test_unparseable_index_raises(tmp_path: Path) -> None:
    bad = pd.DataFrame({"value": [1, 2, 3]}, index=["alpha", "beta", "gamma"])
    path = tmp_path / "bad.csv"
    bad.to_csv(path)
    with pytest.raises(DataError, match="could not be parsed as timestamps"):
        read_csv(path)
