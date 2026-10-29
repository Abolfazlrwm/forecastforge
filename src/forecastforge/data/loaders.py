"""File-based readers and writers for time-series frames.

These helpers round-trip a pandas DataFrame with a
:class:`~pandas.DatetimeIndex` to/from Parquet and CSV, preserving the index.
They are the persistence layer behind caching and the file-backed data source.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from forecastforge.core.errors import DataError

__all__ = ["read_csv", "read_parquet", "write_csv", "write_parquet"]


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame, preserving the index."""
    return _coerce_index(pd.read_parquet(path))


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to ``path`` as Parquet, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)
    return target


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by :func:`write_csv`, restoring the DatetimeIndex."""
    return _coerce_index(pd.read_csv(path, index_col=0))


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to ``path`` as CSV, keeping the index as a column."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target)
    return target


def _coerce_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the index is a :class:`~pandas.DatetimeIndex`."""
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.name is None:
            frame.index.name = "date"
        return frame
    with warnings.catch_warnings():
        # pandas warns (and, under strict warning filters, errors) when it cannot
        # infer a format even though we are explicitly coercing failures to NaT.
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(frame.index, errors="coerce")
    if parsed.isna().any():
        unparseable = [str(value) for value in frame.index[parsed.isna()]]
        raise DataError(f"file index could not be parsed as timestamps: {unparseable}")
    frame.index = parsed
    frame.index.name = frame.index.name or "date"
    return frame
