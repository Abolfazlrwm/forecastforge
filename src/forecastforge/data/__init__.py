"""Data ingestion and access layer.

Connectors, loaders, and the shared source interface for reading raw and
external data into a validated
:class:`~forecastforge.schemas.time_series.TimeSeries`.

The :class:`~forecastforge.data.synthetic.SyntheticSource` is fully offline and
deterministic and is the project's default dataset; the
:class:`~forecastforge.data.yfinance_source.YFinanceSource` is an optional,
network-backed connector installed via the ``data`` extra.
"""

from __future__ import annotations

from forecastforge.data.loaders import read_csv, read_parquet, write_csv, write_parquet
from forecastforge.data.source import BaseDataSource, DataSource
from forecastforge.data.synthetic import Regime, SyntheticConfig, SyntheticSource
from forecastforge.data.yfinance_source import YFinanceSource

__all__ = [
    "BaseDataSource",
    "DataSource",
    "Regime",
    "SyntheticConfig",
    "SyntheticSource",
    "YFinanceSource",
    "read_csv",
    "read_parquet",
    "write_csv",
    "write_parquet",
]
