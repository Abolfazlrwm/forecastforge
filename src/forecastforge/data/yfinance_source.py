"""Optional Yahoo Finance data source.

This is an *optional* connector: it depends on the ``yfinance`` package, which
is only installed with the ``data`` extra. The import is resolved lazily through
:func:`forecastforge.config._optional.optional_import`, so the core package and
every test import cleanly without network access or the dependency present.

Downloaded data is cached to Parquet under ``cache_dir`` (default ``data/raw``).
A subsequent fetch for the same symbol reads the cache unless ``force_refresh``
is set, which keeps repeat runs and offline tests fast and network-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd

from forecastforge.config._optional import optional_import
from forecastforge.core.errors import DataError, MissingOptionalDependencyError
from forecastforge.data.loaders import read_parquet, write_parquet
from forecastforge.data.source import BaseDataSource

__all__ = ["YFinanceSource"]

_COLUMN_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


@dataclass(frozen=True, slots=True)
class YFinanceSource(BaseDataSource):
    """A cached Yahoo Finance connector, lazily importing ``yfinance``.

    Args:
        cache_dir: Directory for the Parquet cache. Created on first write.
        default_start: Default download window start (ISO date string).
        default_end: Default download window end; ``None`` means "up to now".
        auto_adjust: Forwarded to ``yfinance.download`` to adjust for splits/
            dividends on the ``close`` column.
    """

    cache_dir: Path = field(default_factory=lambda: Path("data/raw"))
    default_start: str = "2015-01-01"
    default_end: str | None = None
    auto_adjust: bool = True

    source_name = "yfinance"
    default_frequency = "B"

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return Path(self.cache_dir) / f"{safe}.parquet"

    def _load_frame(self, *, symbol: str, force_refresh: bool) -> pd.DataFrame:
        cache_path = self._cache_path(symbol)
        if not force_refresh and cache_path.exists():
            return read_parquet(cache_path)

        yf = optional_import("yfinance")
        if yf is None:
            raise MissingOptionalDependencyError(
                "the yfinance data source requires the optional 'data' extra: "
                "install with `pip install 'forecastforge[data]'`"
            )

        frame = self._download(symbol, yf)
        write_parquet(frame, cache_path)
        return frame

    def _download(self, symbol: str, yf: ModuleType) -> pd.DataFrame:
        """Download and normalize a frame for ``symbol`` using ``yf``."""
        raw = yf.download(
            symbol,
            start=self.default_start,
            end=self.default_end,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            raise DataError(f"yfinance returned no data for {symbol!r}")

        frame = raw
        if isinstance(frame.columns, pd.MultiIndex):
            level_values = frame.columns.get_level_values(1)
            if symbol in set(level_values):
                frame = frame.xs(symbol, axis=1, level=1)
            else:
                frame = frame.droplevel(1)
        frame = frame.rename(columns=_COLUMN_RENAME)
        frame.index.name = "date"

        keep = [col for col in _COLUMN_RENAME.values() if col in frame.columns]
        return cast("pd.DataFrame", frame[keep])

    def _attributes(self, *, symbol: str) -> dict[str, Any]:
        return {"cache_dir": str(self.cache_dir), "default_start": self.default_start}
