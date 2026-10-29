"""Data-source interface and shared fetch logic.

A :class:`DataSource` produces a validated
:class:`~forecastforge.schemas.time_series.TimeSeries` for a given symbol. The
protocol lets downstream code accept *any* source — synthetic, file-backed, or a
live vendor — interchangeably.

:class:`BaseDataSource` implements the common contract (load → slice → validate →
wrap) so concrete sources only decide *how* to obtain the raw frame and never
re-implement the slicing or validation bookkeeping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from forecastforge.core.errors import DataError
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata

__all__ = ["BaseDataSource", "DataSource", "TimestampLike"]

TimestampLike = str | pd.Timestamp | date | datetime


@runtime_checkable
class DataSource(Protocol):
    """Anything that can produce a validated :class:`TimeSeries` for a symbol."""

    def fetch(
        self,
        *,
        symbol: str,
        start: TimestampLike | None = None,
        end: TimestampLike | None = None,
        target: str = "close",
        covariates: Sequence[str] = (),
        frequency: str | None = None,
        force_refresh: bool = False,
    ) -> TimeSeries:
        """Fetch a validated time series for ``symbol`` within ``[start, end]``."""
        ...


class BaseDataSource(ABC):
    """Shared fetch/slice/validate orchestration for concrete sources.

    Subclasses implement :meth:`_load_frame` (how to obtain the raw frame for a
    symbol) and :meth:`_attributes` (provenance to attach). The base class then
    slices the frame to the requested window and wraps it in a validated
    :class:`TimeSeries`.

    Concrete subclasses set the ``source_name`` and ``default_frequency`` class
    attributes to describe themselves.
    """

    source_name: str = "base"
    default_frequency: str | None = None

    def fetch(
        self,
        *,
        symbol: str,
        start: TimestampLike | None = None,
        end: TimestampLike | None = None,
        target: str = "close",
        covariates: Sequence[str] = (),
        frequency: str | None = None,
        force_refresh: bool = False,
    ) -> TimeSeries:
        """Fetch a validated time series for ``symbol`` within ``[start, end]``.

        Args:
            symbol: Instrument identifier passed through to the source.
            start: First timestamp to include (inclusive). ``None`` means from
                the beginning of the source's data.
            end: Last timestamp to include (inclusive). ``None`` means through
                the end of the source's data.
            target: Column to forecast.
            covariates: Additional columns used as exogenous inputs.
            frequency: Expected spacing as a pandas offset alias. Falls back to
                the source's :attr:`default_frequency` when omitted.
            force_refresh: When a source caches, bypass the cache and obtain
                fresh data.

        Returns:
            A validated :class:`TimeSeries`.

        Raises:
            DataError: If the resulting window is empty after slicing.
        """
        frame = self._load_frame(symbol=symbol, force_refresh=force_refresh)
        frame = self._slice(frame, start=start, end=end)
        if frame.empty:
            window = f" in [{start}, {end}]" if start is not None or end is not None else ""
            raise DataError(f"{self.source_name} source returned no rows for {symbol!r}{window}")

        return TimeSeries(
            frame=frame,
            target=target,
            metadata=TimeSeriesMetadata(
                symbol=symbol,
                source=self.source_name,
                frequency=frequency or self.default_frequency,
                attributes=self._attributes(symbol=symbol),
            ),
            covariates=tuple(covariates),
        )

    @staticmethod
    def _slice(
        frame: pd.DataFrame,
        *,
        start: TimestampLike | None,
        end: TimestampLike | None,
    ) -> pd.DataFrame:
        """Return the rows of ``frame`` whose index lies in ``[start, end]``."""
        if start is None and end is None:
            return frame

        mask = pd.Series(True, index=frame.index)
        if start is not None:
            lower = pd.Series(frame.index >= pd.Timestamp(start), index=frame.index)
            mask &= lower
        if end is not None:
            upper = pd.Series(frame.index <= pd.Timestamp(end), index=frame.index)
            mask &= upper
        return frame.loc[mask]

    @abstractmethod
    def _load_frame(self, *, symbol: str, force_refresh: bool) -> pd.DataFrame:
        """Return the full raw frame for ``symbol`` (before any slicing)."""

    @abstractmethod
    def _attributes(self, *, symbol: str) -> dict[str, Any]:
        """Return provenance attributes to attach to the series metadata."""
