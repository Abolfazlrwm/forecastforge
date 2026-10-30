"""Helpers shared by the feature-engineering tests."""

from __future__ import annotations

from forecastforge.schemas.time_series import TimeSeries

__all__ = ["slice_series"]


def slice_series(series: TimeSeries, last_row: int) -> TimeSeries:
    """Return ``series`` truncated to rows ``0..last_row`` inclusive."""
    return TimeSeries(
        frame=series.frame.iloc[: last_row + 1],
        target=series.target,
        metadata=series.metadata,
        covariates=series.covariates,
    )
