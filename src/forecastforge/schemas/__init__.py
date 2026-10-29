"""Data and configuration schemas.

Currently provides the validated time-series container
(:class:`~forecastforge.schemas.time_series.TimeSeries`) and the structural
validation it enforces (:mod:`forecastforge.schemas.validation`).
"""

from __future__ import annotations

from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata
from forecastforge.schemas.validation import Gap, find_gaps, validate_frame

__all__ = [
    "Gap",
    "TimeSeries",
    "TimeSeriesMetadata",
    "find_gaps",
    "validate_frame",
]
