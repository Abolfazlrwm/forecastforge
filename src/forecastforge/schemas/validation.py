"""Structural validation for the ForecastForge time-series contract.

A time series is a :class:`pandas.DataFrame` whose index is a sorted, unique
:class:`~pandas.DatetimeIndex` and which contains at least one numeric target
column, plus any declared numeric covariates. Everything downstream — feature
engineering, backtesting, serving — relies on that contract, so it is checked
once at the boundary and never re-derived.

Validation is deliberately strict: silent gaps, duplicate timestamps, and
NaNs propagate into features and metrics as quietly wrong numbers, which is
far more expensive to debug than an explicit rejection here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError

__all__ = ["Gap", "find_gaps", "validate_frame"]

_MAX_EXAMPLES = 3
_MAX_LISTED_COLUMNS = 6


@dataclass(frozen=True, slots=True)
class Gap:
    """A contiguous run of periods that the index is missing.

    Attributes:
        start: First expected-but-absent timestamp in the run.
        end: Last expected-but-absent timestamp in the run.
        missing: Number of absent periods in the run.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    missing: int

    def __str__(self) -> str:
        return f"{self.start.date()}..{self.end.date()} ({self.missing} missing)"


def validate_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    covariates: Sequence[str] = (),
    frequency: str | None = None,
    require_complete: bool = False,
) -> None:
    """Check that ``frame`` satisfies the time-series contract.

    Args:
        frame: The candidate dataset. Its index must be a
            :class:`~pandas.DatetimeIndex` that is sorted and free of
            duplicates.
        target: Name of the numeric column holding the value to forecast.
        covariates: Names of additional numeric columns that will be used as
            exogenous inputs.
        frequency: pandas offset alias (``"B"``, ``"D"``, …) used only to
            detect gaps. Required when ``require_complete`` is true.
        require_complete: When true, a series with missing periods is
            rejected. Off by default because real market data legitimately
            skips weekends, holidays, and trading halts.

    Raises:
        DataValidationError: If any part of the contract is violated. The
            message names the offending column or timestamp.
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(f"expected a pandas DataFrame, got {type(frame).__name__}")

    if frame.empty:
        raise DataValidationError("time series is empty: at least one row is required")

    index = _validate_index(frame.index)
    _validate_columns(frame, target=target, covariates=covariates)
    _validate_values(frame, [target, *covariates])

    if require_complete:
        if frequency is None:
            raise DataValidationError(
                "require_complete=True needs a frequency to know which periods are expected, "
                "e.g. frequency='B' for business days"
            )
        gaps = find_gaps(index, frequency)
        if gaps:
            listed = ", ".join(str(gap) for gap in gaps[:_MAX_EXAMPLES])
            more = "" if len(gaps) <= _MAX_EXAMPLES else f" (+{len(gaps) - _MAX_EXAMPLES} more)"
            raise DataValidationError(
                f"time series is incomplete at frequency {frequency!r}: "
                f"{len(gaps)} gap(s) found — {listed}{more}"
            )


def find_gaps(index: pd.DatetimeIndex, frequency: str) -> tuple[Gap, ...]:
    """Return the contiguous runs of periods missing from ``index``.

    Args:
        index: A sorted, unique :class:`~pandas.DatetimeIndex`.
        frequency: pandas offset alias describing the expected spacing,
            e.g. ``"B"`` for business days.

    Returns:
        One :class:`Gap` per contiguous run of missing periods, in
        chronological order. Empty when the index is complete.

    Raises:
        DataValidationError: If ``frequency`` is not a valid pandas offset
            alias.
    """
    if len(index) <= 1:
        return ()

    try:
        expected = pd.date_range(start=index.min(), end=index.max(), freq=frequency)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(
            f"frequency {frequency!r} is not a valid pandas offset alias "
            "(use e.g. 'B' for business days or 'D' for calendar days)"
        ) from exc

    present = set(index)
    gaps: list[Gap] = []
    run: list[pd.Timestamp] = []

    for timestamp in expected:
        if timestamp in present:
            if run:
                gaps.append(Gap(start=run[0], end=run[-1], missing=len(run)))
                run = []
            continue
        run.append(timestamp)

    if run:
        gaps.append(Gap(start=run[0], end=run[-1], missing=len(run)))

    return tuple(gaps)


def _validate_index(index: pd.Index) -> pd.DatetimeIndex:
    """Check the index is a clean, sorted DatetimeIndex and return it narrowed."""
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(
            f"index must be a pandas DatetimeIndex, got {type(index).__name__}"
        )

    if index.hasnans:
        raise DataValidationError("index contains NaT values; every row must have a real timestamp")

    duplicated = index[index.duplicated()]
    if len(duplicated) > 0:
        examples = ", ".join(str(ts) for ts in duplicated[:_MAX_EXAMPLES])
        raise DataValidationError(
            f"index contains {len(duplicated)} duplicate timestamp(s), e.g. {examples}; "
            "each timestamp may appear only once"
        )

    if not index.is_monotonic_increasing:
        for position in range(1, len(index)):
            if index[position] <= index[position - 1]:
                raise DataValidationError(
                    f"index is not sorted in increasing order: {index[position]} "
                    f"at position {position} follows {index[position - 1]}"
                )

    return index


def _validate_columns(frame: pd.DataFrame, *, target: str, covariates: Sequence[str]) -> None:
    """Check that every declared column exists."""
    if target not in frame.columns:
        raise DataValidationError(
            f"target column {target!r} is missing; available columns: {_describe_columns(frame)}"
        )

    missing = [name for name in covariates if name not in frame.columns]
    if missing:
        raise DataValidationError(
            f"covariate column(s) {missing} are missing; "
            f"available columns: {_describe_columns(frame)}"
        )


def _validate_values(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Check every declared column is numeric, finite, and free of NaN."""
    for column in columns:
        series = frame[column]

        if not pd.api.types.is_numeric_dtype(series):
            raise DataValidationError(
                f"column {column!r} must be numeric, got dtype {series.dtype}"
            )

        nulls = series.isna()
        if nulls.any():
            first = series.index[nulls][0]
            raise DataValidationError(
                f"column {column!r} contains {int(nulls.sum())} NaN value(s), "
                f"first at {first}; drop or fill them before constructing a TimeSeries"
            )

        infinite = np.isinf(series.to_numpy(dtype="float64"))
        if infinite.any():
            first = series.index[infinite][0]
            raise DataValidationError(
                f"column {column!r} contains {int(infinite.sum())} infinite value(s), "
                f"first at {first}"
            )


def _describe_columns(frame: pd.DataFrame) -> str:
    """Render the available column names for an error message."""
    names = [repr(str(name)) for name in frame.columns[:_MAX_LISTED_COLUMNS]]
    if len(frame.columns) > _MAX_LISTED_COLUMNS:
        names.append("...")
    return ", ".join(names) if names else "(none)"
