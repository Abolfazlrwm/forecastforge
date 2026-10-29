"""The validated time-series container.

:class:`TimeSeries` pairs a pandas DataFrame with the metadata needed to
interpret it, and validates the frame on construction so that every
downstream stage can assume a clean, sorted, NaN-free dataset.

Immutability is shallow by design: the dataclass cannot be rebound, and the
frame is copied on construction so later mutation of the caller's frame
cannot corrupt the container. The returned ``frame`` should still be treated
as read-only; callers that need to derive new columns should copy it first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from forecastforge.schemas.validation import validate_frame

__all__ = ["TimeSeries", "TimeSeriesMetadata"]


class TimeSeriesMetadata(BaseModel):
    """Validated provenance and interpretation details for a :class:`TimeSeries`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(
        min_length=1,
        description="Instrument or dataset identifier, e.g. 'SPY'.",
    )
    source: str = Field(
        default="unknown",
        min_length=1,
        description="Name of the component that produced the data, e.g. 'synthetic'.",
    )
    frequency: str | None = Field(
        default=None,
        description=(
            "Expected spacing as a pandas offset alias ('B' for business days, "
            "'D' for calendar days). Used to detect gaps; None means unknown."
        ),
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form provenance details, e.g. the generator seed.",
    )


@dataclass(frozen=True, slots=True, eq=False)
class TimeSeries:
    """An immutable, validated univariate time series with optional covariates.

    Args:
        frame: The dataset. Must have a sorted, unique
            :class:`~pandas.DatetimeIndex`; ``target`` and every name in
            ``covariates`` must be present, numeric, finite, and non-null.
        target: Column to forecast.
        metadata: Provenance details; see :class:`TimeSeriesMetadata`.
        covariates: Additional numeric columns used as exogenous inputs.

    Raises:
        DataValidationError: If ``frame`` violates the contract.

    Note:
        Equality is identity-based (``eq=False``). Comparing two DataFrames
        with ``==`` is element-wise and returns a frame, which makes value
        equality ambiguous and easy to misuse; compare ``frame.equals(...)``
        explicitly when that is what you mean.
    """

    frame: pd.DataFrame
    target: str
    metadata: TimeSeriesMetadata
    covariates: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        validate_frame(
            self.frame,
            target=self.target,
            covariates=self.covariates,
            frequency=self.metadata.frequency,
        )
        object.__setattr__(self, "frame", self.frame.copy())

    def __len__(self) -> int:
        """Return the number of observations."""
        return len(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The validated time index, guaranteed to be sorted and unique."""
        return cast("pd.DatetimeIndex", self.frame.index)

    @property
    def target_series(self) -> pd.Series:
        """The target column as a series."""
        return self.frame[self.target]
