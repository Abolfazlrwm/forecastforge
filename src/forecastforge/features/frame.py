"""The working container for feature engineering.

:class:`~forecastforge.schemas.time_series.TimeSeries` deliberately rejects
NaN so that every downstream consumer can assume clean numbers. Feature
engineering cannot honour that contract: a 20-day rolling mean has no value
for the first 19 rows, and pretending otherwise (forward-filling, dropping
rows silently, or substituting zeros) is exactly how look-ahead bugs and
quietly wrong metrics get introduced.

:class:`FeatureSet` therefore sits *between* the validated time series and
the model. It keeps the strict index and target guarantees, allows NaN in
generated feature columns, and makes the warm-up region an explicit,
inspectable number rather than an accident.

Typical use::

    features = FeaturePipeline([Lag(5), RollingStat(20)]).apply(series)
    print(features.warmup)          # rows that are not usable yet
    model_input = features.drop_warmup()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError
from forecastforge.schemas.time_series import TimeSeries, TimeSeriesMetadata
from forecastforge.schemas.validation import validate_frame

__all__ = ["FeatureSet"]


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """A validated time series plus generated feature columns.

    The index and the target column keep the full :class:`TimeSeries`
    guarantees (sorted unique ``DatetimeIndex``, numeric finite non-null
    target). Feature columns must exist and be numeric, but are *allowed* to
    contain NaN in their warm-up region.

    Args:
        frame: Dataset whose index is the shared timeline. Contains the
            target column and every column named in ``features``.
        target: Column to forecast. Must be free of NaN and infinite values.
        features: Generated columns, in creation order. May be empty when a
            pipeline produced nothing.
        metadata: Provenance details carried over from the source series.

    Raises:
        DataValidationError: If the index, target, or feature columns violate
            the contract.
    """

    frame: pd.DataFrame
    target: str
    features: tuple[str, ...]
    metadata: TimeSeriesMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise DataValidationError(
                f"expected a pandas DataFrame, got {type(self.frame).__name__}"
            )

        duplicated = self.frame.columns[self.frame.columns.duplicated()]
        if len(duplicated) > 0:
            names = ", ".join(repr(str(name)) for name in duplicated)
            raise DataValidationError(f"duplicate column name(s): {names}")

        # Reuse the strict contract for the index and the target: it already
        # rejects a non-DatetimeIndex, duplicates, unsorted rows, and a
        # non-numeric or NaN-bearing target.
        if self.target not in self.frame.columns:
            available = ", ".join(repr(str(name)) for name in self.frame.columns[:6])
            raise DataValidationError(
                f"target column {self.target!r} is missing; available columns: {available}"
            )
        validate_frame(self.frame[[self.target]], target=self.target)

        missing = [name for name in self.features if name not in self.frame.columns]
        if missing:
            available = ", ".join(repr(str(name)) for name in self.frame.columns[:6])
            raise DataValidationError(
                f"feature column(s) {missing} are missing; available columns: {available}"
            )

        for name in self.features:
            if not pd.api.types.is_numeric_dtype(self.frame[name]):
                raise DataValidationError(
                    f"feature column {name!r} must be numeric, got dtype {self.frame[name].dtype}"
                )

        object.__setattr__(self, "frame", self.frame.copy())

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_time_series(cls, series: TimeSeries) -> FeatureSet:
        """Wrap a validated :class:`TimeSeries` with no features yet."""
        return cls(
            frame=series.frame,
            target=series.target,
            features=(),
            metadata=series.metadata,
        )

    # -- properties -------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of observations, including the warm-up rows."""
        return len(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The shared, validated timeline."""
        return cast("pd.DatetimeIndex", self.frame.index)

    @property
    def target_series(self) -> pd.Series:
        """The target column as a series."""
        return self.frame[self.target]

    @property
    def matrix(self) -> pd.DataFrame:
        """Only the feature columns, in creation order."""
        return self.frame[list(self.features)]

    @property
    def warmup(self) -> int:
        """Number of leading rows that still contain a NaN feature value.

        This is the count a caller must drop before the rows are usable for
        training. It is a prefix length: if a NaN appears *after* a complete
        row, the data is inconsistent and :attr:`is_complete` will be false
        even though ``warmup`` alone does not describe the problem.
        """
        if not self.features:
            return 0
        incomplete = self.matrix.isna().any(axis=1).to_numpy()
        if not incomplete.any():
            return 0
        first_complete = int(np.argmin(incomplete))
        if incomplete[first_complete]:
            return len(incomplete)
        return first_complete

    @property
    def is_complete(self) -> bool:
        """True when no feature column contains a NaN anywhere."""
        return not bool(self.matrix.isna().to_numpy().any())

    @property
    def n_incomplete_rows(self) -> int:
        """Total rows with at least one NaN feature, warm-up included."""
        if not self.features:
            return 0
        return int(self.matrix.isna().any(axis=1).to_numpy().sum())

    # -- derivation -------------------------------------------------------

    def extend(self, added: pd.DataFrame) -> FeatureSet:
        """Return a new set with ``added`` columns appended.

        Args:
            added: Columns aligned to :attr:`index`. Must not collide with an
                existing column name.

        Returns:
            A new :class:`FeatureSet` whose ``features`` include ``added``.

        Raises:
            DataValidationError: If the index differs or a name collides.
        """
        if not added.index.equals(self.frame.index):
            raise DataValidationError(
                "added columns must share the FeatureSet index; "
                "transform output was not aligned to the input timeline"
            )

        collisions = [name for name in added.columns if name in self.frame.columns]
        if collisions:
            names = ", ".join(repr(str(name)) for name in collisions)
            raise DataValidationError(
                f"transform would overwrite existing column(s) {names}; "
                "give the transform a distinct column name"
            )

        frame = pd.concat([self.frame, added], axis=1)
        return FeatureSet(
            frame=frame,
            target=self.target,
            features=(*self.features, *(str(name) for name in added.columns)),
            metadata=self.metadata,
        )

    def drop_warmup(self) -> FeatureSet:
        """Return a new set with the leading incomplete rows removed."""
        warmup = self.warmup
        if warmup == 0:
            return self
        return FeatureSet(
            frame=self.frame.iloc[warmup:],
            target=self.target,
            features=self.features,
            metadata=self.metadata,
        )

    def dropna(self) -> FeatureSet:
        """Return a new set with every row containing a NaN feature removed.

        Prefer :meth:`drop_warmup` when the NaNs are a prefix: it keeps the
        timeline contiguous, which matters for backtesting windows.
        """
        if not self.features:
            return self
        keep = ~self.matrix.isna().any(axis=1).to_numpy()
        return FeatureSet(
            frame=self.frame.loc[keep],
            target=self.target,
            features=self.features,
            metadata=self.metadata,
        )

    def to_time_series(self) -> TimeSeries:
        """Convert back to a NaN-free :class:`TimeSeries`.

        Raises:
            DataValidationError: If any feature value is still NaN, since a
                ``TimeSeries`` may not contain NaN.
        """
        return TimeSeries(
            frame=self.frame,
            target=self.target,
            metadata=self.metadata,
            covariates=self.features,
        )
