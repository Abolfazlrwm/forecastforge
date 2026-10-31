"""The forecaster interface.

Every model in ForecastForge — from the naive baseline to a gradient-boosted
one — is fitted on a history and asked for the next ``horizon`` values. The
horizon is fixed at construction rather than passed to :meth:`Forecaster.predict`
because a direct multi-horizon model has to build one estimator per step: it
needs to know ``h`` when it fits, not when it predicts.

The base class implements the parts that must not vary — input validation, the
fitted-before-predict guard, and the output-shape check — so a concrete model
only writes :meth:`_fit` and :meth:`_predict`. This mirrors
:class:`~forecastforge.features.transform.Transform`: the shared contract lives
in one place, so a subclass cannot quietly break it.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError

__all__ = ["Forecaster", "validate_history"]


class Forecaster(ABC):
    """Base class for a fitted, horizon-fixed forecasting model.

    Args:
        horizon: Number of future periods to predict. Must be positive.

    Attributes:
        name: Short identifier used in reports and result frames.
        requires_features: Whether :meth:`fit` needs an exogenous feature
            frame. Models that read it raise if it is missing.
        supports_intervals: Whether :meth:`predict_interval` is available.

    Raises:
        DataValidationError: If ``horizon`` is not positive.
    """

    name: ClassVar[str] = "forecaster"
    requires_features: ClassVar[bool] = False
    supports_intervals: ClassVar[bool] = False

    def __init__(self, *, horizon: int = 1) -> None:
        if horizon <= 0:
            raise DataValidationError(f"horizon must be positive, got {horizon}")
        self.horizon = horizon
        self._fitted = False

    # -- public API -------------------------------------------------------

    def fit(self, series: pd.Series, features: pd.DataFrame | None = None) -> Self:
        """Fit the model on ``series``.

        Args:
            series: The observable history, ending at the forecast origin.
                Must be a non-empty, NaN-free ``Series`` with a sorted
                ``DatetimeIndex``.
            features: Exogenous columns aligned to ``series.index``. Required
                when :attr:`requires_features` is true; ignored otherwise.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            DataValidationError: If the history violates the contract.
        """
        validate_history(series, features, requires_features=type(self).requires_features)
        self._fit(series, features)
        self._fitted = True
        return self

    def predict(self) -> np.ndarray:
        """Return the next :attr:`horizon` values.

        Returns:
            A float array of length :attr:`horizon`.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            DataValidationError: If the model returns the wrong number of
                values.
        """
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.predict() called before fit()")
        values = np.asarray(self._predict(), dtype="float64").reshape(-1)
        if values.shape[0] != self.horizon:
            raise DataValidationError(
                f"{type(self).__name__} produced {values.shape[0]} value(s) "
                f"for horizon {self.horizon}"
            )
        return values

    def predict_interval(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper prediction bounds.

        Only available when :attr:`supports_intervals` is true.

        Args:
            level: Central coverage probability, strictly between 0 and 1.

        Raises:
            NotImplementedError: When the model provides no intervals.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide prediction intervals")

    def clone(self) -> Self:
        """Return an unfitted copy carrying the same configuration.

        The walk-forward backtester needs a fresh model per fold; cloning
        keeps the caller from having to know how to rebuild each one.
        """
        duplicate = copy.deepcopy(self)
        duplicate._fitted = False
        return duplicate

    def __repr__(self) -> str:
        """Return a representation naming the class and horizon."""
        return f"{type(self).__name__}(horizon={self.horizon})"

    # -- subclass hooks ---------------------------------------------------

    @abstractmethod
    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Learn from ``series``. Validated input; ``features`` may be None."""

    @abstractmethod
    def _predict(self) -> np.ndarray:
        """Produce :attr:`horizon` values. Called only after :meth:`_fit`."""


def validate_history(
    series: pd.Series,
    features: pd.DataFrame | None,
    *,
    requires_features: bool,
) -> None:
    """Check the training history and its exogenous frame.

    Args:
        series: The candidate history.
        features: Optional exogenous columns.
        requires_features: Whether the model needs ``features``.

    Raises:
        DataValidationError: If any part of the contract is violated.
    """
    if not isinstance(series, pd.Series):
        raise DataValidationError(f"expected a pandas Series, got {type(series).__name__}")
    if series.empty:
        raise DataValidationError("history is empty: at least one observation is required")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"history index must be a pandas DatetimeIndex, got {type(series.index).__name__}"
        )
    if not series.index.is_monotonic_increasing:
        raise DataValidationError("history index must be sorted in increasing order")
    nulls = series.isna()
    if nulls.any():
        first = series.index[nulls][0]
        raise DataValidationError(
            f"history contains {int(nulls.sum())} NaN value(s), first at {first}; "
            "fill or drop them before fitting"
        )

    if requires_features and features is None:
        raise DataValidationError("this model requires a feature frame, but features=None")
    if features is not None and not features.index.equals(series.index):
        raise DataValidationError(
            "feature frame must share the history index; "
            "the exogenous rows are not aligned to the series"
        )
