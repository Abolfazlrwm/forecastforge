"""Honest baselines that every model must beat.

These exist to be a hard floor. On near-random-walk price series a naive
forecast is genuinely hard to beat, and a project that hides that fact is
lying to its users. Every benchmark reports them alongside anything fancier.

All four are cheap, deterministic, and need no optional dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError
from forecastforge.models.base import Forecaster

__all__ = ["Drift", "MovingAverage", "Naive", "SeasonalNaive"]

# A slope needs two points: with one observation the denominator is zero.
_MIN_SLOPE_POINTS = 2


class Naive(Forecaster):
    """Forecast the last observed value for every horizon step.

    This is the "no change" model and the reference every other model is
    measured against. MASE is scaled by its one-step in-sample error, so a
    MASE above 1 means the model is worse than doing nothing.
    """

    name = "naive"

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Record the last observed value."""
        self._last_value = float(series.iloc[-1])

    def _predict(self) -> np.ndarray:
        """Return a flat forecast."""
        return np.full(self.horizon, self._last_value, dtype="float64")


class SeasonalNaive(Forecaster):
    """Repeat the value observed one season ago.

    Args:
        season: Period length in observations — 5 for a weekly cycle in daily
            business data, 252 for an annual cycle.

    Raises:
        DataValidationError: If ``season`` is not positive, or the history is
            shorter than one full season.
    """

    name = "seasonal_naive"

    def __init__(self, *, season: int = 5, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        if season <= 0:
            raise DataValidationError(f"season must be positive, got {season}")
        self.season = season

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Store the history, which must span at least one season."""
        if len(series) < self.season:
            raise DataValidationError(
                f"SeasonalNaive(season={self.season}) needs at least "
                f"{self.season} observations, got {len(series)}"
            )
        self._values = series.to_numpy(dtype="float64")

    def _predict(self) -> np.ndarray:
        """Walk forward one season at a time.

        Beyond the first season the model repeats its own forecasts, which is
        what "seasonal naive" means once there is no observed value left to
        copy.
        """
        values = list(self._values)
        start = len(values)
        for step in range(self.horizon):
            values.append(values[start + step - self.season])
        return np.asarray(values[start:], dtype="float64")

    def __repr__(self) -> str:
        """Return a representation including the season length."""
        return f"{type(self).__name__}(season={self.season}, horizon={self.horizon})"


class MovingAverage(Forecaster):
    """Forecast the mean of the last ``window`` observations, flat.

    Args:
        window: Number of trailing observations to average.

    Raises:
        DataValidationError: If ``window`` is not positive or the history is
            shorter than ``window``.
    """

    name = "moving_average"

    def __init__(self, *, window: int = 20, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        if window <= 0:
            raise DataValidationError(f"window must be positive, got {window}")
        self.window = window

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Average the trailing window."""
        if len(series) < self.window:
            raise DataValidationError(
                f"MovingAverage(window={self.window}) needs at least "
                f"{self.window} observations, got {len(series)}"
            )
        self._mean = float(series.iloc[-self.window :].mean())

    def _predict(self) -> np.ndarray:
        """Return a flat forecast at the trailing mean."""
        return np.full(self.horizon, self._mean, dtype="float64")

    def __repr__(self) -> str:
        """Return a representation including the window."""
        return f"{type(self).__name__}(window={self.window}, horizon={self.horizon})"


class Drift(Forecaster):
    """Extrapolate the average change per period from the first to the last value.

    Args:
        window: When set, compute the slope over the trailing ``window``
            observations instead of the whole history.

    Raises:
        DataValidationError: If fewer than two observations are available
            after applying ``window``.
    """

    name = "drift"

    def __init__(self, *, window: int | None = None, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        if window is not None and window <= 1:
            raise DataValidationError(f"window must be at least 2, got {window}")
        self.window = window

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Estimate the per-period slope."""
        values = series.to_numpy(dtype="float64")
        if self.window is not None:
            if len(values) < self.window:
                raise DataValidationError(
                    f"Drift(window={self.window}) needs at least "
                    f"{self.window} observations, got {len(values)}"
                )
            values = values[-self.window :]
        if len(values) < _MIN_SLOPE_POINTS:
            raise DataValidationError(
                f"Drift needs at least {_MIN_SLOPE_POINTS} observations to estimate "
                f"a slope, got {len(values)}"
            )
        self._last_value = float(values[-1])
        self._slope = float((values[-1] - values[0]) / (len(values) - 1))

    def _predict(self) -> np.ndarray:
        """Add ``slope * k`` to the last value for each step ``k``."""
        steps = np.arange(1, self.horizon + 1, dtype="float64")
        return self._last_value + self._slope * steps

    def __repr__(self) -> str:
        """Return a representation including the optional window."""
        return f"{type(self).__name__}(window={self.window}, horizon={self.horizon})"
