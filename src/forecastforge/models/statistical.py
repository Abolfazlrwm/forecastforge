"""Optional statistical forecasters backed by statsmodels.

ETS (exponential smoothing) and ARIMA are the two classical models a
practitioner reaches for before a learned one, so ForecastForge ships them
behind the ``models`` extra rather than in the core install.

The module is importable without statsmodels: the dependency is resolved
through :func:`~forecastforge.config._optional.optional_import`, and the
error naming the extra is raised when a model is constructed, not at import
time. That keeps ``import forecastforge.models`` cheap and lets the core
package install with no scientific stack beyond NumPy and pandas.
"""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

from forecastforge.config._optional import optional_import
from forecastforge.core.errors import DataValidationError
from forecastforge.models.base import Forecaster

__all__ = ["ARIMAForecaster", "ETSForecaster", "statsmodels_available"]

_HOLTWINTERS = optional_import("statsmodels.tsa.holtwinters")
_ARIMA = optional_import("statsmodels.tsa.arima.model")

_MIN_ORDER_TERMS = 3
_SEASONAL_ORDER_TERMS = 4
_INSTALL_HINT = "Install it with: pip install -e '.[models]'"


def statsmodels_available() -> bool:
    """Return whether the models in this module can be constructed.

    Callers that build a list of candidate models use this to skip the ones
    that would raise, instead of catching the construction error.
    """
    return _HOLTWINTERS is not None and _ARIMA is not None


def _require(module: ModuleType | None, *, model: str, dependency: str) -> Any:
    """Return an optional module, or raise naming the extra to install.

    Args:
        module: The result of :func:`optional_import`.
        model: Model name, used in the error message.
        dependency: The missing distribution, used in the error message.

    Returns:
        The module, typed as ``Any`` because its contents are untyped here.

    Raises:
        DataValidationError: If ``module`` is ``None``.
    """
    if module is None:
        raise DataValidationError(
            f"{model} requires the optional {dependency!r} dependency. {_INSTALL_HINT}"
        )
    return cast("Any", module)


def _validate_order(order: tuple[int, ...], *, terms: int, name: str) -> None:
    """Reject an order tuple that is not ``terms`` non-negative integers.

    Args:
        order: The candidate order tuple.
        terms: Required length of the tuple.
        name: Parameter name, used in the error message.

    Raises:
        DataValidationError: If the length or a term is invalid.
    """
    if len(order) != terms:
        raise DataValidationError(f"{name} must have {terms} terms, got {len(order)}")
    if any(not isinstance(term, int) or term < 0 for term in order):
        raise DataValidationError(f"{name} terms must be non-negative integers, got {order}")


class ETSForecaster(Forecaster):
    """Exponential smoothing (ETS) via statsmodels' Holt-Winters.

    Args:
        trend: ``"add"``, ``"mul"``, or ``None`` for no trend component.
        seasonal: ``"add"``, ``"mul"``, or ``None`` for no seasonal component.
        seasonal_periods: Length of the seasonal cycle. Must be set together
            with ``seasonal``.
        horizon: Number of future periods to predict.

    Raises:
        DataValidationError: If statsmodels is not installed, ``seasonal`` and
            ``seasonal_periods`` are not both set or both unset, or
            ``horizon`` is not positive.

    Note:
        statsmodels emits convergence warnings for some series. They are
        silenced during the fit because a point-forecast baseline should not
        fail a test suite on an optimizer message; a genuinely bad fit still
        shows up in the metrics.
    """

    name = "ets"

    def __init__(
        self,
        *,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        horizon: int = 1,
    ) -> None:
        super().__init__(horizon=horizon)
        _require(_HOLTWINTERS, model=type(self).__name__, dependency="statsmodels")
        if (seasonal is None) != (seasonal_periods is None):
            raise DataValidationError(
                "seasonal and seasonal_periods must be set together; "
                f"got seasonal={seasonal!r}, seasonal_periods={seasonal_periods!r}"
            )
        if seasonal_periods is not None and seasonal_periods <= 0:
            raise DataValidationError(f"seasonal_periods must be positive, got {seasonal_periods}")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._result: Any = None

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Fit the Holt-Winters model."""
        holtwinters = _require(
            _HOLTWINTERS, model=type(self).__name__, dependency="statsmodels"
        )
        model = holtwinters.ExponentialSmoothing(
            series.to_numpy(dtype="float64"),
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit()

    def _predict(self) -> np.ndarray:
        """Forecast ``horizon`` steps ahead."""
        return np.asarray(self._result.forecast(steps=self.horizon), dtype="float64")

    def __repr__(self) -> str:
        """Return a representation including the model structure."""
        return (
            f"{type(self).__name__}(trend={self.trend!r}, seasonal={self.seasonal!r}, "
            f"seasonal_periods={self.seasonal_periods}, horizon={self.horizon})"
        )


class ARIMAForecaster(Forecaster):
    """ARIMA, with optional seasonal terms, via statsmodels.

    Args:
        order: ``(p, d, q)`` — autoregressive, differencing, and
            moving-average orders.
        seasonal_order: ``(P, D, Q, s)`` — seasonal orders and the seasonal
            period. All zeros disables the seasonal component.
        horizon: Number of future periods to predict.

    Raises:
        DataValidationError: If statsmodels is not installed, an order is not
            a non-negative integer, or ``horizon`` is not positive.
    """

    name = "arima"

    def __init__(
        self,
        *,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        horizon: int = 1,
    ) -> None:
        super().__init__(horizon=horizon)
        _require(_ARIMA, model=type(self).__name__, dependency="statsmodels")
        _validate_order(order, terms=_MIN_ORDER_TERMS, name="order")
        _validate_order(seasonal_order, terms=_SEASONAL_ORDER_TERMS, name="seasonal_order")
        self.order = order
        self.seasonal_order = seasonal_order
        self._result: Any = None

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Fit the ARIMA model."""
        arima = _require(_ARIMA, model=type(self).__name__, dependency="statsmodels")
        model = arima.ARIMA(
            series.to_numpy(dtype="float64"),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit()

    def _predict(self) -> np.ndarray:
        """Forecast ``horizon`` steps ahead."""
        return np.asarray(self._result.forecast(steps=self.horizon), dtype="float64")

    def __repr__(self) -> str:
        """Return a representation including the model orders."""
        return (
            f"{type(self).__name__}(order={self.order}, "
            f"seasonal_order={self.seasonal_order}, horizon={self.horizon})"
        )
