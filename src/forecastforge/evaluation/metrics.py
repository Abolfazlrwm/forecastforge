"""Forecast accuracy metrics.

Each function is a pure function of arrays, so every number in a report can be
recomputed by hand. The definitions are stated in the docstrings because
forecasting metrics are easy to implement subtly differently — sMAPE in
particular has several incompatible conventions in the literature, and MASE is
meaningless without saying exactly what the scale is.

Scale-dependent metrics (MAE, RMSE) are reported in the units of the target.
sMAPE and directional accuracy are percentages. MASE is unit-free by
construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from forecastforge.core.errors import DataValidationError

__all__ = [
    "MetricSet",
    "directional_accuracy",
    "evaluate",
    "interval_coverage",
    "mae",
    "mase",
    "mase_scale",
    "rmse",
    "smape",
]


@dataclass(frozen=True, slots=True)
class MetricSet:
    """An immutable bundle of the standard accuracy metrics.

    Attributes:
        mae: Mean absolute error, in target units.
        rmse: Root mean squared error, in target units.
        smape: Symmetric mean absolute percentage error, in percent.
        mase: Mean absolute scaled error; 1.0 means "as good as the naive
            one-step forecast on the training window".
        directional_accuracy: Share of correctly signed changes, in percent.
        n_observations: Number of scored predictions.
    """

    mae: float
    rmse: float
    smape: float
    mase: float
    directional_accuracy: float
    n_observations: int

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain mapping, without the sample count."""
        values = asdict(self)
        values.pop("n_observations")
        return values


def _as_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Convert input to a 1-D float array, rejecting empties.

    Args:
        values: Array-like of numbers.
        name: Parameter name, used in the error message.

    Returns:
        A 1-D ``float64`` array.

    Raises:
        DataValidationError: If the input is empty.
    """
    array = np.asarray(values, dtype="float64").reshape(-1)
    if array.size == 0:
        raise DataValidationError(f"{name} is empty: at least one value is required")
    return array


def _paired(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``y_true`` and ``y_pred`` as equal-length float arrays.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Raises:
        DataValidationError: If the arrays are empty or differ in length.
    """
    actual = _as_array(y_true, name="y_true")
    predicted = _as_array(y_pred, name="y_pred")
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"y_true and y_pred must have the same length, "
            f"got {actual.shape[0]} and {predicted.shape[0]}"
        )
    return actual, predicted


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Mean absolute error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        ``mean(|y_true - y_pred|)``.
    """
    actual, predicted = _paired(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Root mean squared error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        ``sqrt(mean((y_true - y_pred) ** 2))``.
    """
    actual, predicted = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Uses the ``2 * |e| / (|y| + |y_hat|)`` convention. When both the actual
    and the predicted value are zero the term is defined as 0 — the forecast
    is exactly right, and the ratio is otherwise undefined.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        The mean percentage error. A perfect forecast scores 0; a forecast of
        equal magnitude and opposite sign scores 200, which is the ceiling of
        this convention.
    """
    actual, predicted = _paired(y_true, y_pred)
    numerator = 2.0 * np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denominator == 0.0, 0.0, numerator / denominator)
    return float(np.mean(terms) * 100.0)


def mase_scale(y_train: Sequence[float] | np.ndarray, *, seasonal_period: int = 1) -> float:
    """Return the in-sample naive error used to scale MASE.

    This is ``mean(|y[t] - y[t - m]|)`` over the training window, where ``m``
    is the seasonal period. It is the denominator of MASE and the reason MASE
    is comparable across series of different scale.

    Args:
        y_train: The training observations the model saw.
        seasonal_period: The naive forecast's lag; 1 for a random-walk naive.

    Returns:
        The mean absolute naive error.

    Raises:
        DataValidationError: If ``seasonal_period`` is not positive, the
            training window is too short, or the series is constant (a zero
            scale makes MASE undefined).
    """
    if seasonal_period <= 0:
        raise DataValidationError(f"seasonal_period must be positive, got {seasonal_period}")
    values = _as_array(y_train, name="y_train")
    if values.shape[0] <= seasonal_period:
        raise DataValidationError(
            f"y_train needs more than seasonal_period={seasonal_period} observations "
            f"to compute the naive scale, got {values.shape[0]}"
        )
    errors = np.abs(values[seasonal_period:] - values[:-seasonal_period])
    scale = float(np.mean(errors))
    if scale == 0.0:
        raise DataValidationError(
            "the training series is constant at this seasonal period, so the "
            "naive error is zero and MASE is undefined"
        )
    return scale


def mase(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    *,
    seasonal_period: int = 1,
) -> float:
    """Mean absolute scaled error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        y_train: The training observations, used only to derive the scale.
        seasonal_period: The naive forecast's lag used for the scale.

    Returns:
        ``MAE / mase_scale(y_train)``. Below 1 beats the naive forecast;
        above 1 is worse than doing nothing.
    """
    scale = mase_scale(y_train, seasonal_period=seasonal_period)
    return mae(y_true, y_pred) / scale


def directional_accuracy(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    previous: Sequence[float] | np.ndarray,
) -> float:
    """Share of predictions that got the direction of change right, in percent.

    The direction of a prediction is the sign of ``y_pred - previous``; the
    direction of the outcome is the sign of ``y_true - previous``. A forecast
    that predicts no change is credited only when the outcome also did not
    change.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        previous: The actual value immediately before each predicted point.

    Returns:
        The percentage of correctly signed changes.

    Raises:
        DataValidationError: If the arrays differ in length.
    """
    actual, predicted = _paired(y_true, y_pred)
    reference = _as_array(previous, name="previous")
    if reference.shape != actual.shape:
        raise DataValidationError(
            f"previous must have the same length as y_true, "
            f"got {reference.shape[0]} and {actual.shape[0]}"
        )
    actual_direction = np.sign(actual - reference)
    predicted_direction = np.sign(predicted - reference)
    return float(np.mean(actual_direction == predicted_direction) * 100.0)


def interval_coverage(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """Share of outcomes that fall inside the predicted interval, in percent.

    Only meaningful for models that emit intervals; a model without them
    should not be scored here at all rather than scored as zero.

    Args:
        y_true: Observed values.
        lower: Lower bound of each interval.
        upper: Upper bound of each interval.

    Returns:
        The percentage of ``y_true`` values inside ``[lower, upper]``.

    Raises:
        DataValidationError: If the arrays differ in length, or a lower bound
            exceeds its upper bound.
    """
    actual, low = _paired(y_true, lower)
    high = _as_array(upper, name="upper")
    if high.shape != actual.shape:
        raise DataValidationError(
            f"upper must have the same length as y_true, got {high.shape[0]} and {actual.shape[0]}"
        )
    if (low > high).any():
        raise DataValidationError("lower bound exceeds upper bound for at least one interval")
    inside = (actual >= low) & (actual <= high)
    return float(np.mean(inside) * 100.0)


def evaluate(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    previous: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    seasonal_period: int = 1,
) -> MetricSet:
    """Compute every standard metric in one call.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        previous: Actual value immediately before each predicted point.
        y_train: Training observations, used only for the MASE scale.
        seasonal_period: The naive lag used for the MASE scale.

    Returns:
        A :class:`MetricSet`.
    """
    return MetricSet(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        smape=smape(y_true, y_pred),
        mase=mase(y_true, y_pred, y_train, seasonal_period=seasonal_period),
        directional_accuracy=directional_accuracy(y_true, y_pred, previous),
        n_observations=int(np.asarray(y_true, dtype="float64").size),
    )
