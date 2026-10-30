"""Tests for the Forecaster base contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.models import Forecaster, Naive


class _FixedValue(Forecaster):
    """Predicts a constant, so tests can isolate the base-class behaviour."""

    name = "fixed"

    def __init__(self, *, value: float = 1.0, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        self.value = value

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        self._n = len(series)

    def _predict(self) -> np.ndarray:
        return np.full(self.horizon, self.value)


class _WrongLength(Forecaster):
    """Returns two values regardless of the declared horizon."""

    name = "wrong-length"

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        return None

    def _predict(self) -> np.ndarray:
        return np.array([1.0, 2.0])


class _NeedsFeatures(Forecaster):
    """Declares a hard dependency on an exogenous frame."""

    name = "needs-features"
    requires_features = True

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        return None

    def _predict(self) -> np.ndarray:
        return np.zeros(self.horizon)


def test_fit_returns_self(synthetic_close: pd.Series) -> None:
    model = Naive()
    assert model.fit(synthetic_close) is model


def test_predict_before_fit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        Naive().predict()


def test_predict_returns_the_declared_horizon(synthetic_close: pd.Series) -> None:
    values = Naive(horizon=4).fit(synthetic_close).predict()
    assert values.shape == (4,)
    assert values.dtype == np.float64


def test_wrong_output_length_is_rejected(synthetic_close: pd.Series) -> None:
    with pytest.raises(DataValidationError, match="produced 2 value"):
        _WrongLength(horizon=1).fit(synthetic_close).predict()


@pytest.mark.parametrize("horizon", [0, -3])
def test_non_positive_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(DataValidationError, match="horizon must be positive"):
        Naive(horizon=horizon)


def test_empty_history_is_rejected() -> None:
    empty = pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(DataValidationError, match="empty"):
        Naive().fit(empty)


def test_nan_history_is_rejected() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([1.0, np.nan, 3.0], index=index)
    with pytest.raises(DataValidationError, match="NaN"):
        Naive().fit(series)


def test_non_datetime_index_is_rejected() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        Naive().fit(series)


def test_unsorted_history_is_rejected() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")[::-1]
    series = pd.Series([1.0, 2.0, 3.0], index=index)
    with pytest.raises(DataValidationError, match="sorted"):
        Naive().fit(series)


def test_missing_features_are_rejected(synthetic_close: pd.Series) -> None:
    with pytest.raises(DataValidationError, match="requires a feature frame"):
        _NeedsFeatures().fit(synthetic_close)


def test_misaligned_features_are_rejected(synthetic_close: pd.Series) -> None:
    features = pd.DataFrame({"x": np.arange(len(synthetic_close))}, index=synthetic_close.index)
    features = features.iloc[:-1]
    with pytest.raises(DataValidationError, match="share the history index"):
        Naive().fit(synthetic_close, features)


def test_clone_is_unfitted_and_independent(synthetic_close: pd.Series) -> None:
    original = Naive().fit(synthetic_close)
    duplicate = original.clone()
    assert duplicate is not original
    assert not duplicate._fitted  # noqa: SLF001 - asserting the contract directly
    with pytest.raises(RuntimeError, match="before fit"):
        duplicate.predict()


def test_intervals_are_unavailable_by_default() -> None:
    with pytest.raises(NotImplementedError, match="does not provide"):
        Naive().predict_interval()


def test_repr_names_the_class_and_horizon() -> None:
    assert repr(Naive(horizon=3)) == "Naive(horizon=3)"
