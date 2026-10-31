"""Tests for the optional statsmodels-backed forecasters.

The module skips wholesale when statsmodels is absent, so a core install still
runs a green suite. CI installs the ``models`` extra, so these tests really
execute there rather than being skipped forever.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.models.statistical import ARIMAForecaster, ETSForecaster

pytest.importorskip("statsmodels", reason="the 'models' extra is not installed")


def _close(length: int = 400) -> pd.Series:
    """Return a deterministic close series long enough to fit either model."""
    series = SyntheticSource(config=SyntheticConfig(seed=17, length=length)).fetch(symbol="SPY")
    return series.target_series


# -- construction ----------------------------------------------------------


def test_ets_forecasts_the_requested_horizon() -> None:
    values = ETSForecaster(horizon=4).fit(_close()).predict()
    assert values.shape == (4,)
    assert np.isfinite(values).all()


def test_ets_seasonal_terms_must_be_set_together() -> None:
    with pytest.raises(DataValidationError, match="must be set together"):
        ETSForecaster(seasonal="add")
    with pytest.raises(DataValidationError, match="must be set together"):
        ETSForecaster(seasonal_periods=12)


def test_ets_rejects_a_non_positive_seasonal_period() -> None:
    with pytest.raises(DataValidationError, match="seasonal_periods must be positive"):
        ETSForecaster(seasonal="add", seasonal_periods=0)


def test_ets_with_a_seasonal_component_fits() -> None:
    model = ETSForecaster(trend=None, seasonal="add", seasonal_periods=12, horizon=2)
    values = model.fit(_close()).predict()
    assert values.shape == (2,)
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    ("order", "message"),
    [
        ((1, 1), "order must have 3 terms"),
        ((1, 1, 1, 1), "order must have 3 terms"),
    ],
    ids=["too-short", "too-long"],
)
def test_arima_rejects_a_malformed_order(order: tuple[int, ...], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        ARIMAForecaster(order=order)  # type: ignore[arg-type]


def test_arima_rejects_a_negative_order_term() -> None:
    with pytest.raises(DataValidationError, match="non-negative integers"):
        ARIMAForecaster(order=(-1, 1, 1))


def test_arima_rejects_a_malformed_seasonal_order() -> None:
    with pytest.raises(DataValidationError, match="seasonal_order must have 4 terms"):
        ARIMAForecaster(seasonal_order=(1, 0, 0))


def test_arima_forecasts_the_requested_horizon() -> None:
    values = ARIMAForecaster(order=(1, 1, 0), horizon=3).fit(_close()).predict()
    assert values.shape == (3,)
    assert np.isfinite(values).all()


# -- the optional-dependency guard -----------------------------------------


@pytest.mark.parametrize(
    ("attribute", "factory"),
    [
        ("_HOLTWINTERS", lambda: ETSForecaster()),
        ("_ARIMA", lambda: ARIMAForecaster()),
    ],
    ids=["ets", "arima"],
)
def test_a_missing_dependency_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    factory: object,
) -> None:
    # Simulate a core install: the module-level handle is None, so constructing
    # the model has to fail with the actionable message rather than an obscure
    # AttributeError on a NoneType.
    monkeypatch.setattr(f"forecastforge.models.statistical.{attribute}", None)
    with pytest.raises(DataValidationError, match=r"pip install -e '\.\[models\]'"):
        factory()  # type: ignore[operator]


# -- behaviour -------------------------------------------------------------


def test_ets_is_deterministic() -> None:
    first = ETSForecaster(horizon=3).fit(_close()).predict()
    second = ETSForecaster(horizon=3).fit(_close()).predict()
    np.testing.assert_array_equal(first, second)


def test_arima_is_deterministic() -> None:
    first = ARIMAForecaster(horizon=3).fit(_close()).predict()
    second = ARIMAForecaster(horizon=3).fit(_close()).predict()
    np.testing.assert_array_equal(first, second)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="called before fit"):
        ETSForecaster().predict()


def test_clone_carries_the_configuration_but_not_the_fit() -> None:
    model = ETSForecaster(trend=None, horizon=2).fit(_close())
    clone = model.clone()
    assert clone.horizon == 2
    assert clone.trend is None
    with pytest.raises(RuntimeError, match="called before fit"):
        clone.predict()


def test_repr_names_the_structure() -> None:
    assert "trend=None" in repr(ETSForecaster(trend=None, horizon=2))
    assert "order=(1, 1, 0)" in repr(ARIMAForecaster(order=(1, 1, 0)))
