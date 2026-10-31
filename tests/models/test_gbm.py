"""Tests for the optional LightGBM forecaster.

The module skips wholesale when lightgbm is absent. The boundary tests use a
stub booster and a spy on the training hook, so the assertions are about which
pairs the model is allowed to learn from — the part that can leak — rather than
about the ensemble's accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.features.builtins import CalendarFeatures, Returns, RollingStat
from forecastforge.features.pipeline import FeaturePipeline
from forecastforge.models.gbm import LightGBMForecaster

pytest.importorskip("lightgbm", reason="the 'models' extra is not installed")


class _StubBooster:
    """Predicts zero and ignores its input.

    A booster that predicts no change makes the model reproduce its last
    observation, which is exactly what is wanted when the test is about the
    training pairs rather than about the ensemble.
    """

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Return a zero change for every row."""
        return np.zeros(matrix.shape[0], dtype="float64")


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series([float(value) for value in values], index=index)


def _features(series: pd.Series, *, warmup: int = 0) -> pd.DataFrame:
    """Return one feature column, NaN for the first ``warmup`` rows."""
    column = np.arange(len(series), dtype="float64")
    column[:warmup] = np.nan
    return pd.DataFrame({"x": column}, index=series.index)


def _pipeline_features() -> tuple[pd.Series, pd.DataFrame]:
    """Return a realistic close series and its feature matrix."""
    time_series = SyntheticSource(config=SyntheticConfig(seed=23, length=400)).fetch(
        symbol="SPY"
    )
    pipeline = FeaturePipeline(
        [
            Returns(periods=1, log=True),
            RollingStat(window=20, stat="std", column="close_logret1"),
            CalendarFeatures(["dayofweek"]),
        ]
    )
    features = pipeline.apply(time_series)
    return time_series.target_series, features.matrix


# -- construction ----------------------------------------------------------


def test_predicts_the_horizon_from_the_last_observation() -> None:
    close, matrix = _pipeline_features()
    values = LightGBMForecaster(horizon=3).fit(close, matrix).predict()
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    # The model learns a change, so its forecast stays anchored to the last
    # observation instead of drifting to the middle of the historical range.
    last = float(close.iloc[-1])
    assert np.all(np.abs(values / last - 1.0) < 0.05)


def test_a_flat_series_is_forecast_flat() -> None:
    series = _series([100.0] * 60)
    model = LightGBMForecaster(horizon=2, num_boost_round=20)
    values = model.fit(series, _features(series)).predict()
    # Every training change is zero, so every booster predicts no change.
    np.testing.assert_allclose(values, [100.0, 100.0], rtol=1e-9)


def test_a_missing_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forecastforge.models.gbm._LIGHTGBM", None)
    with pytest.raises(DataValidationError, match=r"pip install -e '\.\[models\]'"):
        LightGBMForecaster()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_boost_round": 0}, "num_boost_round must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"num_leaves": 1}, "num_leaves must be at least 2"),
        ({"min_data_in_leaf": 0}, "min_data_in_leaf must be positive"),
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        LightGBMForecaster(**kwargs)  # type: ignore[arg-type]


# -- feature handling ------------------------------------------------------


def test_fitting_without_features_is_rejected() -> None:
    series = _series([1.0] * 30)
    with pytest.raises(DataValidationError, match="requires a feature frame"):
        LightGBMForecaster().fit(series)


def test_an_empty_feature_frame_is_rejected() -> None:
    series = _series([1.0] * 30)
    empty = pd.DataFrame(index=series.index)
    with pytest.raises(DataValidationError, match="no columns to train on"):
        LightGBMForecaster().fit(series, empty)


def test_non_numeric_features_are_rejected() -> None:
    series = _series([1.0] * 30)
    features = _features(series)
    features["label"] = "abc"
    with pytest.raises(DataValidationError, match="must be numeric"):
        LightGBMForecaster().fit(series, features)


def test_an_incomplete_final_row_is_rejected() -> None:
    series = _series([1.0] * 30)
    features = _features(series)
    features.iloc[-1, 0] = np.nan
    with pytest.raises(DataValidationError, match="last history row has NaN features"):
        LightGBMForecaster().fit(series, features)


def test_too_few_complete_rows_is_rejected() -> None:
    series = _series([1.0] * 12)
    features = _features(series, warmup=11)
    with pytest.raises(DataValidationError, match="complete training pair"):
        LightGBMForecaster().fit(series, features)


# -- the training-pair boundary -------------------------------------------


def test_training_pairs_stop_at_the_end_of_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pair ``(t, t + h)`` exists only when ``t + h`` is inside the history.

    This is the boundary that leaks if it is off by one: the spy records the
    exact change vector each booster was trained on, so both the number of
    pairs and their offsets are asserted rather than inferred.
    """
    series = _series([float(value) for value in range(1, 41)])
    features = _features(series, warmup=5)
    model = LightGBMForecaster(horizon=3)
    recorded: list[np.ndarray] = []

    def _spy(matrix: np.ndarray, target: np.ndarray) -> _StubBooster:
        recorded.append(target.copy())
        return _StubBooster()

    monkeypatch.setattr(model, "_train_one", _spy)
    model.fit(series, features)

    # 40 rows, 5 warm-up rows, horizon h -> 40 - h - 5 legal pairs.
    assert [target.shape[0] for target in recorded] == [34, 33, 32]
    # The series is 1, 2, 3, ..., so an h-step change is exactly h.
    for step, target in enumerate(recorded, start=1):
        np.testing.assert_allclose(target, np.full(target.shape[0], float(step)))
    # The last pair for the longest horizon ends on the final observation.
    assert recorded[-1][-1] == pytest.approx(float(series.iloc[-1] - series.iloc[-4]))


def test_the_stub_booster_reproduces_the_last_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = _series([float(value) for value in range(1, 41)])
    features = _features(series, warmup=5)
    model = LightGBMForecaster(horizon=2)
    monkeypatch.setattr(model, "_train_one", lambda matrix, target: _StubBooster())
    values = model.fit(series, features).predict()
    np.testing.assert_allclose(values, [40.0, 40.0])


# -- behaviour -------------------------------------------------------------


def test_fitting_twice_is_deterministic() -> None:
    close, matrix = _pipeline_features()
    first = LightGBMForecaster(horizon=2).fit(close, matrix).predict()
    second = LightGBMForecaster(horizon=2).fit(close, matrix).predict()
    np.testing.assert_array_equal(first, second)


def test_clone_carries_the_configuration_but_not_the_fit() -> None:
    close, matrix = _pipeline_features()
    model = LightGBMForecaster(horizon=2, num_leaves=8).fit(close, matrix)
    clone = model.clone()
    assert clone.horizon == 2
    assert clone.num_leaves == 8
    with pytest.raises(RuntimeError, match="called before fit"):
        clone.predict()


def test_repr_names_the_boosting_configuration() -> None:
    text = repr(LightGBMForecaster(horizon=2, num_boost_round=50, learning_rate=0.1))
    assert "num_boost_round=50" in text
    assert "learning_rate=0.1" in text
