"""Optional gradient-boosting forecaster.

A tree ensemble cannot extrapolate a price level — it can only return values it
has already seen. Forecasting raw closes with one is therefore a category
error, which is why this model consumes :class:`~forecastforge.features.frame.FeatureSet`
columns instead of the target itself: returns, volatility, oscillators, and
calendar terms are stationary enough for a tree to say something useful about
them.

The model is *direct*: one booster per horizon step, each trained on
``(features_t, y_{t+h} - y_t)`` pairs. The target is the **change**, not the
level. A tree can only return values inside the range it was trained on, so
regressing on the price level produces a forecast that is permanently pinned
to the middle of the historical range and drifts further from reality as the
series trends — measured on a synthetic path it was off by 15% after one
horizon. Regressing on the change removes that bias by construction: a booster
that learns nothing predicts zero change, which is exactly the naive forecast.

That construction is also where look-ahead bugs live, so the pairing loop stops
at ``t + h <= len(history) - 1`` — a row whose target would fall outside the
visible history is never used. The walk-forward backtester hands this model a
history that ends at the fold cutoff, so the cutoff is respected transitively.

LightGBM is driven through its native :func:`lightgbm.train` API rather than
the scikit-learn wrapper: the wrapper drags in scikit-learn for an estimator
protocol this class does not implement, and the native parameters are the ones
LightGBM's own documentation describes.

The module imports without LightGBM installed; the error naming the extra is
raised when the model is constructed.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from forecastforge.config._optional import optional_import
from forecastforge.core.errors import DataValidationError
from forecastforge.models.base import Forecaster

__all__ = ["LightGBMForecaster", "lightgbm_available"]

_LIGHTGBM = optional_import("lightgbm")

# A direct multi-horizon model fits one booster per step, so it needs enough
# (features_t, y_{t+h}) pairs for a split to mean anything.
_MIN_TRAINING_PAIRS = 10
_INSTALL_HINT = "Install it with: pip install -e '.[models]'"


def lightgbm_available() -> bool:
    """Return whether :class:`LightGBMForecaster` can be constructed.

    Callers that build a list of candidate models use this to skip the ones
    that would raise, instead of catching the construction error.
    """
    return _LIGHTGBM is not None


def _require_lightgbm() -> Any:
    """Return the lightgbm module, or raise naming the extra to install.

    Returns:
        The module, typed as ``Any`` because lightgbm ships no type stubs.

    Raises:
        DataValidationError: If lightgbm is not installed.
    """
    if _LIGHTGBM is None:
        raise DataValidationError(
            f"LightGBMForecaster requires the optional 'lightgbm' dependency. {_INSTALL_HINT}"
        )
    return cast("Any", _LIGHTGBM)


class LightGBMForecaster(Forecaster):
    """Direct multi-horizon gradient boosting over precomputed features.

    Args:
        horizon: Number of future periods to predict. One booster is fitted
            per step.
        num_boost_round: Boosting rounds per step.
        learning_rate: Shrinkage applied to each round.
        num_leaves: Maximum leaves per tree.
        min_data_in_leaf: Minimum rows a leaf may hold. Keeping this well
            above one is what stops the ensemble from memorising a fold.
        random_state: Seed, so a run is reproducible.
        extra_params: Additional LightGBM parameters forwarded verbatim to
            :func:`lightgbm.train`. Use this to tune without subclassing.

    Raises:
        DataValidationError: If lightgbm is not installed, ``horizon`` is not
            positive, or a hyperparameter is out of range.

    Note:
        :attr:`requires_features` is true: pass the output of a
        :class:`~forecastforge.features.pipeline.FeaturePipeline` (its
        :attr:`~forecastforge.features.frame.FeatureSet.matrix`) as ``features``.
        Warm-up rows may contain NaN; they are dropped while building the
        training pairs, because a row with a missing feature cannot teach the
        model anything. The final row of the history must be complete, since
        it is the row the forecast is made from.
    """

    name = "lightgbm"
    requires_features: ClassVar[bool] = True

    def __init__(
        self,
        *,
        horizon: int = 1,
        num_boost_round: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        random_state: int = 0,
        extra_params: dict[str, object] | None = None,
    ) -> None:
        super().__init__(horizon=horizon)
        _require_lightgbm()
        if num_boost_round <= 0:
            raise DataValidationError(f"num_boost_round must be positive, got {num_boost_round}")
        if learning_rate <= 0:
            raise DataValidationError(f"learning_rate must be positive, got {learning_rate}")
        if num_leaves <= 1:
            raise DataValidationError(f"num_leaves must be at least 2, got {num_leaves}")
        if min_data_in_leaf <= 0:
            raise DataValidationError(
                f"min_data_in_leaf must be positive, got {min_data_in_leaf}"
            )
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state = random_state
        self.extra_params = dict(extra_params or {})
        self._boosters: list[Any] = []
        self._prediction_row: np.ndarray | None = None
        self._last_value: float | None = None

    # -- Forecaster hooks -------------------------------------------------

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Fit one booster per horizon step.

        Args:
            series: The visible history, ending at the forecast origin.
            features: Feature columns aligned to ``series.index``.

        Raises:
            DataValidationError: If the features are missing, non-numeric, or
                leave too few complete rows to train on.
        """
        if features is None:
            raise DataValidationError(
                "LightGBMForecaster requires a feature frame, but features=None"
            )
        matrix = self._as_matrix(features)

        values = series.to_numpy(dtype="float64")
        n_rows = values.size
        complete = ~np.isnan(matrix).any(axis=1)

        # The row the forecast starts from has to be usable, otherwise there is
        # nothing to hand the boosters at predict time.
        if not bool(complete[-1]):
            missing = [
                str(name)
                for name, bad in zip(features.columns, np.isnan(matrix[-1]), strict=True)
                if bad
            ]
            raise DataValidationError(
                "the last history row has NaN features, so no forecast can be made "
                f"from it; incomplete column(s): {', '.join(missing)}"
            )
        self._prediction_row = matrix[-1]
        self._last_value = float(values[-1])

        self._boosters = []
        for step in range(1, self.horizon + 1):
            # Legal pairs only: the target at t + step must be inside the
            # visible history. This is the boundary the backtester relies on.
            last_origin = n_rows - 1 - step
            origins = np.arange(0, last_origin + 1)
            origins = origins[complete[origins]]
            if origins.size < _MIN_TRAINING_PAIRS:
                raise DataValidationError(
                    f"step {step} has {origins.size} complete training pair(s), "
                    f"but at least {_MIN_TRAINING_PAIRS} are required; "
                    "supply a longer history or a shorter warm-up"
                )
            # The booster learns the change, never the level: see the module
            # docstring for why a tree cannot be trusted with a price level.
            change = values[origins + step] - values[origins]
            self._boosters.append(self._train_one(matrix[origins], change))

    def _predict(self) -> np.ndarray:
        """Forecast each horizon step from the final history row."""
        row = self._prediction_row
        last = self._last_value
        if row is None or last is None:  # pragma: no cover - guarded by _fit
            raise RuntimeError("LightGBMForecaster.predict() called before fit()")
        batch = row.reshape(1, -1)
        changes = np.array(
            [float(booster.predict(batch)[0]) for booster in self._boosters],
            dtype="float64",
        )
        return last + changes

    # -- internals --------------------------------------------------------

    def _as_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Return the feature frame as a float matrix.

        Args:
            features: Candidate feature columns.

        Returns:
            A ``(n_rows, n_columns)`` float array.

        Raises:
            DataValidationError: If there are no columns or one is not numeric.
        """
        if features.shape[1] == 0:
            raise DataValidationError("the feature frame has no columns to train on")
        non_numeric = [
            str(name)
            for name in features.columns
            if not pd.api.types.is_numeric_dtype(features[name])
        ]
        if non_numeric:
            raise DataValidationError(
                f"feature column(s) {', '.join(non_numeric)} must be numeric"
            )
        return features.to_numpy(dtype="float64")

    def _train_one(self, matrix: np.ndarray, target: np.ndarray) -> Any:
        """Fit a single booster on one horizon step.

        Args:
            matrix: Training features, already restricted to legal pairs.
            target: The matching outcomes.

        Returns:
            A fitted LightGBM booster.
        """
        lightgbm = _require_lightgbm()
        params: dict[str, object] = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "seed": self.random_state,
            # Bit-identical folds are the whole point of a reproducible
            # benchmark, so determinism is pinned and threading is disabled:
            # a fixed seed alone does not guarantee the same summation order.
            "deterministic": True,
            "num_threads": 1,
            "force_col_wise": True,
            "verbose": -1,
        }
        params.update(self.extra_params)
        dataset = lightgbm.Dataset(matrix, label=target, free_raw_data=True)
        with warnings.catch_warnings():
            # LightGBM reports "no further splits" for small folds through the
            # warnings machinery, and the suite turns warnings into errors. A
            # poor fit still shows up in the metrics.
            warnings.simplefilter("ignore")
            return lightgbm.train(params, dataset, num_boost_round=self.num_boost_round)

    def __repr__(self) -> str:
        """Return a representation including the boosting configuration."""
        return (
            f"{type(self).__name__}(horizon={self.horizon}, "
            f"num_boost_round={self.num_boost_round}, "
            f"learning_rate={self.learning_rate}, num_leaves={self.num_leaves})"
        )
