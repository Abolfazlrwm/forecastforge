"""Leakage-safe walk-forward backtesting.

A backtest is only as honest as its fold boundaries. The rule this module
enforces is simple and absolute: **the model receives exactly the rows at or
before the fold's cutoff, and nothing else.** Every prediction is scored
against a target that lies strictly after that cutoff.

The ``gap`` parameter adds an embargo, following the convention of
``sklearn.model_selection.TimeSeriesSplit``: with ``gap = g``, the last ``g``
observations before the forecast origin are withheld, so the training window
ends ``g`` periods before the first predicted point. The embargo rows are
neither trained on nor predicted — they are discarded, which is what stops a
model from fitting on data adjacent to the target.

Training rows whose target extends past the cutoff are excluded by
construction: the model is only ever handed a history that ends at the
cutoff, so there is no row it could pair with a future target. A supervised
model that builds ``(features_t, y_{t+h})`` pairs must therefore stop at
``t + h <= len(history) - 1``, which is what
:class:`~forecastforge.models.gbm.LightGBMForecaster` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError
from forecastforge.evaluation.metrics import (
    MetricSet,
    directional_accuracy,
    mae,
    mase_scale,
    rmse,
    smape,
)
from forecastforge.models.base import Forecaster, validate_history

__all__ = ["BacktestConfig", "BacktestResult", "Fold", "WalkForwardBacktester"]

_PREDICTION_COLUMNS = (
    "fold",
    "timestamp",
    "horizon",
    "y_true",
    "y_pred",
    "model",
    "previous",
)

# A training window shorter than this cannot produce a naive error, and a
# rolling window shorter than this is not a window at all.
_MIN_TRAIN_ROWS = 2


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """How to slice the series into walk-forward folds.

    Args:
        horizon: Number of periods predicted from each origin.
        initial_train: Rows available before the first forecast origin. The
            first origin is the last of these rows, so the first training
            window holds ``initial_train - gap`` rows.
        step: Rows between consecutive forecast origins. A step above 1 trades
            resolution for runtime.
        gap: Observations withheld from the end of each training window, so
            the model fits on data ending ``gap`` periods before the origin.
            The embargo rows are skipped entirely: they are neither trained
            on nor predicted.
        window: Rolling training-window length. ``None`` means expanding.
        seasonal_period: Naive lag used for the MASE scale and for the
            seasonal baseline.

    Raises:
        DataValidationError: If any value is out of range.
    """

    horizon: int = 1
    initial_train: int = 252
    step: int = 1
    gap: int = 0
    window: int | None = None
    seasonal_period: int = 1

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise DataValidationError(f"horizon must be positive, got {self.horizon}")
        if self.initial_train < _MIN_TRAIN_ROWS:
            raise DataValidationError(
                f"initial_train must be at least {_MIN_TRAIN_ROWS}, got {self.initial_train}"
            )
        if self.step <= 0:
            raise DataValidationError(f"step must be positive, got {self.step}")
        if self.gap < 0:
            raise DataValidationError(f"gap must be non-negative, got {self.gap}")
        if self.gap >= self.initial_train - 1:
            raise DataValidationError(
                f"gap={self.gap} leaves no training data before the first origin: "
                f"initial_train={self.initial_train} rows minus a gap of {self.gap} "
                f"leaves {self.initial_train - self.gap} row(s), and at least "
                f"{_MIN_TRAIN_ROWS} are required"
            )
        if self.window is not None and self.window < _MIN_TRAIN_ROWS:
            raise DataValidationError(
                f"window must be at least {_MIN_TRAIN_ROWS}, got {self.window}"
            )
        if self.seasonal_period <= 0:
            raise DataValidationError(
                f"seasonal_period must be positive, got {self.seasonal_period}"
            )


@dataclass(frozen=True, slots=True)
class Fold:
    """One training window and the origin it forecasts from.

    Attributes:
        number: Zero-based fold index.
        train_start: First row of the training window, inclusive.
        train_end: Last row of the training window, inclusive.
        origin: Last row observed before the forecast. The first prediction
            targets ``origin + 1``. With no embargo this equals ``train_end``;
            with ``gap = g`` it is ``train_end + g``.
    """

    number: int
    train_start: int
    train_end: int
    origin: int

    @property
    def n_train(self) -> int:
        """Number of rows the model is allowed to see."""
        return self.train_end - self.train_start + 1

    @property
    def gap(self) -> int:
        """Number of embargoed rows between the training window and the origin."""
        return self.origin - self.train_end


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The outcome of a walk-forward run.

    Attributes:
        predictions: Tidy frame with one row per (fold, horizon step, model).
        fold_scales: MASE scale per fold, computed from that fold's training
            window only.
        config: The configuration that produced the run.
    """

    predictions: pd.DataFrame
    fold_scales: pd.Series
    config: BacktestConfig

    @property
    def models(self) -> tuple[str, ...]:
        """Model names present in the result, in insertion order."""
        return tuple(dict.fromkeys(self.predictions["model"].tolist()))

    @property
    def folds(self) -> tuple[int, ...]:
        """Fold numbers present in the result."""
        return tuple(sorted(self.predictions["fold"].unique().tolist()))

    def metrics_for(self, model: str) -> MetricSet:
        """Compute the metric set for one model.

        Args:
            model: A name present in :attr:`predictions`.

        Returns:
            The metrics, with MASE averaged over per-fold scaled errors so
            that each fold contributes one comparable number.

        Raises:
            DataValidationError: If ``model`` is not in the result.
        """
        subset = self.predictions[self.predictions["model"] == model]
        if subset.empty:
            available = ", ".join(self.models)
            raise DataValidationError(
                f"model {model!r} is not in this result; available: {available}"
            )

        per_fold_mase: list[float] = []
        for fold, group in subset.groupby("fold", sort=True):
            scale = float(self.fold_scales.loc[fold])
            per_fold_mase.append(
                mae(
                    group["y_true"].to_numpy(dtype="float64"),
                    group["y_pred"].to_numpy(dtype="float64"),
                )
                / scale
            )

        y_true = subset["y_true"].to_numpy(dtype="float64")
        y_pred = subset["y_pred"].to_numpy(dtype="float64")
        previous = subset["previous"].to_numpy(dtype="float64")
        return MetricSet(
            mae=mae(y_true, y_pred),
            rmse=rmse(y_true, y_pred),
            smape=smape(y_true, y_pred),
            mase=float(np.mean(per_fold_mase)),
            directional_accuracy=directional_accuracy(y_true, y_pred, previous),
            n_observations=int(y_true.size),
        )

    def summary(self) -> pd.DataFrame:
        """Return one row of metrics per model, indexed by model name."""
        rows = {model: self.metrics_for(model).as_dict() for model in self.models}
        return pd.DataFrame.from_dict(rows, orient="index")


class WalkForwardBacktester:
    """Runs models over expanding or rolling walk-forward folds.

    Args:
        config: The fold layout.

    Example:
        ::

            backtester = WalkForwardBacktester(BacktestConfig(horizon=1))
            result = backtester.run(series, {"naive": Naive()})
            print(result.summary())
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def folds(self, n_rows: int) -> tuple[Fold, ...]:
        """Return the fold layout for a series of ``n_rows`` rows.

        Args:
            n_rows: Length of the series being backtested.

        Returns:
            Folds in chronological order. Empty when the series is too short
            for even one complete fold.
        """
        config = self.config
        folds: list[Fold] = []
        number = 0
        for origin in range(config.initial_train - 1, n_rows, config.step):
            # The embargo pushes the training window back; the rows between
            # train_end and origin are neither trained on nor predicted.
            train_end = origin - config.gap
            if train_end < 1:
                continue
            if origin + config.horizon > n_rows - 1:
                break
            train_start = 0 if config.window is None else max(0, train_end - config.window + 1)
            # MASE needs at least one naive error in the training window.
            if train_end - train_start + 1 <= config.seasonal_period:
                continue
            folds.append(
                Fold(
                    number=number,
                    train_start=train_start,
                    train_end=train_end,
                    origin=origin,
                )
            )
            number += 1
        return tuple(folds)

    def run(
        self,
        series: pd.Series,
        models: Mapping[str, Forecaster],
        features: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Backtest ``models`` over ``series``.

        Args:
            series: The full series. Only the rows a fold is allowed to see are
                ever passed to a model.
            models: Model name to an *unfitted* prototype. Each fold receives
                a fresh clone, so state cannot carry across folds.
            features: Optional exogenous columns aligned to ``series.index``.
                Sliced to the same rows as the history for each fold.

        Returns:
            A :class:`BacktestResult`.

        Raises:
            DataValidationError: If the inputs are invalid, a model's horizon
                does not match the config, or no fold fits in the series.
        """
        validate_history(series, features, requires_features=False)
        self._check_models(models)

        folds = self.folds(len(series))
        if not folds:
            raise DataValidationError(
                f"no fold fits in a series of {len(series)} rows with "
                f"initial_train={self.config.initial_train}, "
                f"gap={self.config.gap} and horizon={self.config.horizon}"
            )

        records: list[dict[str, object]] = []
        scales: dict[int, float] = {}

        for fold in folds:
            history = series.iloc[fold.train_start : fold.train_end + 1]
            history_features = None
            if features is not None:
                history_features = features.iloc[fold.train_start : fold.train_end + 1]
            target_start = fold.origin + 1
            target_slice = series.iloc[target_start : target_start + self.config.horizon]
            # previous[k] is the observation immediately before the target at
            # step k + 1; all of them are actual values, used only for scoring.
            previous_slice = series.iloc[fold.origin : fold.origin + self.config.horizon]

            scales[fold.number] = mase_scale(
                history.to_numpy(dtype="float64"),
                seasonal_period=self.config.seasonal_period,
            )

            for model_name, prototype in models.items():
                fitted = prototype.clone().fit(history, history_features)
                y_pred = fitted.predict()
                records.extend(
                    {
                        "fold": fold.number,
                        "timestamp": target_slice.index[step],
                        "horizon": step + 1,
                        "y_true": float(target_slice.iloc[step]),
                        "y_pred": float(y_pred[step]),
                        "model": model_name,
                        "previous": float(previous_slice.iloc[step]),
                    }
                    for step in range(self.config.horizon)
                )

        predictions = pd.DataFrame.from_records(records, columns=list(_PREDICTION_COLUMNS))
        return BacktestResult(
            predictions=predictions,
            fold_scales=pd.Series(scales, name="mase_scale", dtype="float64"),
            config=self.config,
        )

    def _check_models(self, models: Mapping[str, Forecaster]) -> None:
        """Validate the model mapping before doing any work.

        Args:
            models: The prototypes supplied to :meth:`run`.

        Raises:
            DataValidationError: If the mapping is empty, a value is not a
                ``Forecaster``, or a prototype's horizon differs from the
                config's.
        """
        if not models:
            raise DataValidationError("at least one model is required")
        for model_name, prototype in models.items():
            if not isinstance(prototype, Forecaster):
                raise DataValidationError(
                    f"model {model_name!r} is a {type(prototype).__name__}, not a Forecaster"
                )
            if prototype.horizon != self.config.horizon:
                raise DataValidationError(
                    f"model {model_name!r} was built with horizon={prototype.horizon} "
                    f"but the backtest config uses horizon={self.config.horizon}; "
                    "construct the model with the same horizon"
                )
