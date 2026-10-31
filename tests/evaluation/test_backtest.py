"""Tests for the walk-forward backtester.

Two kinds of test live here. The first check the arithmetic of the fold
layout and the tidy result frame. The second check the property the module
exists for: **a fold's prediction must be computable from that fold's
training window alone.** Those follow the same three-step pattern as the
feature-engineering leakage tests — a structural check on the boundaries, a
perturbation check on the inputs, and a negative control proving the check
fails when a leak really is present.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.evaluation.backtest import (
    BacktestConfig,
    BacktestResult,
    Fold,
    WalkForwardBacktester,
)
from forecastforge.evaluation.metrics import mase_scale
from forecastforge.models import Drift, Forecaster, Naive

_HORIZON_MISMATCH = "construct the model with the same horizon"


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series([float(value) for value in values], index=index)


def _counting() -> pd.Series:
    """1, 2, 3, ... 10 — a series whose naive forecast is always one too low."""
    return _series([float(value) for value in range(1, 11)])


# -- helper models ---------------------------------------------------------


class _Recorder(Forecaster):
    """Records what each fold was allowed to see.

    ``clone`` is overridden to share the log, because the base implementation
    deep-copies the instance and the copy's appends would be lost.
    """

    name = "recorder"

    def __init__(self, log: list[dict[str, object]], *, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        self._log = log

    def clone(self) -> _Recorder:
        """Return a fresh recorder writing to the same log."""
        return _Recorder(self._log, horizon=self.horizon)

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Record the slice boundaries."""
        self._log.append(
            {
                "n": len(series),
                "first": series.index[0],
                "last": series.index[-1],
                "n_features": 0 if features is None else len(features),
                "aligned": features is not None and features.index.equals(series.index),
            }
        )

    def _predict(self) -> np.ndarray:
        """Return a constant, since only the log matters here."""
        return np.zeros(self.horizon, dtype="float64")


class _Counting(Forecaster):
    """Predicts how many times it has been fitted."""

    name = "counting"

    def __init__(self, *, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        self.fits = 0

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Increment the fit counter."""
        self.fits += 1

    def _predict(self) -> np.ndarray:
        """Return the fit counter, so a shared instance is visible."""
        return np.full(self.horizon, float(self.fits), dtype="float64")


class _Peeker(Forecaster):
    """Tries to read the observation immediately after its training window."""

    name = "peeker"

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Keep the history."""
        self._history = series

    def _predict(self) -> np.ndarray:
        """Index one past the end, which is exactly what leakage would look like."""
        return np.array([float(self._history.iloc[len(self._history)])])


class _Oracle(Forecaster):
    """A model that has been handed the answer key.

    It is fitted on the fold's history but reads its prediction from the full
    series. If the harness could be fooled, this is what would fool it.
    """

    name = "oracle"

    def __init__(self, full: pd.Series, *, horizon: int = 1) -> None:
        super().__init__(horizon=horizon)
        self._full = full
        self._n_train = 0

    def _fit(self, series: pd.Series, features: pd.DataFrame | None) -> None:
        """Remember how many rows the fold was allowed to see."""
        self._n_train = len(series)

    def _predict(self) -> np.ndarray:
        """Return the value the model was not supposed to have."""
        return np.array([float(self._full.iloc[self._n_train])])


# -- fold layout -----------------------------------------------------------


def test_expanding_window_folds_grow_by_one_row() -> None:
    # Ten rows, first origin at row 4, horizon 1 -> origins 4..8, and every
    # fold starts at row 0.
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5)).folds(10)
    assert [fold.origin for fold in folds] == [4, 5, 6, 7, 8]
    assert {fold.train_start for fold in folds} == {0}
    assert [fold.n_train for fold in folds] == [5, 6, 7, 8, 9]


def test_rolling_window_folds_keep_a_constant_length() -> None:
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5, window=3)).folds(10)
    assert [fold.train_start for fold in folds] == [2, 3, 4, 5, 6]
    assert {fold.n_train for fold in folds} == {3}


def test_step_skips_origins() -> None:
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5, step=3)).folds(10)
    assert [fold.origin for fold in folds] == [4, 7]


def test_folds_stop_when_the_horizon_runs_off_the_end() -> None:
    # With ten rows the last origin that can supply two targets is row 7,
    # because the second target would be row 9 and row 10 does not exist.
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5, horizon=2)).folds(10)
    assert [fold.origin for fold in folds] == [4, 5, 6, 7]


def test_gap_embargoes_rows_between_the_training_window_and_the_origin() -> None:
    # Origins 4..8 with gap 2 -> training ends at 2..6, so rows 3 and 4 are
    # never seen by the first fold, and the target is row 5.
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5, gap=2)).folds(10)
    assert [fold.train_end for fold in folds] == [2, 3, 4, 5, 6]
    assert [fold.origin for fold in folds] == [4, 5, 6, 7, 8]
    assert {fold.gap for fold in folds} == {2}
    assert folds[0].n_train == 3


# -- gap semantics ---------------------------------------------------------
#
# The gap is the one part of the fold layout that is easy to get subtly wrong,
# so its contract is pinned by four properties rather than left implicit in
# the slicing: the origin does not move, a supervised pair is legal only when
# its target ends by the origin, a gap shorter than the horizon is exactly
# where naive pairing leaks, and the scored rows are identical for any gap.


def _pair_targets(fold: Fold, horizon: int, *, causal: bool) -> list[int]:
    """Return the target indices a supervised model would train on.

    A model that respects the horizon stops its last target at ``train_end``
    (the last row the fold may see). A model with the classic off-by-horizon
    bug keeps feature rows up to ``train_end`` and reads targets ``horizon``
    further, so its last target is ``train_end + horizon``.

    Args:
        fold: The fold whose history the model is fitted on.
        horizon: Forecast horizon, i.e. how far ahead each target sits.
        causal: Whether to use the horizon-respecting rule.

    Returns:
        The target indices, in order.
    """
    last = fold.train_end if causal else fold.train_end + horizon
    return list(range(horizon, last + 1))


def assert_training_targets_are_causal(targets: list[int], origin: int) -> None:
    """Fail unless every training target is at or before the forecast origin.

    Args:
        targets: Target indices used during training.
        origin: The fold's forecast origin.

    Raises:
        AssertionError: If any target lies after the origin.
    """
    past = [target for target in targets if target > origin]
    assert not past, f"training targets past the origin {origin}: {past}"


def test_gap_keeps_the_forecast_origin_fixed() -> None:
    """The gap shortens the training window; it never moves the origin."""
    base = WalkForwardBacktester(BacktestConfig(initial_train=10, gap=0)).folds(25)
    gapped = WalkForwardBacktester(BacktestConfig(initial_train=10, gap=3)).folds(25)
    assert [fold.origin for fold in base] == [fold.origin for fold in gapped]
    assert [fold.train_end for fold in gapped] == [fold.origin - 3 for fold in gapped]
    assert {fold.gap for fold in gapped} == {3}


def test_a_training_row_is_legal_only_if_its_target_ends_by_the_origin() -> None:
    """A pair ``(row i, target i + h)`` is legal only when ``i + h + gap <= origin``.

    The backtester hands each model a history ending at ``origin - gap``, so
    the last legal pair row is ``origin - gap - h``. Stating that bound with
    equality on the last legal row is what makes the invariant testable.
    """
    config = BacktestConfig(initial_train=12, horizon=3, gap=2)
    folds = WalkForwardBacktester(config).folds(40)
    assert folds, "the fixture must produce at least one fold"
    for fold in folds:
        targets = _pair_targets(fold, config.horizon, causal=True)
        assert_training_targets_are_causal(targets, fold.origin)
        last_row = targets[-1] - config.horizon
        assert last_row >= fold.train_start
        assert last_row + config.horizon + fold.gap == fold.origin


def test_a_gap_shorter_than_the_horizon_is_where_naive_pairing_leaks() -> None:
    """With ``gap < h``, the naive off-by-horizon rule trains past the origin.

    This is the failure the horizon rule prevents: the last ``h - gap`` target
    indices fall after the origin, i.e. into the future the model is asked to
    predict. The audit must reject them.
    """
    horizon = 3
    config = BacktestConfig(initial_train=10, horizon=horizon, gap=1)
    fold = WalkForwardBacktester(config).folds(30)[0]
    assert config.gap < horizon

    causal = _pair_targets(fold, horizon, causal=True)
    assert_training_targets_are_causal(causal, fold.origin)

    leaky = _pair_targets(fold, horizon, causal=False)
    with pytest.raises(AssertionError, match="past the origin"):
        assert_training_targets_are_causal(leaky, fold.origin)
    leaked = [target for target in leaky if target > fold.origin]
    assert len(leaked) == horizon - config.gap


def test_gap_changes_the_training_window_but_not_the_scored_rows(
    synthetic_close: pd.Series,
) -> None:
    """The same folds score the same timestamps and outcomes for any gap.

    Only the training window moves, so the comparison across gaps is honest:
    the models are judged on identical targets.
    """
    base = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10, gap=0)).run(
        synthetic_close, {"naive": Naive()}
    )
    gapped = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10, gap=4)).run(
        synthetic_close, {"naive": Naive()}
    )
    assert base.folds == gapped.folds
    pd.testing.assert_series_equal(
        base.predictions["timestamp"].reset_index(drop=True),
        gapped.predictions["timestamp"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        base.predictions["y_true"].reset_index(drop=True),
        gapped.predictions["y_true"].reset_index(drop=True),
    )


def test_folds_are_skipped_when_the_training_window_cannot_scale_mase() -> None:
    # A seasonal period of 5 needs more than 5 training rows, so the first
    # origin (5 rows) is skipped and the layout starts one row later.
    folds = WalkForwardBacktester(BacktestConfig(initial_train=5, seasonal_period=5)).folds(12)
    assert folds[0].train_end == 5
    assert [fold.number for fold in folds] == list(range(len(folds)))


def test_fold_numbers_are_contiguous_after_skips() -> None:
    # The first origin leaves only three training rows, which is not more than
    # the seasonal period, so it is dropped; numbering still starts at zero.
    folds = WalkForwardBacktester(BacktestConfig(initial_train=3, seasonal_period=3)).folds(10)
    assert [fold.origin for fold in folds] == [3, 4, 5, 6, 7, 8]
    assert [fold.number for fold in folds] == [0, 1, 2, 3, 4, 5]


def test_no_fold_fits_a_series_shorter_than_the_initial_window() -> None:
    assert WalkForwardBacktester(BacktestConfig(initial_train=10)).folds(10) == ()


# -- configuration validation ---------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon must be positive"),
        ({"initial_train": 1}, "initial_train must be at least 2"),
        ({"step": 0}, "step must be positive"),
        ({"gap": -1}, "gap must be non-negative"),
        ({"initial_train": 5, "gap": 4}, "leaves no training data"),
        ({"window": 1}, "window must be at least 2"),
        ({"seasonal_period": 0}, "seasonal_period must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        BacktestConfig(**kwargs)


# -- the tidy result frame -------------------------------------------------


def test_predictions_frame_has_the_documented_columns() -> None:
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        _counting(), {"naive": Naive()}
    )
    assert list(result.predictions.columns) == [
        "fold",
        "timestamp",
        "horizon",
        "y_true",
        "y_pred",
        "model",
        "previous",
    ]
    assert result.predictions["y_true"].dtype == np.dtype("float64")
    assert result.predictions["y_pred"].dtype == np.dtype("float64")


def test_naive_predictions_are_the_last_training_value() -> None:
    # Fold 0 trains on 1..5 and is scored against 6, so the naive forecast of
    # 5 is off by exactly one.
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        _counting(), {"naive": Naive()}
    )
    fold0 = result.predictions[result.predictions["fold"] == 0]
    assert fold0["y_true"].tolist() == [6.0]
    assert fold0["y_pred"].tolist() == [5.0]
    assert fold0["previous"].tolist() == [5.0]
    assert fold0["timestamp"].tolist() == [_counting().index[5]]
    assert fold0["horizon"].tolist() == [1]


def test_multi_horizon_predictions_are_indexed_by_step() -> None:
    result = WalkForwardBacktester(BacktestConfig(initial_train=5, horizon=2)).run(
        _counting(), {"naive": Naive(horizon=2)}
    )
    fold0 = result.predictions[result.predictions["fold"] == 0]
    assert fold0["horizon"].tolist() == [1, 2]
    assert fold0["y_true"].tolist() == [6.0, 7.0]
    assert fold0["y_pred"].tolist() == [5.0, 5.0]
    # previous[k] is the observation before target k, so it walks forward too.
    assert fold0["previous"].tolist() == [5.0, 6.0]


def test_each_fold_gets_a_fresh_clone() -> None:
    model = _Counting()
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        _counting(), {"counting": model}
    )
    # Every fold fits from scratch, so each one reports its first fit.
    assert set(result.predictions["y_pred"].tolist()) == {1.0}
    assert model.fits == 0


def test_features_are_sliced_with_the_history() -> None:
    series = _counting()
    features = pd.DataFrame({"x": np.arange(len(series), dtype="float64")}, index=series.index)
    log: list[dict[str, object]] = []
    WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        series, {"recorder": _Recorder(log)}, features=features
    )
    assert [entry["n"] for entry in log] == [5, 6, 7, 8, 9]
    assert [entry["n_features"] for entry in log] == [5, 6, 7, 8, 9]
    assert all(entry["aligned"] for entry in log)
    assert [entry["last"] for entry in log] == [series.index[i] for i in (4, 5, 6, 7, 8)]


def test_fold_scales_are_computed_from_each_training_window() -> None:
    series = _counting()
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(series, {"naive": Naive()})
    expected = [
        mase_scale(series.iloc[: fold.train_end + 1].to_numpy(dtype="float64"))
        for fold in WalkForwardBacktester(result.config).folds(len(series))
    ]
    assert result.fold_scales.tolist() == pytest.approx(expected)
    assert result.fold_scales.index.tolist() == list(result.folds)


def test_metrics_are_scaled_per_fold() -> None:
    # Every fold predicts a flat 5..9 against 6..10 and 7..11: MAE is 1.5 per
    # fold, and the one-step naive error of every training window is 1.0.
    result = WalkForwardBacktester(BacktestConfig(initial_train=5, horizon=2)).run(
        _counting(), {"naive": Naive(horizon=2)}
    )
    metrics = result.metrics_for("naive")
    assert metrics.mase == pytest.approx(1.5)
    assert metrics.mae == pytest.approx(1.5)
    assert metrics.directional_accuracy == 0.0
    assert metrics.n_observations == 8


def test_metrics_for_an_unknown_model_raises() -> None:
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        _counting(), {"naive": Naive()}
    )
    with pytest.raises(DataValidationError, match="not in this result"):
        result.metrics_for("arima")


def test_summary_has_one_row_per_model() -> None:
    result = WalkForwardBacktester(BacktestConfig(initial_train=5)).run(
        _counting(), {"naive": Naive(), "drift": Drift()}
    )
    summary = result.summary()
    assert summary.index.tolist() == ["naive", "drift"]
    assert "mase" in summary.columns
    assert result.models == ("naive", "drift")


# -- run() validation ------------------------------------------------------


def test_run_rejects_an_empty_model_mapping() -> None:
    with pytest.raises(DataValidationError, match="at least one model"):
        WalkForwardBacktester().run(_counting(), {})


def test_run_rejects_a_non_forecaster() -> None:
    with pytest.raises(DataValidationError, match="not a Forecaster"):
        WalkForwardBacktester().run(_counting(), {"bogus": object()})  # type: ignore[dict-item]


def test_run_rejects_a_horizon_mismatch() -> None:
    with pytest.raises(DataValidationError, match=_HORIZON_MISMATCH):
        WalkForwardBacktester(BacktestConfig(horizon=2)).run(
            _counting(), {"naive": Naive(horizon=1)}
        )


def test_run_rejects_a_series_too_short_for_any_fold() -> None:
    with pytest.raises(DataValidationError, match="no fold fits"):
        WalkForwardBacktester(BacktestConfig(initial_train=20)).run(_counting(), {"naive": Naive()})


def test_run_rejects_an_empty_series() -> None:
    with pytest.raises(DataValidationError, match="history is empty"):
        WalkForwardBacktester().run(pd.Series(dtype="float64"), {"naive": Naive()})


# -- leakage ---------------------------------------------------------------


def _layout(result: BacktestResult, series: pd.Series) -> dict[int, Fold]:
    """Recompute the fold layout a result claims to come from."""
    folds = WalkForwardBacktester(result.config).folds(len(series))
    return {fold.number: fold for fold in folds}


def assert_no_leakage(result: BacktestResult, series: pd.Series) -> None:
    """Fail unless every scored row is strictly after its fold's training window.

    Args:
        result: The backtest output to audit.
        series: The full series the result was produced from.

    Raises:
        AssertionError: If a fold scores a row its model could have seen.
    """
    layout = _layout(result, series)
    assert set(result.folds) <= set(layout), "result contains folds the config cannot produce"
    for (number, model), rows in result.predictions.groupby(["fold", "model"]):
        fold = layout[int(number)]
        cutoff = series.index[fold.train_end]
        for step in range(1, result.config.horizon + 1):
            expected = series.index[fold.origin + step]
            observed = rows.loc[rows["horizon"] == step, "timestamp"].tolist()
            assert observed == [expected], (
                f"fold {number} ({model}) step {step} targets {observed}, expected [{expected}]"
            )
        assert (rows["timestamp"] > cutoff).all(), (
            f"fold {number} ({model}) scored a row at or before its cutoff {cutoff}"
        )


def test_the_backtester_scores_only_rows_after_the_training_window(
    synthetic_close: pd.Series,
) -> None:
    result = WalkForwardBacktester(BacktestConfig(initial_train=60, horizon=3, step=10)).run(
        synthetic_close, {"naive": Naive(horizon=3), "drift": Drift(horizon=3)}
    )
    assert_no_leakage(result, synthetic_close)


def test_the_leakage_check_rejects_an_off_by_one_boundary(
    synthetic_close: pd.Series,
) -> None:
    # Simulate the classic bug: the "target" is really the fold's last
    # training row. The audit must catch it, otherwise it proves nothing.
    result = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10)).run(
        synthetic_close, {"naive": Naive()}
    )
    layout = _layout(result, synthetic_close)
    leaky = result.predictions.copy()
    leaky["timestamp"] = [synthetic_close.index[layout[int(n)].train_end] for n in leaky["fold"]]
    with pytest.raises(AssertionError, match="targets"):
        assert_no_leakage(replace(result, predictions=leaky), synthetic_close)


def test_perturbing_the_future_leaves_the_first_fold_untouched(
    synthetic_close: pd.Series,
) -> None:
    backtester = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10))
    models = {"naive": Naive(), "drift": Drift(window=20)}
    baseline = backtester.run(synthetic_close, models)

    first = backtester.folds(len(synthetic_close))[0]
    perturbed = synthetic_close.copy()
    tail = perturbed.index[first.origin + 2 :]
    perturbed.loc[tail] = perturbed.loc[tail] * 1.5
    after = backtester.run(perturbed, models)

    before_rows = baseline.predictions[baseline.predictions["fold"] == 0]
    after_rows = after.predictions[after.predictions["fold"] == 0]
    pd.testing.assert_frame_equal(
        before_rows.reset_index(drop=True), after_rows.reset_index(drop=True)
    )


def test_truncating_the_series_after_the_target_leaves_the_prediction_unchanged(
    synthetic_close: pd.Series,
) -> None:
    backtester = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10))
    models = {"naive": Naive(), "drift": Drift(window=20)}
    full = backtester.run(synthetic_close, models)

    first = backtester.folds(len(synthetic_close))[0]
    short = backtester.run(synthetic_close.iloc[: first.origin + 2], models)

    full_rows = full.predictions[full.predictions["fold"] == 0].reset_index(drop=True)
    short_rows = short.predictions[short.predictions["fold"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_rows, short_rows)


def test_a_model_cannot_read_past_its_training_window(synthetic_close: pd.Series) -> None:
    # The only channel into a fold is the history slice, and that slice ends
    # at the cutoff: reading one row further raises rather than returning data.
    with pytest.raises(IndexError):
        WalkForwardBacktester(BacktestConfig(initial_train=60, step=10)).run(
            synthetic_close, {"peeker": _Peeker()}
        )


def test_the_harness_would_reward_a_model_that_could_see_the_future(
    synthetic_close: pd.Series,
) -> None:
    # Negative control for the two checks above: a model holding the answer
    # key scores a perfect MAE, so a leak, if one existed, would be obvious
    # rather than silent.
    result = WalkForwardBacktester(BacktestConfig(initial_train=60, step=10)).run(
        synthetic_close, {"oracle": _Oracle(synthetic_close), "naive": Naive()}
    )
    assert result.metrics_for("oracle").mae == 0.0
    assert result.metrics_for("naive").mae > 0.0
