"""Offline model benchmark.

Runs every available forecaster over several deterministic synthetic series
and reports the mean and standard deviation across seeds. The point is not to
crown a winner but to state plainly what happens: on a near-random-walk price
path the naive forecast is hard to beat, and a benchmark that hides that is
worse than no benchmark at all.

Nothing here is tuned on the test folds. The backtester only ever shows a model
the history at or before each fold's cutoff, the hyperparameters are fixed in
:func:`build_models`, and no model is selected or adjusted after its scores are
known. The series are generated from fixed seeds, so a rerun reproduces every
number in the report.

Usage::

    python -m forecastforge.evaluation.benchmark
    python -m forecastforge.evaluation.benchmark --seeds 1 2 3 --length 1200
    python -m forecastforge.evaluation.benchmark --fast        # 1 seed, short series
    BENCHMARK_FAST=1 python -m forecastforge.evaluation.benchmark

The optional forecasters are included only when their extra is installed; the
report names the models it actually ran, so a core install produces a shorter
but still honest table.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd

from forecastforge.config._optional import optional_import
from forecastforge.core.errors import DataValidationError
from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.evaluation.backtest import BacktestConfig, BacktestResult, WalkForwardBacktester
from forecastforge.evaluation.metrics import mae
from forecastforge.features.builtins import CalendarFeatures, Returns, RollingStat
from forecastforge.features.frame import FeatureSet
from forecastforge.features.pipeline import FeaturePipeline
from forecastforge.models import Drift, Forecaster, MovingAverage, Naive, SeasonalNaive
from forecastforge.models.gbm import LightGBMForecaster, lightgbm_available
from forecastforge.models.statistical import (
    ARIMAForecaster,
    ETSForecaster,
    statsmodels_available,
)
from forecastforge.schemas.time_series import TimeSeries

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "build_models",
    "main",
    "render_markdown",
    "run_benchmark",
    "save_report",
]

_PYPLOT = optional_import("matplotlib.pyplot")

# Reported in this order, headline metric first.
_METRICS = ("mase", "mae", "rmse", "smape", "directional_accuracy")

# Per-metric display precision: errors in price units need more digits than a
# unit-free ratio.
_DIGITS = {"mase": 3, "mae": 3, "rmse": 3, "smape": 2, "directional_accuracy": 1}

_BAR_COLOR = "#4C78A8"
_REFERENCE_COLOR = "#C44E52"
_GRID_COLOR = "#D8DCE3"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """What to run and over which series.

    Args:
        seeds: Generator seeds. One synthetic path per seed, so the spread
            across seeds is the uncertainty the table reports. Duplicates are
            rejected because they would silently double-weight a path.
        length: Rows per synthetic series. Short series make every model look
            alike, so the default is long enough for the fold count to matter.
        horizon: Forecast horizon in periods.
        initial_train: Rows in the first training window.
        step: Rows between forecast origins.
        gap: Embargo between the training window and the origin.
        season: Period the seasonal naive baseline uses. Five business days is
            one week, the shortest calendar cycle in the data.

    The MASE scale is always the one-step naive error, so 1.0 means "exactly
    the naive forecast" and the table can be read without checking a footnote.
    Scaling by a seasonal error instead would make the same scores look better
    by a factor of roughly ``sqrt(season)`` on a random walk, which is a
    presentation choice, not a finding.

    Raises:
        DataValidationError: If the configuration cannot produce folds.
    """

    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    length: int = 2000
    horizon: int = 1
    initial_train: int = 500
    step: int = 20
    gap: int = 0
    season: int = 5

    def __post_init__(self) -> None:
        if not self.seeds:
            raise DataValidationError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise DataValidationError(f"seeds must be distinct, got {self.seeds}")
        if self.season <= 0:
            raise DataValidationError(f"season must be positive, got {self.season}")
        if self.length <= self.initial_train + self.horizon:
            raise DataValidationError(
                f"length={self.length} leaves no out-of-sample rows after "
                f"initial_train={self.initial_train} and horizon={self.horizon}"
            )
        # Constructing the backtest config validates step, gap, and the rest.
        _ = self.backtest_config

    @property
    def backtest_config(self) -> BacktestConfig:
        """The fold layout these settings describe."""
        return BacktestConfig(
            horizon=self.horizon,
            initial_train=self.initial_train,
            step=self.step,
            gap=self.gap,
            # One-step scale on purpose; see the class docstring.
            seasonal_period=1,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Every per-seed score, plus the per-fold MASE behind them.

    Attributes:
        metrics: Long frame with one row per (seed, model) and one column per
            metric.
        fold_mase: Long frame with one row per (seed, model, fold), used for
            the distribution plot.
        models: Model names in the order they were run.
        config: The configuration that produced the run.
    """

    metrics: pd.DataFrame
    fold_mase: pd.DataFrame
    models: tuple[str, ...]
    config: BenchmarkConfig

    def summary(self) -> pd.DataFrame:
        """Return mean and standard deviation per model, across seeds."""
        grouped = self.metrics.groupby("model", sort=False)
        columns: dict[str, pd.Series] = {}
        for metric in _METRICS:
            columns[f"{metric}_mean"] = grouped[metric].mean()
            columns[f"{metric}_std"] = grouped[metric].std(ddof=0)
        return pd.DataFrame(columns).loc[list(self.models)]

    def n_folds(self) -> int:
        """Return the number of folds in one seed's backtest."""
        return int(self.fold_mase["fold"].nunique())


def _feature_matrix(time_series: TimeSeries) -> FeatureSet:
    """Build the feature set the learned model consumes.

    The pipeline is causal — the feature tests prove it — so computing it once
    over the full series and slicing it per fold cannot leak. The backtester
    slices it with the history, which keeps the alignment honest.
    """
    pipeline = FeaturePipeline(
        [
            Returns(periods=1, log=True),
            RollingStat(window=20, stat="std", column="close_logret1"),
            RollingStat(window=10, stat="mean", column="close_logret1"),
            CalendarFeatures(["dayofweek", "month_cyclic"]),
        ]
    )
    return pipeline.apply(time_series)


def build_models(config: BenchmarkConfig) -> dict[str, Forecaster]:
    """Return the models to compare, keyed by report name.

    Args:
        config: Supplies the horizon and the seasonal period.

    Returns:
        The four dependency-free baselines, plus ETS, ARIMA, and LightGBM when
        their extra is installed.
    """
    horizon = config.horizon
    models: dict[str, Forecaster] = {
        "naive": Naive(horizon=horizon),
        "seasonal_naive": SeasonalNaive(season=config.season, horizon=horizon),
        "moving_average": MovingAverage(window=20, horizon=horizon),
        "drift": Drift(window=20, horizon=horizon),
    }
    if statsmodels_available():
        models["ets"] = ETSForecaster(horizon=horizon)
        models["arima"] = ARIMAForecaster(order=(1, 1, 0), horizon=horizon)
    if lightgbm_available():
        models["lightgbm"] = LightGBMForecaster(horizon=horizon)
    return models


def _per_fold_mase(result: BacktestResult, model: str, seed: int) -> list[dict[str, object]]:
    """Return one row per fold holding ``MAE / scale`` for ``model``."""
    subset = result.predictions[result.predictions["model"] == model]
    rows: list[dict[str, object]] = []
    for fold, group in subset.groupby("fold", sort=True):
        scale = float(result.fold_scales.loc[fold])
        rows.append(
            {
                "seed": seed,
                "model": model,
                "fold": int(cast("int", fold)),
                "mase": mae(
                    group["y_true"].to_numpy(dtype="float64"),
                    group["y_pred"].to_numpy(dtype="float64"),
                )
                / scale,
            }
        )
    return rows


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """Run every model over every seed.

    Args:
        config: Benchmark settings. Defaults to :class:`BenchmarkConfig`.

    Returns:
        The per-seed scores and the per-fold MASE they came from.
    """
    cfg = config or BenchmarkConfig()
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    models: tuple[str, ...] = ()

    for seed in cfg.seeds:
        time_series = SyntheticSource(config=SyntheticConfig(seed=seed, length=cfg.length)).fetch(
            symbol="SPY"
        )
        candidates = build_models(cfg)
        if not models:
            models = tuple(candidates)
        result = WalkForwardBacktester(cfg.backtest_config).run(
            time_series.target_series, candidates, _feature_matrix(time_series).matrix
        )
        for name in candidates:
            metric_rows.append({"seed": seed, "model": name, **result.metrics_for(name).as_dict()})
            fold_rows.extend(_per_fold_mase(result, name, seed))

    return BenchmarkResult(
        metrics=pd.DataFrame.from_records(metric_rows),
        fold_mase=pd.DataFrame.from_records(fold_rows),
        models=models,
        config=cfg,
    )


def _format(mean: float, std: float, *, digits: int) -> str:
    """Return ``mean ± std`` at the given precision."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def render_markdown(result: BenchmarkResult) -> str:
    """Render the benchmark table as markdown.

    Args:
        result: A completed benchmark.

    Returns:
        A document with a short method note, the table, and the reading of it.
    """
    summary = result.summary()
    cfg = result.config
    header = [
        "# ForecastForge benchmark",
        "",
        "Generated by `python -m forecastforge.evaluation.benchmark`. Every number is",
        "reproducible: the series come from fixed seeds and no model is tuned on the",
        "scored folds.",
        "",
        f"* {len(cfg.seeds)} synthetic series of {cfg.length} business days, seeds "
        f"{', '.join(str(seed) for seed in cfg.seeds)}",
        f"* horizon {cfg.horizon}, {result.n_folds()} walk-forward folds per series "
        f"(first training window {cfg.initial_train} rows, step {cfg.step}, gap {cfg.gap})",
        "* MASE is scaled by the one-step naive error",
        f"* models: {', '.join(result.models)}",
        "",
        "| Model | MASE | MAE | RMSE | sMAPE (%) | Directional accuracy (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = []
    # A plain mapping keeps the rendering free of pandas indexers, whose
    # return types depend on the key in ways that are awkward to annotate.
    scores: dict[str, dict[str, float]] = {
        str(model): {str(column): float(value) for column, value in row.items()}
        for model, row in summary.to_dict(orient="index").items()
    }
    for model in result.models:
        cells = [
            _format(
                scores[model][f"{metric}_mean"],
                scores[model][f"{metric}_std"],
                digits=_DIGITS[metric],
            )
            for metric in _METRICS
        ]
        rows.append(f"| {model} | " + " | ".join(cells) + " |")

    best = min(result.models, key=lambda name: scores[name]["mase_mean"])
    best_value = scores[best]["mase_mean"]
    naive_mean = scores["naive"]["mase_mean"]
    naive_std = scores["naive"]["mase_std"]
    # A model only counts as better if it clears the seed-to-seed spread of the
    # naive baseline itself. Without this rule a 0.001 gap on a 0.14 spread
    # would be reported as a win, which is not what the numbers say.
    threshold = naive_mean - naive_std
    qualifiers = [
        model
        for model in result.models
        if model != "naive" and scores[model]["mase_mean"] < threshold
    ]

    footer = [
        "",
        "Values are mean ± standard deviation across seeds. MASE is unit-free and",
        "scaled by the one-step naive error on each fold's training window, so 1.0",
        "is exactly the naive forecast: below 1 beats it, above 1 is worse than",
        "doing nothing.",
        "",
        "A model is only described as better when its mean MASE is at least one",
        "naive standard deviation below the naive mean; anything closer is within",
        "seed noise and is reported as a tie.",
        "",
    ]
    if qualifiers:
        winner = min(qualifiers, key=lambda name: scores[name]["mase_mean"])
        verdict = (
            f"**{winner} beats the naive forecast.** Its mean MASE is "
            f"{scores[winner]['mase_mean']:.3f} against {naive_mean:.3f} for naive, "
            f"a gap wider than the {naive_std:.3f} seed-to-seed spread of naive "
            "itself. That is a result about this synthetic generator, not a promise "
            "about real prices."
        )
    else:
        verdict = (
            "**No model beats the naive forecast.** Naive scores "
            f"{naive_mean:.3f} ± {naive_std:.3f} and nothing is more than one "
            f"standard deviation below it; the closest is {best} at {best_value:.3f}, "
            "which is a tie. That is what a near-random-walk price path implies: "
            "there is no exploitable structure left to find, and every additional "
            "parameter mostly adds noise."
        )
    return "\n".join([*header, *rows, *footer, verdict, ""])


def _require_pyplot() -> ModuleType:
    """Return ``matplotlib.pyplot``, or explain how to install it.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if _PYPLOT is None:
        raise ImportError(
            'matplotlib is required for the benchmark plots. Install with: pip install -e ".[viz]"'
        )
    return _PYPLOT


def _plot(result: BenchmarkResult, path: Path) -> Path:
    """Draw the MASE summary and the per-fold distribution.

    Args:
        result: A completed benchmark.
        path: Destination image.

    Returns:
        ``path``, for chaining.
    """
    pyplot = _require_pyplot()
    summary = result.summary()
    models = list(result.models)
    positions = np.arange(len(models), dtype="float64")

    figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    left = axes[0]
    left.bar(
        positions,
        summary.loc[models, "mase_mean"].to_numpy(dtype="float64"),
        yerr=summary.loc[models, "mase_std"].to_numpy(dtype="float64"),
        capsize=4,
        color=_BAR_COLOR,
        edgecolor="white",
    )
    left.axhline(
        1.0, color=_REFERENCE_COLOR, linestyle="--", linewidth=1.2, label="naive (MASE = 1)"
    )
    left.set_xticks(positions, models, rotation=20)
    left.set_ylabel("MASE, mean ± std across seeds")
    left.set_title("Mean MASE (lower is better)")
    left.grid(axis="y", color=_GRID_COLOR, linewidth=0.8, alpha=0.8)
    left.set_axisbelow(True)
    left.legend(frameon=False, fontsize=9)

    right = axes[1]
    distributions = [
        result.fold_mase.loc[result.fold_mase["model"] == model, "mase"].to_numpy(dtype="float64")
        for model in models
    ]
    right.boxplot(distributions, tick_labels=models, showfliers=False, widths=0.6)
    right.axhline(1.0, color=_REFERENCE_COLOR, linestyle="--", linewidth=1.2)
    right.set_ylabel("per-fold MASE")
    right.set_title("Distribution of per-fold MASE")
    right.grid(axis="y", color=_GRID_COLOR, linewidth=0.8, alpha=0.8)
    right.set_axisbelow(True)
    right.tick_params(axis="x", rotation=20)

    for axis in axes:
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    figure.suptitle(
        "ForecastForge baselines on synthetic price paths "
        f"({len(result.config.seeds)} seeds, {result.config.length} days)",
        fontsize=12,
    )
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def save_report(
    result: BenchmarkResult, directory: Path, *, plot: bool = True
) -> tuple[Path, Path | None]:
    """Write the markdown table and, when asked and possible, the plot.

    Args:
        result: A completed benchmark.
        directory: Destination directory, created if missing.
        plot: Whether to draw the figure. Ignored when matplotlib is absent.

    Returns:
        The table path and the plot path (``None`` when no figure was drawn).
    """
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / "benchmark_results.md"
    # newline="\n" so a rerun on Windows produces the same bytes as one on
    # Linux, otherwise the committed report shows a spurious diff every time.
    table.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    if not plot or _PYPLOT is None:
        return table, None
    return table, _plot(result, directory / "benchmark_mase.png")


def _emit(text: str) -> None:
    """Write ``text`` to standard output.

    ``print`` is banned in the package by the lint configuration, and this is
    the one place a command-line report legitimately needs raw output.
    """
    sys.stdout.write(text if text.endswith("\n") else f"{text}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write the report.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(
        prog="python -m forecastforge.evaluation.benchmark",
        description="Benchmark every available forecaster on synthetic series.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--length", type=int, default=defaults.length)
    parser.add_argument("--initial-train", type=int, default=defaults.initial_train)
    parser.add_argument("--step", type=int, default=defaults.step)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--season", type=int, default=defaults.season)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the figure (useful on a machine without the viz extra)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="reduced run for CI smoke checks: 1 seed, short series, no plot",
    )
    arguments = parser.parse_args(argv)

    # A reduced run for CI so `make benchmark` can never silently break: one
    # seed, a short series, and no figure. The same effect can be forced
    # without CLI access through the BENCHMARK_FAST environment variable.
    fast = arguments.fast or os.environ.get("BENCHMARK_FAST", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if fast:
        seeds: tuple[int, ...] = (1,)
        length = 600
        initial_train = 300
        step = 100
        plot = False
    else:
        seeds = tuple(arguments.seeds)
        length = arguments.length
        initial_train = arguments.initial_train
        step = arguments.step
        plot = not arguments.no_plot

    config = BenchmarkConfig(
        seeds=seeds,
        length=length,
        horizon=arguments.horizon,
        initial_train=initial_train,
        step=step,
        season=arguments.season,
    )
    result = run_benchmark(config)
    _emit(render_markdown(result))
    table, figure = save_report(result, Path(arguments.output_dir), plot=plot)
    _emit(f"wrote {table}")
    if figure is not None:
        _emit(f"wrote {figure}")
    elif not arguments.no_plot:
        _emit('matplotlib is not installed, so the figure was skipped (pip install -e ".[viz]")')
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
