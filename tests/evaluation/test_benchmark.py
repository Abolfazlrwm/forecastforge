"""Tests for the offline benchmark.

The benchmark is a reporting tool, so the tests check that it reports the right
things — the models it claims to have run, the shape of the summary, and the
honesty rule that a difference smaller than the seed spread is not called a
win — rather than checking any particular accuracy number.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    build_models,
    main,
    render_markdown,
    run_benchmark,
    save_report,
)
from forecastforge.models.gbm import lightgbm_available
from forecastforge.models.statistical import statsmodels_available

# Small enough to run in well under a second, large enough for two folds.
_TINY = {
    "seeds": (1, 2),
    "length": 400,
    "initial_train": 200,
    "step": 100,
}


def _tiny(**overrides: object) -> BenchmarkConfig:
    """Return the tiny configuration, with any field overridden."""
    return BenchmarkConfig(**{**_TINY, **overrides})  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def tiny_result() -> BenchmarkResult:
    """Run the tiny benchmark once and share it across the read-only tests."""
    return run_benchmark(_tiny())


# -- configuration ---------------------------------------------------------


def test_an_empty_seed_list_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="at least one seed"):
        _tiny(seeds=())


def test_duplicate_seeds_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="seeds must be distinct"):
        _tiny(seeds=(1, 1))


def test_a_non_positive_season_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="season must be positive"):
        _tiny(season=0)


def test_a_series_with_no_out_of_sample_rows_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="no out-of-sample rows"):
        _tiny(length=200)


def test_the_backtest_config_uses_a_one_step_scale() -> None:
    # MASE of 1.0 has to mean "exactly naive", so the scale lag is fixed even
    # when the seasonal baseline uses a different period.
    config = _tiny(season=5)
    assert config.backtest_config.seasonal_period == 1
    assert config.season == 5


# -- model selection -------------------------------------------------------


def test_the_baselines_are_always_present() -> None:
    models = build_models(_tiny())
    assert {"naive", "seasonal_naive", "moving_average", "drift"} <= set(models)
    assert models["seasonal_naive"].horizon == 1


def test_the_seasonal_baseline_uses_the_configured_period() -> None:
    models = build_models(_tiny(season=7))
    seasonal = models["seasonal_naive"]
    assert getattr(seasonal, "season", None) == 7


def test_optional_models_are_listed_only_when_importable() -> None:
    models = build_models(_tiny())
    assert ("ets" in models) is statsmodels_available()
    assert ("arima" in models) is statsmodels_available()
    assert ("lightgbm" in models) is lightgbm_available()


# -- the result ------------------------------------------------------------


def test_every_seed_and_model_contributes_one_row(tiny_result: BenchmarkResult) -> None:
    metrics = tiny_result.metrics
    expected = len(_tiny().seeds) * len(tiny_result.models)
    assert len(metrics) == expected
    assert set(metrics["model"]) == set(tiny_result.models)
    assert set(metrics["seed"]) == set(_tiny().seeds)


def test_the_summary_has_a_mean_and_a_std_for_every_metric(tiny_result: BenchmarkResult) -> None:
    summary = tiny_result.summary()
    assert summary.index.tolist() == list(tiny_result.models)
    for metric in ("mase", "mae", "rmse", "smape", "directional_accuracy"):
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_std" in summary.columns


def test_the_standard_deviation_is_defined_for_a_single_seed() -> None:
    # ddof=0 rather than ddof=1, otherwise a one-seed run reports NaN and the
    # table is unreadable.
    result = run_benchmark(_tiny(seeds=(1,)))
    summary = result.summary()
    assert summary["mase_std"].notna().all()


def test_per_fold_mase_covers_every_fold(tiny_result: BenchmarkResult) -> None:
    fold_mase = tiny_result.fold_mase
    folds = tiny_result.n_folds()
    assert folds > 0
    per_model = fold_mase.groupby("model")["fold"].nunique()
    assert (per_model == folds).all()


def test_the_naive_scaled_error_is_close_to_one(tiny_result: BenchmarkResult) -> None:
    # MASE is scaled by the naive error, so the naive model's own score must
    # land near 1. A large deviation would mean the scale and the forecast are
    # measuring different things.
    summary = tiny_result.summary()
    naive = summary.loc["naive", "mase_mean"]
    assert 0.5 < naive < 2.0


# -- the report ------------------------------------------------------------


def test_the_markdown_table_lists_every_model(tiny_result: BenchmarkResult) -> None:
    text = render_markdown(tiny_result)  # type: ignore[arg-type]
    for model in tiny_result.models:
        assert f"| {model} |" in text
    assert "| Model | MASE |" in text
    assert "±" in text


def test_the_verdict_names_the_honesty_rule(tiny_result: BenchmarkResult) -> None:
    text = render_markdown(tiny_result)
    # The threshold rule is stated so a reader can check the verdict themselves.
    assert "standard deviation below the naive mean" in text
    assert ("beats the naive forecast" in text) or ("No model beats" in text)


def test_save_report_writes_the_table_without_a_figure(tmp_path: Path) -> None:
    result = run_benchmark(_tiny())
    table, figure = save_report(result, tmp_path, plot=False)
    assert table.exists()
    assert figure is None
    assert table.read_text(encoding="utf-8").startswith("# ForecastForge benchmark")


def test_the_plot_is_written_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    result = run_benchmark(_tiny())
    table, figure = save_report(result, tmp_path)
    assert table.exists()
    assert figure is not None
    assert figure.exists()
    assert figure.stat().st_size > 0


# -- the command line ------------------------------------------------------


def test_main_writes_a_report_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--seeds",
            "1",
            "--length",
            "400",
            "--initial-train",
            "200",
            "--step",
            "100",
            "--no-plot",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "benchmark_results.md").exists()
    captured = capsys.readouterr()
    assert "wrote" in captured.out
    assert "# ForecastForge benchmark" in captured.out


def test_main_round_trips_its_arguments(tmp_path: Path) -> None:
    main(
        [
            "--seeds",
            "1",
            "2",
            "--length",
            "400",
            "--initial-train",
            "200",
            "--step",
            "100",
            "--season",
            "5",
            "--no-plot",
            "--output-dir",
            str(tmp_path),
        ]
    )
    text = (tmp_path / "benchmark_results.md").read_text(encoding="utf-8")
    assert "seeds 1, 2" in text


def test_the_metrics_frame_is_typed_for_plotting(tiny_result: BenchmarkResult) -> None:
    # The plot reads these columns as floats; an object dtype would break it
    # only at render time, which is exactly when it is least convenient.
    metrics = tiny_result.metrics
    assert isinstance(metrics, pd.DataFrame)
    assert metrics["mase"].dtype == "float64"
