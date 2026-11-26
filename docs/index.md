---
title: ForecastForge
description: Leakage-safe time-series forecasting, walk-forward evaluation, and a reproducible benchmark.
---

# ForecastForge

ForecastForge is an open-source Python library for time-series forecasting
work that has to be trustworthy: features that cannot leak the future, a
backtester whose fold boundaries are audited, metrics computed against
hand-computed values, and a benchmark that reports ties as ties.

> [!WARNING]
> **ForecastForge is pre-alpha.** Experiment tracking, a serving layer, and
> monitoring are not implemented yet. What is here — the data layer, feature
> engineering, models, and evaluation — is complete and tested.

## What works today

The pipeline runs end to end on fully offline data: generate a series,
engineer features, fit a model, backtest it, and score it.

- **Data ingestion** (`forecastforge.data`): a deterministic synthetic OHLCV
  generator, CSV and Parquet loaders that preserve a `DatetimeIndex`, and an
  optional Yahoo Finance connector behind the `[data]` extra.
- **Feature engineering** (`forecastforge.features`): a `FeatureSet` container
  with an explicit warm-up count, seven causal transforms (lags, rolling
  statistics, returns, volatility, RSI, MACD, calendar features), and an
  ordered `FeaturePipeline`. Every transform is covered by a leakage test
  suite that truncates and perturbs the future.
- **Models** (`forecastforge.models`): one `fit`/`predict` interface with the
  shared validation in the base class; four dependency-free baselines, plus
  optional ETS, ARIMA, and LightGBM forecasters behind the `[models]` extra.
- **Evaluation** (`forecastforge.evaluation`): MAE, RMSE, sMAPE, MASE,
  directional accuracy, and interval coverage; a walk-forward backtester with
  expanding and rolling windows, step size, and a gap embargo; and a
  reproducible benchmark (`make benchmark`).

## Quickstart

Install the package and the optional plotting dependency:

```bash
pip install -e ".[viz]"
```

Run the backtest demo, which needs nothing beyond the core install:

```python
from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.evaluation import BacktestConfig, WalkForwardBacktester
from forecastforge.models import Drift, Naive

series = SyntheticSource(config=SyntheticConfig(seed=3, length=600)).fetch(symbol="SPY")
result = WalkForwardBacktester(BacktestConfig(initial_train=252, horizon=1)).run(
    series.target_series, {"naive": Naive(), "drift": Drift(window=20)}
)
print(result.summary())
```

The [guides](architecture.md) walk through each stage, and the
[API reference](api/overview.md) documents every public symbol.

## Project status

A roadmap, honest about what exists and what does not, lives in the
repository README. Phases 0-3 (repository health, data, features, models and
evaluation) are complete; the CLI, tracking, serving, and monitoring phases
are planned. Significant design decisions are recorded as ADRs.

## Where to go next

- [Architecture](architecture.md) — the component boundaries and which are real.
- [Data pipeline](data-pipeline.md) — sources, loaders, and validation.
- [Feature engineering](feature-engineering.md) — the transform contract and leakage tests.
- [Model comparison](model-comparison.md) — metrics, backtesting, and the benchmark.
- [Development](development.md) — local setup, tooling, and conventions.
