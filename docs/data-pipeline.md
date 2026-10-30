# Data Pipeline

## Purpose

How ForecastForge ingests, validates, and exposes data — from a raw vendor
feed or a deterministic generator through to the validated container that
feature engineering consumes.

## Current Status

**Implemented.** `src/forecastforge/data/` provides the source interface, a
deterministic offline generator, file loaders, and an optional network
connector. Structural validation lives in `src/forecastforge/schemas/`.

## The data contract

Every stage hands forward a
[`TimeSeries`](../src/forecastforge/schemas/time_series.py): a
`pandas.DataFrame` with a sorted, unique `DatetimeIndex`, a numeric target
column, and any declared numeric covariates. `validate_frame` enforces this
once at the boundary rather than trusting each downstream consumer to
re-check it:

- the index must be a `DatetimeIndex`, free of `NaT`, unique, and sorted;
- the target and every declared covariate must exist, be numeric, and contain
  no NaN or infinite values;
- optional gap detection (`find_gaps`, `require_complete=True`) reports
  contiguous runs of missing periods against an expected frequency.

Gaps are **not** rejected by default: real market data legitimately skips
weekends, holidays, and trading halts, so rejecting them would make the
library unusable for its primary use case. `find_gaps` makes them visible
instead.

NaN is rejected outright, because silently propagating it into features and
metrics produces quietly wrong numbers that are far more expensive to debug
than an explicit failure. Feature engineering therefore works in
[`FeatureSet`](feature-engineering.md), which permits NaN in *generated*
columns only.

## Sources

`DataSource` is a runtime-checkable protocol; `BaseDataSource` implements the
shared orchestration so a concrete source only decides *how* to obtain a raw
frame:

```
_load_frame()  ->  slice to [start, end]  ->  validate  ->  TimeSeries
```

| Source | Extra | Behaviour |
|---|---|---|
| `SyntheticSource` | none | Deterministic OHLCV from a fixed seed; the default offline dataset |
| `read_csv` / `read_parquet` | `[data]` for Parquet | Loaders that preserve a `DatetimeIndex` |
| `YFinanceSource` | `[data]` | Optional network connector, lazily imported, cached to `data/raw` |

`SyntheticSource` is the backbone of the test suite and every example. It
generates a log-price random walk with constant drift, sinusoidal
seasonality, gaussian noise, and optional structural regimes — so the same
seed yields the identical series on any machine, in CI, and on a
contributor's laptop.

### Regimes

A `Regime` marks a structural break at a given row, inclusive:

| Field | Effect from `at` onward |
|---|---|
| `drift_shift` | Additive change to the per-period log-return |
| `volatility_shift` | Additive change to the per-period log-return standard deviation |
| `level_shift` | Opt-in one-period jump of `exp(level_shift) - 1` in the close price |

By default a regime changes the **process, not the price level**. That choice
is deliberate:

- A sudden 8% overnight move is indistinguishable from a data error, and a
  reader of the demo plot reasonably concludes the generator is broken.
- A single outlier return stays inside every rolling volatility window that
  contains it for the entire window length, so a 20-day volatility series
  shows a flat 20-row plateau. That plateau is an artefact of the injected
  jump, not a property of the process, and it silently contaminates any
  volatility feature computed from the series.

The volatility of each period is therefore a vector rather than a scalar: a
regime scales the noise, and the price path stays continuous. `level_shift`
remains available for callers who explicitly want a discontinuity — for
example to test how a model copes with a gap.

The resulting per-period standard deviation must stay non-negative; a
`volatility_shift` that would drive it below zero raises `DataError` rather
than silently producing NaN prices.

## Staging conventions

| Directory | Contents | Mutability |
|---|---|---|
| `data/raw/` | As-received vendor data and connector caches | Immutable |
| `data/interim/` | Partially processed, reproducible from raw | Regenerable |
| `data/processed/` | Model-ready datasets | Regenerable |
| `data/external/` | Third-party reference data | Immutable |

Only the directory structure is tracked; contents are git-ignored.

## Reproducibility

Every generated dataset carries its provenance in `TimeSeriesMetadata`:
`symbol`, `source`, `frequency`, and a free-form `attributes` mapping that
includes the generator seed. Two runs with the same metadata and seed are
byte-identical, which is what allows the backtests in later phases to be
compared across machines.

## Not implemented yet

- Incremental or streaming ingestion.
- Schema versioning for on-disk datasets.
- A catalog of external data providers beyond Yahoo Finance.
- Automated data-quality reports beyond structural validation.
