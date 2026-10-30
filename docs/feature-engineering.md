# Feature Engineering

## Purpose

Turns a validated
[`TimeSeries`](../src/forecastforge/schemas/time_series.py) into the numeric
columns a forecasting model consumes, without ever letting information from
the future into the past.

The package lives in `src/forecastforge/features/`:

| Module | Role |
|---|---|
| `frame.py` | `FeatureSet` — the container features are built in |
| `transform.py` | `Transform` — the base class and its contract |
| `builtins.py` | The seven built-in transforms |
| `pipeline.py` | `FeaturePipeline` — ordered composition |

## Why features need their own container

`TimeSeries` rejects NaN on purpose: every downstream consumer should be able
to assume clean numbers. Feature engineering cannot honour that. A 20-day
rolling mean has no value for the first 19 rows, and every way of hiding that
fact is a bug in disguise:

- **Forward-filling** invents observations that never existed.
- **Dropping rows silently** shifts the timeline, which silently changes what
  a walk-forward split means.
- **Substituting zero** is indistinguishable from a real zero and biases every
  model that sees it.

`FeatureSet` therefore keeps the strict guarantees that still apply — sorted
unique `DatetimeIndex`, numeric finite non-null target — allows NaN in
generated columns, and makes the warm-up an explicit number:

```python
features = FeaturePipeline([Lag(5), MACD()]).apply(series)
features.warmup  # 33 — leading rows that are not usable yet
features.is_complete  # False
usable = features.drop_warmup()  # 471 rows, no NaN
features.to_time_series()  # raises while any NaN remains
```

`warmup` is a *prefix length*: the number of leading rows that still contain a
NaN. If a NaN appears after a complete row, that is a data inconsistency
rather than a warm-up, and `n_incomplete_rows` reports the true count so the
two cases cannot be confused.

## The transform contract

Every transform implements two methods:

```python
class Transform(ABC):
    def output_columns(self, features: FeatureSet) -> tuple[str, ...]: ...
    def compute(self, features: FeatureSet) -> pd.DataFrame: ...
```

`compute` returns **only** the new columns, aligned to the input index.
`apply` is implemented once on the base class: it calls `compute`, checks that
the produced column names match the declaration exactly, and appends them.
Two consequences follow:

- A transform that lies about its output fails at the moment it is applied,
  not three stages later as a confusing `KeyError`.
- Appending two columns with the same name is rejected rather than silently
  producing a frame with duplicate labels.

Transforms are **stateless**. Nothing is fitted, so the same input always
produces the same output. Anything that must learn from data (scalers,
encoders) belongs to the model pipeline, where an explicit fit/transform split
makes leakage auditable.

## The built-in transforms

| Transform | Columns | Warm-up | Notes |
|---|---|---|---|
| `Lag(periods)` | `{col}_lag{p}` | `max(periods)` | The series' own memory |
| `RollingStat(window, stat)` | `{col}_rolling_{stat}_{window}` | `window - 1` | Trailing only; `mean`, `std`, `min`, `max` |
| `Returns(periods, log)` | `{col}_ret{p}` / `{col}_logret{p}` | `periods` | Scale-free, unlike raw price |
| `Volatility(window)` | `{col}_vol{window}` | `window` | Rolling std of log returns, annualised |
| `RSI(window)` | `{col}_rsi{window}` | `window` | Wilder smoothing as an EMA with `alpha=1/window` |
| `MACD(fast, slow, signal)` | `{col}_macd`, `_signal`, `_hist` | `(slow-1) + (signal-1)` | 33 with the defaults |
| `CalendarFeatures(components)` | `dt_*` | 0 | Includes sine/cosine cyclic encodings |

Raw price levels are deliberately **not** emitted as features. They are
non-stationary, and a model given them will learn a trend that does not
extrapolate.

`RollingStat` defaults `min_periods` to `window` rather than `1`. A
half-filled window produces a statistic computed from too little data, which
looks like a real value and is not; requiring a full window keeps the warm-up
honest.

## Leakage: what is guaranteed and how

Causality is the property that makes a walk-forward backtest mean anything.
A feature at row *t* may depend only on rows `<= t`. Three common mistakes all
look reasonable and all leak:

- `shift(-1)` — reading tomorrow's value.
- A centred rolling window — averaging across the present.
- A statistic over the whole sample, such as a global z-score — the past is
  normalised with information from the future.

The framework cannot enforce causality statically, so it is **verified**.
`tests/features/test_leakage.py` applies two independent checks to every
built-in transform:

1. **Truncation.** Recompute on the series cut off at row `cut`; every value
   at or before `cut` must be identical.
2. **Perturbation.** Replace everything after `cut` with a large alternating
   shock and recompute; the prefix must not move.

Truncation alone can be fooled by a transform that reads the future without
changing magnitude, and perturbation alone can be fooled by one that only
depends on series length, so both are needed. The harness is itself tested
against three deliberately leaky transforms — a forward shift, a centred
window, and a global z-score — so a regression in the harness fails loudly
instead of quietly weakening the guarantee. A further test fails if a
transform class is added without a leakage case.

## Composing pipelines

`FeaturePipeline` is an ordered list, not a dependency graph. Later steps see
columns created by earlier ones, which is what makes "volatility of returns"
expressible:

```python
pipeline = FeaturePipeline(
    [
        Returns(periods=1, log=True),  # close_logret1
        RollingStat(window=20, stat="std", column="close_logret1"),
        CalendarFeatures(["dayofweek", "month_cyclic"]),
    ]
)
```

Ordering is the caller's responsibility on purpose: a pipeline that reorders
itself is much harder to review for leakage. Referencing a column that does
not exist yet raises `DataValidationError` naming the missing column.

## Trying it

```bash
pip install -e ".[viz]"
python examples/02_features.py
```

The script builds thirteen features over the deterministic 504-day synthetic
series, prints each column's first valid date and the resulting warm-up (33
rows, 93% of the series usable), and writes
`docs/assets/feature_overview.png`.

## Not implemented yet

- Feature selection and importance ranking.
- Fitted transformations (scalers, target encoders) with an explicit
  fit/transform split.
- Caching and incremental computation for streaming inference.
- Feature versioning for reproducibility across model runs.

These belong with the modelling phase, where a fitted pipeline has something
to fit against.
