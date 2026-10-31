# Model Comparison

## Purpose

How ForecastForge evaluates forecasting models against one another: the
forecaster contract, the walk-forward backtester that produces every score,
the metrics and their definitions, and the benchmark that compares the
shipped models on reproducible data.

## Current Status

**Implemented.** `forecastforge.models` provides the `Forecaster` interface,
four dependency-free baselines, and three optional models (ETS, ARIMA,
LightGBM). `forecastforge.evaluation` provides the metrics, the walk-forward
backtester, and the benchmark that generates
[`docs/assets/benchmark_results.md`](assets/benchmark_results.md).

## The forecaster contract

Every model implements `forecastforge.models.base.Forecaster`:

```python
class Forecaster(ABC):
    name: ClassVar[str] = "forecaster"
    requires_features: ClassVar[bool] = False
    supports_intervals: ClassVar[bool] = False

    def __init__(self, *, horizon: int = 1) -> None: ...
    def fit(self, series: pd.Series, features: pd.DataFrame | None = None) -> Self: ...
    def predict(self) -> np.ndarray: ...
    def predict_interval(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]: ...
    def clone(self) -> Self: ...
```

Three details are deliberate:

- **The horizon is fixed at construction, not passed to `predict`.** A direct
  multi-horizon model fits one estimator per step and needs to know `h` while
  fitting. Passing it at prediction time would force those models to refit or
  to fake it.
- **The base class owns the invariants** — history validation, the
  fitted-before-predict guard, and the output-length check — so a subclass
  writes only `_fit` and `_predict`. This mirrors
  [`Transform`](feature-engineering.md): the shared contract lives in one
  place, so no model can quietly break it.
- **`clone()` returns an unfitted copy.** The backtester needs a fresh model
  per fold, and cloning keeps it from having to know how to rebuild each one.

The target at time `t` is the value at `t + h`. Everything below follows from
that.

## Shipped models

| Name | Class | Needs | Notes |
|---|---|---|---|
| `naive` | `Naive` | — | Last observed value, repeated for every step. |
| `seasonal_naive` | `SeasonalNaive` | — | Value from `season` periods back. |
| `moving_average` | `MovingAverage` | — | Mean of the last `window` observations. |
| `drift` | `Drift` | — | Last value plus the mean change over the window. |
| `ets` | `ETSForecaster` | `[models]` | Holt-Winters via statsmodels. |
| `arima` | `ARIMAForecaster` | `[models]` | ARIMA with optional seasonal terms. |
| `lightgbm` | `LightGBMForecaster` | `[models]` | Direct multi-horizon boosting over a `FeatureSet`. |

The optional models import cleanly without their dependency: the error naming
`pip install -e ".[models]"` is raised when a model is constructed, not when
the package is imported. `statsmodels_available()` and `lightgbm_available()`
let a caller build a model list without catching exceptions.

### Why LightGBM predicts a change, not a level

A tree ensemble can only return values inside the range it was trained on, so
regressing on the raw price produces a forecast pinned to the middle of the
historical range. Measured on a synthetic path, a level-targeted booster was
15% below the last observation after a single horizon.

`LightGBMForecaster` therefore regresses on `y[t+h] - y[t]` and adds the
result to the last observation. That removes the bias by construction: a
booster that learns nothing predicts zero change, which is exactly the naive
forecast.

## Metrics

Defined in `forecastforge.evaluation.metrics`, each a pure function of arrays.
Every one is tested against a value computed by hand rather than against a
restatement of the implementation.

| Metric | Definition | Units |
|---|---|---|
| MAE | `mean(|y - ŷ|)` | target |
| RMSE | `sqrt(mean((y - ŷ)²))` | target |
| sMAPE | `mean(2|y - ŷ| / (|y| + |ŷ|)) × 100` | percent |
| MASE | `MAE / mean(|y[t] - y[t-m]|)` over the training window | unit-free |
| Directional accuracy | share of correctly signed `ŷ - previous` | percent |
| Interval coverage | share of `y` inside `[lower, upper]` | percent |

Two conventions matter enough to state explicitly, because the literature does
not agree on them:

- **sMAPE** uses the `2|e| / (|y| + |ŷ|)` form. Its ceiling is 200%, not 100%.
  A zero denominator is defined as 0 — the forecast was exactly right.
- **MASE** is scaled by the naive error of *the fold's training window*, never
  by anything from the test region. `m = 1` makes 1.0 mean "exactly the naive
  forecast". A larger `m` scales by the seasonal naive error, and on a random
  walk that divides every score by roughly `sqrt(m)`, which makes the whole
  table look better without changing a single model.

Interval coverage is only computed for models that emit intervals; a model
without them is not scored as zero coverage.

## The walk-forward backtester

`WalkForwardBacktester` hands each fold exactly the rows at or before its
cutoff and scores it only on rows after. Configuration:

| Field | Meaning |
|---|---|
| `horizon` | Steps predicted from each origin. |
| `initial_train` | Rows before the first forecast origin. |
| `step` | Rows between consecutive origins. |
| `gap` | Embargo between the training window and the origin. |
| `window` | Rolling training length; `None` means expanding. |
| `seasonal_period` | Naive lag for the MASE scale. |

The result is a tidy frame with one row per `(fold, horizon step, model)`:

```
fold, timestamp, horizon, y_true, y_pred, model, previous
```

`previous` is the actual observation immediately before each target. It is
used only for scoring directional accuracy, and it is always a value the model
was allowed to see.

### Gap semantics

With `gap = g` the last `g` observations before the origin are withheld: the
training window ends at `origin - g`, and the embargoed rows are neither
trained on nor predicted. This follows
`sklearn.model_selection.TimeSeriesSplit`, where a gap exists to stop a model
from fitting on data adjacent to the target.

Four properties are asserted directly in `tests/evaluation/test_backtest.py`
rather than left implicit in the slicing:

1. **The origin does not move.** `gap` shortens the training window; every
   fold forecasts from the same origin regardless of `gap`.
2. **A pair is legal only if it ends by the origin.** A supervised row
   `(i, i + h)` is legal exactly when `i + h + gap <= origin`. The test states
   this with equality on the last legal row.
3. **A gap shorter than the horizon is where naive pairing leaks.** With
   `g < h`, the classic off-by-horizon rule trains on `h - g` targets that
   fall *after* the origin. The audit rejects them, and the test asserts that
   it does.
4. **The scored rows never change.** For any `gap`, the folds score identical
   timestamps and identical outcomes, so models compared across gaps are
   judged on the same targets.

### Leakage tests for the backtester itself

The feature suite tests transforms for causality. The backtester gets the same
treatment, in the same three steps:

- **Structural audit** — `assert_no_leakage` recomputes the fold layout and
  checks that every scored timestamp equals `origin + step` and lies strictly
  after the cutoff.
- **Perturbation and truncation** — multiplying the future by 1.5, or cutting
  the series off just after the first target, must leave the first fold's
  predictions byte-identical.
- **Negative controls** — a `_Peeker` that indexes one row past its history
  must raise `IndexError`, and an `_Oracle` holding the full series must score
  MAE 0.0. If the harness could be fooled, these are what would fool it.

The audit is itself tested: feeding it a result whose timestamps are shifted
back by one row must make it fail.

## The benchmark

```bash
make benchmark
# or
python -m forecastforge.evaluation.benchmark
```

Runs offline on five deterministic 2000-day synthetic paths (75 walk-forward
folds each, horizon 1) and writes
[`assets/benchmark_results.md`](assets/benchmark_results.md) and
[`assets/benchmark_mase.png`](assets/benchmark_mase.png). The optional models
are included only when installed; the report names the models it actually ran.
No model is tuned on the scored folds, and no model is selected after seeing
the scores.

### Measured result

| Model | MASE | MAE | RMSE | sMAPE (%) | Directional accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive | 1.088 ± 0.141 | 1.004 ± 0.295 | 1.289 ± 0.373 | 0.81 ± 0.08 | 0.0 ± 0.0 |
| seasonal_naive | 2.549 ± 0.388 | 2.367 ± 0.751 | 3.014 ± 0.992 | 1.92 ± 0.12 | 48.5 ± 4.5 |
| moving_average | 3.160 ± 0.701 | 2.967 ± 1.133 | 3.746 ± 1.379 | 2.33 ± 0.20 | 46.4 ± 4.9 |
| drift | 1.104 ± 0.164 | 1.022 ± 0.318 | 1.314 ± 0.391 | 0.82 ± 0.09 | 52.0 ± 6.1 |
| ets | 1.089 ± 0.147 | 1.007 ± 0.302 | 1.300 ± 0.384 | 0.81 ± 0.07 | 50.9 ± 4.8 |
| arima | 1.087 ± 0.143 | 1.004 ± 0.297 | 1.288 ± 0.374 | 0.81 ± 0.08 | 54.1 ± 4.7 |
| lightgbm | 1.145 ± 0.172 | 1.062 ± 0.335 | 1.340 ± 0.381 | 0.85 ± 0.10 | 49.3 ± 4.4 |

Values are mean ± standard deviation across seeds.

**No model beats the naive forecast.** ARIMA's 1.087 against naive's 1.088 is
a 0.001 difference on a 0.141 spread: a tie, not a win. ETS and drift land
within noise. The moving average is three times worse, and the seasonal naive
baseline is two and a half times worse because these paths have no weekly
cycle for it to exploit.

That is the expected answer for a random walk, and it is the honest one. A
forecasting library that reported otherwise on synthetic data would be
reporting a bug.

Two observations worth carrying forward:

- **Directional accuracy is 0% for naive, by definition.** A naive forecast
  never predicts a change, so it is credited only when the outcome did not
  change either. Models that do predict a direction score around 50%, which is
  coin-flip territory. Direction is not where the signal is.
- **A low MAE on price does not mean a good forecast.** Naive wins on MAE
  because prices are persistent, not because it predicts anything.

### A note on MASE near 1.09

Naive scores 1.088 rather than exactly 1.0. The scale is the *mean* absolute
change over a fold's training window, while the error is a *single* change at
that fold's forecast point. Averaging those ratios over 75 folds recovers
something close to, but not exactly, 1. The gap is 1.4 standard errors of the
seed mean, so it is noise rather than bias.

## Comparing models safely

- Never tune on the scored folds. Hyperparameters are fixed in
  `build_models` and in the model constructors.
- Report the spread, not just the mean. A 0.001 gap on a 0.14 spread is not a
  result.
- Treat MASE as the headline and MAE as context. On price series MAE rewards
  persistence, which the naive baseline has for free.
- Re-run the benchmark after changing a model. The numbers in this document
  come from `make benchmark` and nothing else.
