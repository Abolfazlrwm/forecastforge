"""Forecast evaluation: metrics and walk-forward backtesting.

* :mod:`forecastforge.evaluation.metrics` — MAE, RMSE, sMAPE, MASE,
  directional accuracy, and interval coverage, each a pure function.
* :mod:`forecastforge.evaluation.backtest` — the leakage-safe walk-forward
  backtester that produces the tidy prediction frame every report is built
  from.
"""

from __future__ import annotations

from forecastforge.evaluation.backtest import (
    BacktestConfig,
    BacktestResult,
    Fold,
    WalkForwardBacktester,
)
from forecastforge.evaluation.metrics import (
    MetricSet,
    directional_accuracy,
    evaluate,
    interval_coverage,
    mae,
    mase,
    mase_scale,
    rmse,
    smape,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Fold",
    "MetricSet",
    "WalkForwardBacktester",
    "directional_accuracy",
    "evaluate",
    "interval_coverage",
    "mae",
    "mase",
    "mase_scale",
    "rmse",
    "smape",
]
