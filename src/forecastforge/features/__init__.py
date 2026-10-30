"""Leakage-safe feature engineering.

The pipeline turns a validated
:class:`~forecastforge.schemas.time_series.TimeSeries` into a
:class:`~forecastforge.features.frame.FeatureSet`: the same timeline and
target, plus derived columns that may still contain NaN during their warm-up
period.

Two guarantees hold for every transform in this package:

* **Causality** — a feature at row *t* depends only on rows ``<= t``, so a
  walk-forward backtest never sees information the model would not have had.
* **Declared output** — each transform states the columns it produces, and
  :meth:`~forecastforge.features.transform.Transform.apply` verifies it.

Example::

    from forecastforge.data import SyntheticSource
    from forecastforge.features import (
        CalendarFeatures,
        FeaturePipeline,
        Lag,
        Returns,
        RollingStat,
    )

    series = SyntheticSource().fetch(symbol="SPY")
    features = FeaturePipeline(
        [
            Lag([1, 5, 10]),
            Returns(periods=1, log=True),
            RollingStat(window=20, stat="std", column="close_logret1"),
            CalendarFeatures(["dayofweek", "month_cyclic"]),
        ]
    ).apply(series)

    print(features.warmup)          # rows that are not usable yet
    model_input = features.drop_warmup()
"""

from __future__ import annotations

from forecastforge.features.builtins import (
    MACD,
    RSI,
    CalendarFeatures,
    Lag,
    Returns,
    RollingStat,
    Volatility,
)
from forecastforge.features.frame import FeatureSet
from forecastforge.features.pipeline import FeaturePipeline
from forecastforge.features.transform import Transform

__all__ = [
    "MACD",
    "RSI",
    "CalendarFeatures",
    "FeaturePipeline",
    "FeatureSet",
    "Lag",
    "Returns",
    "RollingStat",
    "Transform",
    "Volatility",
]
