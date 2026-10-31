"""Forecasting models and their shared interface.

Every model implements :class:`~forecastforge.models.base.Forecaster`: fit on
a history, predict the next ``horizon`` values. The horizon is fixed at
construction because a direct multi-horizon model builds one estimator per
step and needs to know ``h`` while fitting.

Available without any optional dependency:

* :class:`~forecastforge.models.baselines.Naive`
* :class:`~forecastforge.models.baselines.SeasonalNaive`
* :class:`~forecastforge.models.baselines.MovingAverage`
* :class:`~forecastforge.models.baselines.Drift`

Optional, behind the ``models`` extra:

* :class:`~forecastforge.models.statistical.ETSForecaster` (statsmodels)
* :class:`~forecastforge.models.statistical.ARIMAForecaster` (statsmodels)
* :class:`~forecastforge.models.gbm.LightGBMForecaster` (lightgbm)

The optional models are importable without their dependency installed; the
error is raised when you construct one, and it names the extra to install.
"""

from __future__ import annotations

from forecastforge.models.base import Forecaster
from forecastforge.models.baselines import Drift, MovingAverage, Naive, SeasonalNaive

__all__ = [
    "Drift",
    "Forecaster",
    "MovingAverage",
    "Naive",
    "SeasonalNaive",
]
