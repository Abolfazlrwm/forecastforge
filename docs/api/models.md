# forecastforge.models

Every forecaster implements the same interface: fit on a history, predict the
next `horizon` values. The horizon is fixed at construction, and the base
class owns validation, the fitted-before-predict guard, and the output-length
check, so a concrete model implements only `_fit` and `_predict`.

## Base interface

::: forecastforge.models.base

## Dependency-free baselines

::: forecastforge.models.baselines

## Optional statistical models (statsmodels)

::: forecastforge.models.statistical

## Optional gradient-boosted model (LightGBM)

::: forecastforge.models.gbm
