# forecastforge.features

Leakage-safe feature engineering: a `FeatureSet` container that keeps the
strict index and target guarantees of `TimeSeries` while allowing NaN in
generated columns and reporting the warm-up period explicitly; a `Transform`
base class whose `apply` verifies the declared output contract on every call;
seven causal transforms; and an ordered `FeaturePipeline` whose later steps
consume columns created by earlier ones.

## FeatureSet

::: forecastforge.features.frame

## Transform base class

::: forecastforge.features.transform

## Built-in transforms

::: forecastforge.features.builtins

## Pipeline

::: forecastforge.features.pipeline
