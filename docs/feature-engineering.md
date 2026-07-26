# Feature Engineering

## Purpose

This document will describe ForecastForge's feature engineering
framework: how processed data is transformed into the features consumed
by forecasting models, and how those transformations are defined,
tested, and reused.

## Current Status

**Skeleton.** No feature engineering logic exists yet. The
`src/forecastforge/features/` package is reserved for this purpose but
currently contains no implementation.

## Planned Contents

- **Feature definition model** — how a feature is declared (inputs,
  transformation logic, output schema) and how definitions are kept
  reusable across models.
- **Time-series-specific concerns** — handling of lags, rolling windows,
  seasonality, and calendar features without introducing data leakage.
- **Feature validation** — how feature outputs are checked against the
  schemas in `src/forecastforge/schemas/` before being used for training
  or inference.
- **Feature versioning** — how changes to feature logic are tracked so
  that model results remain reproducible.
- **Train/serve consistency** — how the same feature logic is guaranteed
  to run identically at training time and at inference time.
- **Performance considerations** — expectations around computing
  features at batch versus low-latency online scale.

This document will be populated once the feature engineering framework
is designed and an ADR has been recorded for the chosen approach.
