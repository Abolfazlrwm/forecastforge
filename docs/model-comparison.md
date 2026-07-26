# Model Comparison

## Purpose

This document will describe how candidate forecasting models are
evaluated against one another within ForecastForge — the methodology,
metrics, and criteria used to select a model for a given use case.

## Current Status

**Skeleton.** No models, evaluation harnesses, or comparison tooling
exist yet. The `src/forecastforge/models/` and
`src/forecastforge/evaluation/` packages are reserved for this purpose
but currently contain no implementation.

## Planned Contents

- **Candidate model classes** — the categories of forecasting approaches
  ForecastForge is designed to support (e.g. statistical, machine
  learning, deep learning), once selected.
- **Evaluation methodology** — backtesting strategy, train/validation/
  test splitting appropriate to time series, and how leakage is avoided.
- **Metrics** — the forecasting accuracy and calibration metrics used
  for comparison, and why they were chosen.
- **Comparison reporting** — how comparison results will be recorded,
  visualized, and linked to experiment tracking (see `docs/adr/` for the
  tracking backend decision once made).
- **Selection criteria** — how a "winning" model is chosen beyond raw
  accuracy (e.g. latency, interpretability, resource cost).

This document will be populated once the evaluation framework is
designed and an ADR has been recorded for the chosen approach.
