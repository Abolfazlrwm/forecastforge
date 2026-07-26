# Data Pipeline

## Purpose

This document will describe how ForecastForge ingests, validates, and
processes data — from raw source data through to the interim and
processed datasets used by feature engineering and modeling.

## Current Status

**Skeleton.** No data ingestion or processing logic exists yet. The
`src/forecastforge/data/` package and the `data/{raw,interim,processed,external}`
directories are reserved for this purpose but currently contain no
implementation.

## Planned Contents

- **Data sources** — the categories of data ForecastForge is designed to
  ingest, and the connectors/loaders provided for each.
- **Data contracts** — the schemas (see `src/forecastforge/schemas/`)
  that define what a valid raw, interim, and processed dataset looks
  like, and how contract violations are surfaced.
- **Staging conventions** — the rules governing what belongs in
  `data/raw/` (immutable, as-received), `data/interim/` (partially
  processed, reproducible from raw), `data/processed/` (model-ready),
  and `data/external/` (third-party reference data).
- **Validation and quality checks** — how incoming data is validated
  before being promoted between stages.
- **Idempotency and reproducibility** — how pipeline runs are made safe
  to re-run without producing inconsistent state.
- **Failure handling** — how partial or malformed ingestion runs are
  detected and handled.

This document will be populated once the ingestion framework is
designed and an ADR has been recorded for the chosen approach.
