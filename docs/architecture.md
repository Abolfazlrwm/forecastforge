# Architecture

## Purpose

ForecastForge's system architecture: the major components, their
responsibilities, and how they interact — from data ingestion through to
model serving and monitoring.

## Current Status

**Partially implemented.** The data and feature stages are built and tested;
the modelling, tracking, serving, and monitoring stages are not.

| Stage | Package | Status |
|---|---|---|
| Configuration | `config` | Implemented |
| Core types and errors | `core` | Implemented |
| Data contracts | `schemas` | Implemented |
| Ingestion | `data` | Implemented |
| Feature engineering | `features` | Implemented |
| Models | `models` | Placeholder |
| Evaluation | `evaluation` | Placeholder |
| Tracking | `tracking` | Placeholder |
| Serving | `serving` | Placeholder |
| Monitoring | `monitoring` | Placeholder |
| Orchestration | `pipelines` | Placeholder |

The root [README](../README.md) carries the target-architecture diagram, with
the implemented stages marked. Sections below will be filled in as each
remaining stage is designed.

## Component boundaries

- **`config`** owns settings resolution and is the only place environment
  variables are read. Nothing else parses the environment.
- **`core`** holds the exception hierarchy. Every error the package raises
  deliberately derives from `ForecastForgeError`, so callers can catch
  library failures without also catching unrelated built-ins.
- **`schemas`** owns the data contract. Validation happens once, at the
  boundary, and is never re-derived downstream.
- **`data`** turns a symbol and a window into a validated `TimeSeries`.
  Sources are interchangeable behind the `DataSource` protocol.
- **`features`** turns a `TimeSeries` into a `FeatureSet` — the same timeline
  and target plus derived columns, which may contain NaN during warm-up.
  Transforms are causal and stateless; causality is verified by tests, not
  assumed.

## Planned Contents

- **System context diagram** — ForecastForge's boundaries and its external
  dependencies (data sources, tracking backends, consumers of forecasts).
- **Data flow** — how data moves from raw sources through interim and
  processed stages into features, and from trained models into served
  predictions.
- **Pipeline orchestration model** — how pipeline stages are composed,
  scheduled, and retried, and what orchestration tooling (if any) is
  adopted, with the decision recorded as an ADR.
- **Deployment topology** — how components map onto the containers defined
  under `docker/` once those are implemented.
- **Cross-cutting concerns** — logging, error handling, and versioning
  conventions used across components.

See [docs/adr/](adr/) for the architectural decisions already made.
