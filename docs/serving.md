# Serving

## Purpose

This document will describe how trained ForecastForge models are served
for inference — the API surface, deployment model, and operational
characteristics of the serving layer.

## Current Status

**Skeleton.** No serving logic exists yet. The
`src/forecastforge/serving/` package and the `docker/app/` container
context are reserved for this purpose but currently contain no
implementation.

## Planned Contents

- **API design** — the interface (e.g. REST/gRPC) through which
  consumers request forecasts, including request/response schemas.
- **Model loading and versioning** — how a specific trained model
  artifact under `models/` is selected and loaded for serving, and how
  model versions are rolled forward and rolled back.
- **Latency and scaling** — expected performance characteristics and how
  the serving layer is expected to scale.
- **Deployment** — how the `docker/app/` container is built and run, and
  how it fits into `docker-compose.yml` once services are implemented.
- **Error handling and fallback behavior** — how the serving layer
  behaves when a model is unavailable or a request is malformed.

This document will be populated once the serving layer is designed and
an ADR has been recorded for the chosen approach.
