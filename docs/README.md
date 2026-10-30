# ForecastForge Documentation

This directory contains the documentation set for ForecastForge. It is
currently a **scaffold**: each document below describes the purpose and
planned contents of its topic area, not a finished specification.
Documents will be filled in incrementally as the corresponding
functionality is designed and implemented.

## Status

All documents in this directory are in **skeleton** status unless noted
otherwise. "Skeleton" means: purpose and planned scope are defined, but
implementation details, APIs, and diagrams describing built functionality
do not yet exist because the functionality itself does not yet exist.
`data-pipeline.md`, `feature-engineering.md`, and the configuration section
of `development.md` document functionality that is implemented, tested, and
runnable.

## Index

| Document | Topic | Status |
|---|---|---|
| [architecture.md](architecture.md) | System architecture and component boundaries | Partially implemented |
| [development.md](development.md) | Local development setup and workflow | Partially implemented |
| [data-pipeline.md](data-pipeline.md) | Data ingestion and processing design | Implemented |
| [feature-engineering.md](feature-engineering.md) | Feature engineering framework design | Implemented |
| [model-comparison.md](model-comparison.md) | Model selection and comparison methodology | Skeleton |
| [serving.md](serving.md) | Model serving and inference API design | Skeleton |
| [monitoring.md](monitoring.md) | Observability and monitoring design | Skeleton |
| [adr/](adr/) | Architectural Decision Records | Active |

## Conventions

- Documentation is written in Markdown and lives alongside the code it
  describes.
- Significant, hard-to-reverse design choices are recorded as an ADR in
  [`adr/`](adr/) rather than buried in prose elsewhere.
- Each topic document follows the same structure: **Purpose**,
  **Current Status**, **Planned Contents** — so readers can quickly tell
  what is real versus what is planned.
