# Development Guide

## Purpose

This document will be the canonical guide for setting up a local
development environment, running the quality gates, and understanding
the day-to-day contributor workflow for ForecastForge.

## Current Status

**Skeleton.** The core developer workflow already exists and is
summarized in the root [README.md](../README.md) and
[CONTRIBUTING.md](../CONTRIBUTING.md) (environment setup, `make`
targets, pre-commit, Conventional Commits). This document will expand on
that summary with more detail once there is enough day-to-day workflow
(e.g. running pipelines locally, working with sample data) to document.

## Planned Contents

- **Environment setup** — detailed walkthrough of `scripts/bootstrap.sh`
  / `scripts/bootstrap.ps1`, virtual environment management, and
  troubleshooting common setup issues.
- **Configuration** — how `configs/*.toml` and `.env` interact, and how
  to select an environment profile (development / testing / production).
- **Running the quality gates** — detailed usage of `make lint`,
  `make format`, `make typecheck`, `make test`, and `make run-precommit`,
  including how to interpret common failures.
- **Working with data locally** — once ingestion exists, how to point
  ForecastForge at sample or local datasets under `data/`.
- **Debugging and profiling** — recommended tools and workflows once
  pipeline and model code exist.
- **Release process** — how versions are cut, how `CHANGELOG.md` is
  maintained, and how releases are published.

This document will grow alongside the codebase; sections above will be
filled in as the underlying functionality is implemented.
