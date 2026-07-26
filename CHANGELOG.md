# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded `src/forecastforge/` into its full planned subpackage
  structure (`config`, `core`, `common`, `utils`, `data`, `features`,
  `models`, `evaluation`, `tracking`, `serving`, `monitoring`,
  `pipelines`, `schemas`), each with a documented, empty `__init__.py`.
- Documentation scaffolding under `docs/` (`architecture.md`,
  `development.md`, `data-pipeline.md`, `feature-engineering.md`,
  `model-comparison.md`, `serving.md`, `monitoring.md`) and a docs index
  (`docs/README.md`).
- First Architectural Decision Record: `docs/adr/0001-use-src-layout.md`.
- Continuous integration workflow (`.github/workflows/ci.yml`) running
  Ruff, MyPy, and pytest on push and pull request.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant v2.1), `SECURITY.md`.
- Environment-specific configuration overlays:
  `configs/development.toml`, `configs/testing.toml`,
  `configs/production.toml`.
- Placeholder Docker build contexts: `docker/app/Dockerfile`,
  `docker/mlflow/Dockerfile`, `docker/monitor/Dockerfile`.
- Bootstrap scripts for local environment setup:
  `scripts/bootstrap.sh` (POSIX) and `scripts/bootstrap.ps1` (Windows).
- `VERSION` file tracking the current project version.

### Changed

- Expanded `README.md` with badges, project goals, guiding principles, a
  Mermaid architecture diagram, an updated repository layout, a
  development workflow section, and a roadmap summary.

## [0.1.0] - 2026-07-26

### Added

- Initial repository foundation: `src`-layout Python package skeleton,
  `pyproject.toml` (Hatchling build, Ruff, MyPy, pytest configuration),
  `Makefile`, `.pre-commit-config.yaml`, `.editorconfig`, `.gitignore`,
  `LICENSE` (MIT), and an initial `README.md`.
- Minimal application configuration (`configs/settings.toml`,
  `configs/logging.yaml`) and environment template (`.env.example`).
- Placeholder `docker-compose.yml` for future local service
  orchestration.
- Directory scaffolding for `docs/`, `pipelines/`, `notebooks/`,
  `docker/`, `scripts/`, `data/{raw,interim,processed,external}/`,
  `models/`, and `mlruns/`.

[Unreleased]: https://github.com/forecastforge/forecastforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/forecastforge/forecastforge/releases/tag/v0.1.0
