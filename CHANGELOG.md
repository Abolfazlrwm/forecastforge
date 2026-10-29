# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Implemented the application configuration subsystem under
  `src/forecastforge/config/`: `settings.py` (a `pydantic-settings`
  `Settings` model with a cached singleton factory), `environment.py`
  (deployment environment detection), `paths.py` (project-relative path
  resolution), `runtime.py` (execution-context detection), `device.py`
  (CPU/CUDA/MPS device detection), and `seed.py` (deterministic seeding
  for Python, NumPy, and PyTorch, with graceful fallback when the
  latter two are not installed).
- Added `pydantic-settings` as a core dependency.
- Added a `docs/development.md` "Configuration" section documenting the
  configuration hierarchy, TOML file mechanics, and environment
  variable overrides.
- Added a full `tests/config/` suite covering settings loading,
  environment/TOML/env-var precedence, path resolution, device
  detection fallback, and seed reproducibility.

### Changed

- Restructured `configs/settings.toml` and its `development.toml` /
  `testing.toml` / `production.toml` overlays to match the fields the
  new `Settings` model loads (`[project]`, `[logging]`, `[paths]`,
  `[defaults]`), removing the unused placeholder `[pipeline]` section.
- Expanded `.env.example` with the full set of `FORECASTFORGE_*`
  variables corresponding to `Settings` fields.

### Fixed

- Built wheels were not importable. `packages = ["src"]` in
  `pyproject.toml` archived every module under a `src/` prefix, so a
  non-editable install placed the code at
  `site-packages/src/forecastforge/`. Editable installs hid the problem,
  which is why it survived a green test suite; CI now builds the wheel
  and imports it in a clean virtual environment.
- Eleven URLs pointed at `github.com/forecastforge/forecastforge`
  instead of this repository, breaking the CI badge, the security
  advisory link, and the package metadata links.
- Line endings were normalised to LF via `.gitattributes`, so
  `ruff format --check` no longer fails on a Windows checkout using
  `core.autocrlf=true`.
- Optional numpy/PyTorch imports moved into a documented helper
  (`forecastforge.config._optional`) instead of lazy imports inside
  functions, removing three PLC0415 violations and the mypy
  `import-not-found` errors they caused.
- The coverage gate enforced nothing (`fail_under = 0`); it is now 85,
  with total coverage at 98%.
- Pre-commit pinned Ruff v0.5.5 while the lint config still referenced
  the removed ANN101/ANN102 rules, producing warnings on every run.

### Added (previous phase)

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

[Unreleased]: https://github.com/Abolfazlrwm/forecastforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Abolfazlrwm/forecastforge/releases/tag/v0.1.0
