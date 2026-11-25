# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-25

### Added

- A release workflow (`.github/workflows/release.yml`): pushing a `v*` tag
  builds the sdist and wheel, imports the wheel in a clean virtual
  environment, and attaches both files to the GitHub release.

### Changed

- The README opens with a prominent pre-alpha warning that names what is
  implemented and states that tracking, serving, and monitoring are not.
- The package description in `pyproject.toml` no longer calls the pre-alpha
  project "production-grade"; it names what the library actually provides.
- Prose pass over `README.md` and `docs/`: fewer em-dash constructions, and
  the repository-layout tree in the README is aligned again (its comment
  column had drifted by one space per line).

### Fixed

- SECURITY.md said there were no tagged releases yet and that data ingestion
  and model training do not exist. Both claims are out of date: `v0.1.0` was
  tagged in November, and the data, model, and evaluation layers are
  implemented. The page now describes the supported tag and the actual attack
  surface, including the optional Yahoo Finance connector.
- The package docstring in `src/forecastforge/__init__.py` still listed model
  training and evaluation as planned; it now lists them as implemented.
- ADR 0001 quoted the wheel target's original `packages = ["src"]` value,
  which 0.1.0 corrected to `packages = ["src/forecastforge"]` after the
  broken wheel layout was discovered; the ADR now names the corrected value.
- `scripts/bootstrap.sh` still required `python3.12` on PATH, the same defect
  the Makefile and pre-commit config had already shed. It honors `PYTHON_BIN`
  first and falls back from `python3.12` to `python3`.

## [0.1.0] - 2026-11-02

The first tagged release: repository foundation, the data layer, feature
engineering, forecasting models, and evaluation, with CI, packaging, and the
community health files.

### Added

- Initial repository foundation: `src`-layout Python package skeleton,
  `pyproject.toml` (Hatchling build, Ruff, MyPy, pytest configuration),
  `Makefile`, `.pre-commit-config.yaml`, `.editorconfig`, `.gitignore`,
  `LICENSE` (MIT), and an initial `README.md`.
- Minimal application configuration (`configs/settings.toml`,
  `configs/logging.yaml`) and environment template (`.env.example`).
- Placeholder `docker-compose.yml` for future local service orchestration.
- Directory scaffolding for `docs/`, `pipelines/`, `notebooks/`, `docker/`,
  `scripts/`, `data/{raw,interim,processed,external}/`, `models/`, and
  `mlruns/`.
- The full planned subpackage structure under `src/forecastforge/` (`config`,
  `core`, `common`, `utils`, `data`, `features`, `models`, `evaluation`,
  `tracking`, `serving`, `monitoring`, `pipelines`, `schemas`), each with a
  documented `__init__.py`.
- Documentation scaffolding under `docs/` (`architecture.md`,
  `development.md`, `data-pipeline.md`, `feature-engineering.md`,
  `model-comparison.md`, `serving.md`, `monitoring.md`) and a docs index
  (`docs/README.md`).
- First Architectural Decision Record: `docs/adr/0001-use-src-layout.md`.
- Continuous integration workflow (`.github/workflows/ci.yml`) running Ruff,
  MyPy, and pytest on push and pull request.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant v2.1), `SECURITY.md`.
- Environment-specific configuration overlays: `configs/development.toml`,
  `configs/testing.toml`, `configs/production.toml`.
- Bootstrap scripts for local environment setup: `scripts/bootstrap.sh`
  (POSIX) and `scripts/bootstrap.ps1` (Windows).
- `VERSION` file tracking the current project version.
- A validated `TimeSeries` container (`forecastforge.schemas`) with strict
  structural validation: sorted `DatetimeIndex`, numeric and finite columns,
  optional gap detection.
- The data ingestion layer (`forecastforge.data`): a `DataSource` protocol
  with `BaseDataSource` handling load, slice, validate, and wrap; a
  deterministic, fully-seeded synthetic OHLCV generator (`SyntheticSource`)
  as the default offline dataset; an optional Yahoo Finance connector
  (`YFinanceSource`) behind the `[data]` extra, lazily imported and cached to
  Parquet; and CSV and Parquet file loaders that preserve a `DatetimeIndex`.
- The feature engineering framework (`forecastforge.features`): a `FeatureSet`
  container that keeps the strict index and target guarantees of `TimeSeries`
  while allowing NaN in generated columns and reporting the warm-up period as
  an explicit row count; a `Transform` base class with a checked output
  contract; seven causal transforms (lags, rolling statistics, simple and log
  returns, annualised volatility, RSI, MACD, and calendar features with
  cyclic encodings); and an ordered `FeaturePipeline` whose later steps
  consume columns created by earlier ones. A leakage test suite truncates and
  perturbs the future for every transform and is itself tested against three
  deliberately leaky implementations.
- The forecasting interface (`forecastforge.models`): a `Forecaster` base
  class that fixes the horizon at construction, validates the history, guards
  against predicting before fitting, and checks the output length, so a
  concrete model implements only `_fit` and `_predict`. Four dependency-free
  baselines (`Naive`, `SeasonalNaive`, `MovingAverage`, `Drift`) and three
  optional models behind the `[models]` extra (`ETSForecaster`,
  `ARIMAForecaster` via statsmodels, `LightGBMForecaster`). The optional
  models import cleanly without their dependency and raise a message naming
  the extra when constructed.
- The evaluation layer (`forecastforge.evaluation`): MAE, RMSE, sMAPE, MASE,
  directional accuracy, and interval coverage, each a pure function tested
  against a hand-computed value; a `WalkForwardBacktester` with expanding and
  rolling windows, step size, and a gap embargo, returning a tidy frame of
  one row per (fold, horizon, model); and a reproducible benchmark over five
  2000-day synthetic paths, runnable with `make benchmark`.
- Leakage tests for the backtester itself: a structural audit, a perturbation
  check, a truncation check, and two negative controls (a model that reads
  one row past its history and a model holding the answer key), with the
  audit itself tested against an off-by-one boundary.
- `python -m forecastforge.evaluation.benchmark` and `make benchmark`, which
  run offline and write `docs/assets/benchmark_results.md` and
  `docs/assets/benchmark_mase.png`.
- `tests/models/`, `tests/evaluation/`, and a CI job that installs the
  `[models]` extra so the optional-model tests run instead of skipping.
- A fast benchmark mode (`make benchmark-fast`: one seed, short series, no
  figure) and a CI benchmark smoke job that runs it on every push.
- The application configuration subsystem under `src/forecastforge/config/`:
  `settings.py` (a `pydantic-settings` `Settings` model with a cached
  singleton factory), `environment.py` (deployment environment detection),
  `paths.py` (project-relative path resolution), `runtime.py`
  (execution-context detection), `device.py` (CPU/CUDA/MPS device detection),
  and `seed.py` (deterministic seeding for Python, NumPy, and PyTorch, with
  graceful fallback when the latter two are not installed).
- `pydantic-settings` as a core dependency.
- A Configuration section in `docs/development.md` covering the hierarchy,
  TOML file mechanics, and environment variable overrides.
- A full `tests/config/` suite covering settings loading, environment/TOML/
  env-var precedence, path resolution, device detection fallback, and seed
  reproducibility.
- `examples/01_synthetic_data.py` and `examples/02_features.py` demo scripts,
  with the figures they generate under `docs/assets/`.

### Changed

- CI runs the full gate on Python 3.12 and 3.13, matching the local
  virtualenv instead of testing only the declared floor.
- The `[viz]` extra floor moved to matplotlib 3.9 for
  `Axes.boxplot(tick_labels=)`, which replaced the deprecated `labels`
  keyword.
- `Regime` changes the process rather than the price level: it gains a
  `volatility_shift` field, and `level_shift` is off by default and opt-in.
  Previously `level_shift` injected an additive shock into a single log
  return, producing a one-day jump of `exp(level_shift) - 1` (7 to 11 percent
  in the demo series) and a flat 20-row plateau in any rolling volatility
  series containing that row. Series generated without regimes are bit-for-bit
  unchanged.
- `configs/settings.toml` and its `development.toml` / `testing.toml` /
  `production.toml` overlays restructured to match the fields the new
  `Settings` model loads (`[project]`, `[logging]`, `[paths]`, `[defaults]`),
  removing the unused placeholder `[pipeline]` section.
- `.env.example` expanded with the full set of `FORECASTFORGE_*` variables
  corresponding to `Settings` fields.
- README expanded with badges, project goals, guiding principles, a Mermaid
  architecture diagram, the repository layout, a development workflow
  section, and a roadmap summary.

### Fixed

- Built wheels were not importable. `packages = ["src"]` in `pyproject.toml`
  archived every module under a `src/` prefix, so a non-editable install
  placed the code at `site-packages/src/forecastforge/`. Editable installs
  hid the problem, which is why it survived a green test suite; CI now builds
  the wheel and imports it in a clean virtual environment.
- Eleven URLs pointed at `github.com/forecastforge/forecastforge` instead of
  this repository, breaking the CI badge, the security advisory link, and the
  package metadata links.
- Calendar features read `index.day_of_week` / `index.day_of_year` instead of
  the `dayofweek` / `dayofyear` spellings that pandas 3 removed, so the
  calendar transform keeps working when pandas 3 is installed.
- Line endings normalised to LF via `.gitattributes`, so
  `ruff format --check` no longer fails on a Windows checkout using
  `core.autocrlf=true`.
- Optional numpy/PyTorch imports moved into a documented helper
  (`forecastforge.config._optional`) instead of lazy imports inside
  functions, removing three PLC0415 violations and the mypy
  `import-not-found` errors they caused.
- The coverage gate enforced nothing (`fail_under = 0`); it is now 85, with
  total coverage at 98 percent.
- Pre-commit pinned Ruff v0.5.5 while the lint config still referenced the
  removed ANN101/ANN102 rules, producing warnings on every run.
- The Makefile and pre-commit config hardcoded a `python3.12` interpreter;
  both now detect the virtualenv interpreter and fall back to `python`.

[Unreleased]: https://github.com/Abolfazlrwm/forecastforge/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Abolfazlrwm/forecastforge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Abolfazlrwm/forecastforge/releases/tag/v0.1.0
