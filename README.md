# ForecastForge

[![CI](https://github.com/Abolfazlrwm/forecastforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Abolfazlrwm/forecastforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2A6DB2.svg)](https://mypy-lang.org/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org/en/v1.0.0/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

> **Status: Pre-Alpha — foundation, data layer, feature engineering, models,
> and evaluation.**
> **ForecastForge does not yet implement serving or monitoring.** What exists
> and is tested: the core exception hierarchy, a validated ``TimeSeries``
> container with structural validation, a fully-offline data ingestion layer
> (synthetic source, file loaders, and optional Yahoo Finance connector), a
> leakage-safe feature engineering framework, a forecaster interface with
> four baselines and three optional models, and a walk-forward backtester with
> a reproducible benchmark.

## Project Goals

ForecastForge aims to be an open-source, production-grade platform for
building, evaluating, and operating time-series forecasting systems at
scale. Concretely, the project is working toward:

- A **composable pipeline** that takes raw data through ingestion,
  feature engineering, model training/evaluation, and serving as
  independently testable stages.
- A **reproducible experimentation workflow**, where every training run
  is tracked, versioned, and comparable against prior runs.
- A **maintainable codebase** that a distributed group of open-source
  contributors — not just its original authors — can safely extend.
- A **deployable serving layer** that exposes trained models through a
  documented, versioned API.
- **Built-in observability**, so that data drift and model degradation
  in production are detected rather than discovered by users.

## Guiding Principles

- **Reproducibility** — deterministic environments, pinned tooling, and
  versioned experiment artifacts.
- **Observability** — structured logging and experiment tracking as
  first-class citizens, not afterthoughts.
- **Maintainability** — strict linting, static typing, automated
  testing, and a documented architectural decision record (ADR) trail
  so the project remains legible as it grows and as contributors
  change.
- **Composability** — a pipeline-oriented architecture where ingestion,
  feature engineering, modeling, and serving are independently testable
  and independently deployable stages.
- **Openness** — a governance and contribution model (see
  [CONTRIBUTING.md](CONTRIBUTING.md) and
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)) that welcomes external
  maintainers, not just a single team.
- **Incrementalism** — each phase of the project is scoped narrowly so
  the engineering foundation is solid before higher-risk,
  higher-complexity work (data ingestion, feature engineering, and
  modeling) begins.

## Architecture Overview

The diagram below shows the **target architecture** and marks which stages
exist today. Ingestion, feature engineering, and model training/evaluation are
implemented and tested; experiment tracking, serving, and monitoring are still
planned.

```mermaid
flowchart TD
    subgraph sources["Data Sources"]
        A[Raw & External Data]
    end

    subgraph pipeline["ForecastForge Pipeline"]
        B["Ingestion<br/>(forecastforge.data)"]
        C["Feature Engineering<br/>(forecastforge.features)"]
        D["Model Training & Evaluation<br/>(forecastforge.models,<br/>forecastforge.evaluation)"]
        E["Experiment Tracking<br/>(forecastforge.tracking)"]
        F["Serving / Inference API<br/>(forecastforge.serving)"]
    end

    subgraph ops["Operations"]
        G["Monitoring<br/>(forecastforge.monitoring)"]
    end

    A --> B --> C --> D --> F
    D <--> E
    F --> G
    G -. feedback .-> B

    classDef implemented fill:#e6f2e6,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef planned fill:#f5f5f5,stroke:#999,stroke-dasharray: 4 3,color:#555;
    class B,C,D implemented;
    class E,F,G planned;
```

Design principles guiding future implementation phases:

- Each stage will be a separately testable, independently importable
  subpackage under `src/forecastforge/` (see
  [Repository Layout](#repository-layout) below), orchestrated by
  pipeline definitions under `pipelines/`.
- Configuration will be environment-driven (`configs/`, `.env`) and
  validated with `pydantic`, rather than scattered as magic constants.
  See `configs/development.toml`, `configs/testing.toml`, and
  `configs/production.toml` for the environment overlay structure.
- Experiment and run artifacts will be tracked under `mlruns/`, with
  raw, interim, processed, and external datasets kept in clearly
  separated `data/` subdirectories to prevent accidental mutation of
  source data.
- Containerization (`docker/`, `docker-compose.yml`) will formalize the
  runtime environment once services exist to containerize.
- Significant, hard-to-reverse architectural choices are recorded as
  ADRs under [`docs/adr/`](docs/adr/) — see
  [0001-use-src-layout.md](docs/adr/0001-use-src-layout.md) for the
  first example.

Full architectural documentation, including which stages are implemented,
lives in [docs/architecture.md](docs/architecture.md).

## Repository Layout

```
forecastforge/
├── .github/
│   ├── workflows/
│   │   └── ci.yml            # Lint, type-check, test, wheel smoke test
│   ├── ISSUE_TEMPLATE/        # Bug report and feature request forms
│   ├── dependabot.yml         # Automated dependency updates
│   └── pull_request_template.md
├── src/
│   └── forecastforge/         # Installable Python package (src-layout)
│       ├── config/             # Settings loading & validation
│       ├── core/                # Shared domain types & interfaces
│       ├── common/              # Cross-cutting helpers (placeholder)
│       ├── utils/                # General-purpose utilities (placeholder)
│       ├── data/                  # Data ingestion & access layer
│       ├── features/               # Leakage-safe feature engineering
│       ├── models/                  # Forecaster interface, baselines, optional models
│       ├── evaluation/               # Metrics, walk-forward backtester, benchmark
│       ├── tracking/                  # Experiment tracking integration (planned)
│       ├── serving/                    # Inference API layer (planned)
│       ├── monitoring/                  # Runtime & model monitoring (planned)
│       ├── pipelines/                    # Pipeline orchestration (planned)
│       └── schemas/                       # Shared data/config schemas
├── tests/                     # Automated test suite (pytest)
├── docs/                      # Project documentation
│   ├── architecture.md
│   ├── development.md
│   ├── data-pipeline.md
│   ├── feature-engineering.md
│   ├── model-comparison.md
│   ├── serving.md
│   ├── monitoring.md
│   └── adr/                    # Architectural Decision Records
├── pipelines/                 # Pipeline/orchestration definitions (future)
├── notebooks/                 # Exploratory notebooks (not for production code)
├── docker/                    # Container build contexts (placeholders)
│   ├── app/Dockerfile
│   ├── mlflow/Dockerfile
│   └── monitor/Dockerfile
├── scripts/                   # Developer & operational scripts
│   ├── bootstrap.sh             # Local setup (POSIX)
│   └── bootstrap.ps1            # Local setup (Windows/PowerShell)
├── configs/                   # Application & logging configuration
│   ├── settings.toml            # Base settings
│   ├── development.toml         # Development overrides
│   ├── testing.toml             # Testing overrides
│   ├── production.toml          # Production overrides
│   └── logging.yaml             # Logging configuration
├── data/                      # Local data workspace (git-ignored contents)
│   ├── raw/                     # Immutable, original data
│   ├── interim/                  # Intermediate, partially processed data
│   ├── processed/                 # Final, model-ready datasets
│   └── external/                   # Data sourced from third parties
├── models/                    # Serialized model artifacts (git-ignored)
├── mlruns/                    # Experiment tracking store (git-ignored)
├── pyproject.toml             # Packaging, dependency, and tool configuration
├── Makefile                   # Common developer workflows
├── docker-compose.yml         # Local service orchestration (placeholder)
├── .pre-commit-config.yaml    # Pre-commit hook configuration
├── CONTRIBUTING.md            # Contribution guide
├── CODE_OF_CONDUCT.md         # Contributor Covenant v2.1
├── SECURITY.md                # Vulnerability reporting process
├── CHANGELOG.md               # Keep a Changelog-formatted history
└── VERSION                    # Current project version
```


## Current Status

This repository is in the **model evaluation phase**. What exists
today:

- A `src`-layout Python package (`forecastforge`) with its full planned
  subpackage structure. The `core`, `schemas`, `data`, `features`, `models`,
  and `evaluation` subpackages are implemented; `serving`, `monitoring`, and
  `tracking` are stubbed and await future phases.
- **Core exception hierarchy** (`forecastforge.core.errors`) and a
  **validated ``TimeSeries`` container** (`forecastforge.schemas`) with
  strict structural validation (sorted ``DatetimeIndex``, numeric/finite
  columns, optional gap detection).
- **Data ingestion layer** (`forecastforge.data`):
  * ``DataSource`` protocol + ``BaseDataSource`` for load → slice →
    validate → wrap.
  * Deterministic, fully-seeded **synthetic OHLCV generator**
    (`SyntheticSource`) used as the default offline dataset.
  * Optional **Yahoo Finance connector** (`YFinanceSource`) installed
    via the ``[data]`` extra, lazily imported and cached to Parquet.
  * CSV and Parquet **file loaders** that preserve a ``DatetimeIndex``.
- **Feature engineering framework** (`forecastforge.features`):
  * ``FeatureSet`` — a container that keeps the strict index and target
    guarantees of ``TimeSeries`` while allowing NaN in generated columns,
    and reports the warm-up period as an explicit row count.
  * ``Transform`` base class with a checked contract: a transform declares
    the columns it produces and ``apply`` verifies them on every call.
  * Seven causal transforms — lags, rolling statistics, simple and log
    returns, annualised volatility, RSI, MACD, and calendar features with
    cyclic sine/cosine encodings.
  * ``FeaturePipeline`` for ordered composition, where later steps consume
    columns created by earlier ones.
  * A leakage test suite that truncates and perturbs the future for every
    transform, and is itself tested against three deliberately leaky
    implementations.
- **Forecasting models** (`forecastforge.models`):
  * ``Forecaster`` — a fit/predict interface with the horizon fixed at
    construction, input validation and the fitted-before-predict guard
    implemented once in the base class.
  * Four dependency-free baselines — naive, seasonal naive, moving average,
    and drift.
  * Optional ``ETSForecaster``, ``ARIMAForecaster`` (statsmodels), and
    ``LightGBMForecaster`` (LightGBM) behind the ``[models]`` extra. They
    import cleanly without their dependency and raise a message naming the
    extra when constructed.
- **Evaluation** (`forecastforge.evaluation`):
  * MAE, RMSE, sMAPE, MASE, directional accuracy, and interval coverage, each
    a pure function tested against a hand-computed value.
  * ``WalkForwardBacktester`` with expanding and rolling windows, step size,
    and a gap embargo, returning a tidy frame of one row per
    ``(fold, horizon, model)``.
  * Leakage tests for the backtester itself: a structural audit, perturbation
    and truncation checks, and two negative controls that would catch a
    harness that could be fooled.
  * A reproducible benchmark (``make benchmark``) over five 2000-day synthetic
    paths that writes a markdown table and a figure to ``docs/assets/``.
- A modern `pyproject.toml`-based build (Hatchling) targeting Python
  3.12+.
- Linting and formatting via [Ruff](https://github.com/astral-sh/ruff);
  static type checking via [MyPy](https://mypy-lang.org/) in strict
  mode.
- A test harness via [pytest](https://docs.pytest.org/) with coverage
  reporting configured and enforced at an 85% floor.
- Continuous integration (`.github/workflows/ci.yml`) running Ruff,
  MyPy, and pytest on Python 3.12 and 3.13, a second job that installs the
  `[models]` extra so the optional-model tests really run, then building the
  wheel and importing it in a clean virtual environment, on every push and
  pull request.
- Pre-commit hooks, Dependabot, YAML issue forms, a pull request
  template, and community health files (`CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`).
- Environment-specific configuration overlays
  (`configs/development.toml`, `configs/testing.toml`,
  `configs/production.toml`).
- Bootstrap scripts (`scripts/bootstrap.sh`, `scripts/bootstrap.ps1`)
  automating local environment setup.

What does **not** exist yet (by design):

- A serving/API layer.
- Monitoring or alerting logic.
- Experiment tracking and a model registry.
- Built Docker images (Dockerfiles are placeholders only).

## What works today

The pipeline runs end to end on offline data: generate a series, engineer
features, fit a model, backtest it, and score it — all **real, tested, and
runnable**. There is no serving layer or monitoring yet.

The first demo uses the deterministic synthetic generator (`SyntheticSource`)
to produce a ~2-year daily OHLCV series for a ticker, validates it through the
`TimeSeries` container, and plots it with the optional `[viz]` extra.

Run it yourself (after installing the optional visualization dependency):

```bash
pip install -e ".[viz]"
python examples/01_synthetic_data.py
```

The script validates the generated frame into a `TimeSeries` and writes its
plot to `docs/assets/synthetic_spy_demo.png`:

![Synthetic asset (SPY-like) - 504 days of deterministically generated daily OHLCV with regime shading](docs/assets/synthetic_spy_demo.png)

This demo exercises **only** the capabilities implemented so far:

- **Data generation** — a fully-seeded, reproducible synthetic OHLCV source.
- **Validation** — every generated frame is parsed into a validated `TimeSeries`
  (sorted `DatetimeIndex`, finite numeric columns).
- **Loading** — the same `TimeSeries` round-trips through the CSV and Parquet
  file loaders without loss of structure.

### Feature engineering

The second demo runs a `FeaturePipeline` over that same series. Each transform
is causal — the value at any row is computed only from rows at or before it —
and the pipeline reports how many leading rows are still warming up:

```bash
python examples/02_features.py
```

```text
Feature columns: 13
Warm-up rows  : 33 of 504
Usable rows   : 471 (93% of the series)
Complete      : True
```

The figure below is produced by that command. The shaded band is the warm-up
region: 33 rows where MACD's two smoothing stages have not yet accumulated
enough history, so no model may use them.

![Leakage-safe features on the synthetic series: price with a 20-day rolling mean, RSI(14), MACD with signal and histogram, and annualised 20-day volatility](docs/assets/feature_overview.png)

What this demo demonstrates:

- **Lags, rolling statistics, returns, volatility, RSI, MACD, and calendar
  features** — thirteen columns, all causal.
- **Explicit warm-up** — `FeatureSet.warmup` counts the unusable prefix
  instead of hiding it behind a fill or a silent row drop.
- **Verified leakage safety** — a dedicated test suite truncates and perturbs
  the future for every transform and asserts the past never changes.

### Forecasting and backtesting

Models implement one interface — fit on a history, predict the next `horizon`
values — and the walk-forward backtester hands each fold only the rows at or
before its cutoff:

```python
from forecastforge.data import SyntheticConfig, SyntheticSource
from forecastforge.evaluation import BacktestConfig, WalkForwardBacktester
from forecastforge.models import Drift, Naive

series = SyntheticSource(config=SyntheticConfig(seed=3, length=600)).fetch(symbol="SPY")
result = WalkForwardBacktester(BacktestConfig(initial_train=252, horizon=1)).run(
    series.target_series, {"naive": Naive(), "drift": Drift(window=20)}
)
print(result.summary())
```

`make benchmark` compares every installed model over five deterministic
2000-day paths and writes [`docs/assets/benchmark_results.md`](docs/assets/benchmark_results.md)
and the figure below:

![Mean and per-fold MASE for each model across five synthetic paths](docs/assets/benchmark_mase.png)

**The measured result is that no model beats the naive forecast.** ARIMA scores
a mean MASE of 1.087 against naive's 1.088 — a 0.001 difference on a 0.141
seed-to-seed spread, which is a tie, not a win. ETS and drift land within
noise, the moving average is three times worse, and LightGBM is worse still.
On a near-random-walk price path there is no exploitable structure to find, and
this repository says so rather than tuning until the table looks better. The
full table and the rules behind it are in
[docs/model-comparison.md](docs/model-comparison.md).

What this demonstrates:

- **A contract that cannot be quietly broken** — validation, the
  fitted-before-predict guard, and the output-length check live in the base
  class.
- **Leakage tests for the harness, not just the models** — the backtester is
  audited structurally, perturbed, truncated, and then tested against two
  deliberately cheating models to prove the audit would catch them.
- **Honest reporting** — the benchmark states its tolerance for calling
  something a win, and reports a tie as a tie.

Everything beyond model evaluation (CLI, tracking, serving, containers,
monitoring — phases 4–9) is planned and not yet implemented.

## Development Workflow

```bash
# Clone the repository
git clone https://github.com/Abolfazlrwm/forecastforge.git
cd forecastforge

# One-shot environment setup: creates a venv, installs the package and
# dev dependencies, installs the pre-commit hook, and copies .env.example
# to .env if it doesn't already exist.
./scripts/bootstrap.sh        # macOS / Linux
# or
./scripts/bootstrap.ps1       # Windows / PowerShell

# Activate the environment in new shells
source .venv/bin/activate     # macOS / Linux
# or
. .venv/Scripts/Activate.ps1  # Windows / PowerShell
```

Day-to-day commands, all defined in the [Makefile](Makefile):

| Command | Purpose |
|---|---|
| `make install` | Editable-install the package with dev + docs dependencies, install pre-commit hook |
| `make lint` | Run Ruff lint checks |
| `make format` | Auto-format code with Ruff |
| `make typecheck` | Run MyPy in strict mode |
| `make test` | Run the pytest suite with coverage |
| `make benchmark` | Run the offline model benchmark and refresh `docs/assets/` (needs `[models,viz]`) |
| `make run-precommit` | Run all pre-commit hooks against the whole repo |
| `make clean` | Remove caches, build artifacts, and coverage output |

All `make` targets auto-detect the project virtualenv (`.venv/Scripts/python.exe`
on Windows, `.venv/bin/python` elsewhere) and fall back to `python` otherwise;
override the interpreter for every target with `PYTHON=...`. `make benchmark-fast`
runs a reduced benchmark (one seed, short series, no figure) for CI smoke checks.

Before opening a pull request, run:

```bash
make lint typecheck test run-precommit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution
workflow, including branching strategy, commit conventions, and the
pull request checklist, and [docs/development.md](docs/development.md)
for the extended development guide.

## Roadmap Summary

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Repository health — tooling, CI, packaging, docs, OSS hygiene | ✅ Completed |
| 1 | Data layer — schemas, validation, DataSource protocol, synthetic generator, file loaders, optional YFinance connector | ✅ Completed |
| 2 | Feature engineering framework — `FeatureSet`, causal transforms, pipeline, leakage tests | ✅ Completed |
| 3 | Models + evaluation framework — `Forecaster` interface, baselines, optional ETS/ARIMA/LightGBM, metrics, walk-forward backtester, benchmark | ✅ Completed |
| 4 | CLI / pipeline orchestration | ⏳ Planned |
| 5 | Experiment tracking & model registry integration | ⏳ Planned |
| 6 | Serving layer / inference API | ⏳ Planned |
| 7 | Containerized deployment & orchestration | ⏳ Planned |
| 8 | Observability, alerting, and SLOs | ⏳ Planned |
| 9 | Showcase, documentation, and community onboarding | ⏳ Planned |

This roadmap will evolve; significant architectural choices will be
captured as ADRs under [`docs/adr/`](docs/adr/) as they are made.
**Phases 0–3 are completed. Phases 4–9 are planned.**

## Contributing

ForecastForge is intended to be maintained by multiple contributors.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching
strategy, commit message convention (Conventional Commits), pull
request checklist, and coding standards, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1)
for community expectations. Security issues should be reported per
[SECURITY.md](SECURITY.md) rather than filed as public issues.

## License

ForecastForge is licensed under the [MIT License](LICENSE).
