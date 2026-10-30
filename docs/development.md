# Development Guide

## Purpose

This document will be the canonical guide for setting up a local
development environment, running the quality gates, and understanding
the day-to-day contributor workflow for ForecastForge.

## Current Status

**Partially implemented.** The core developer workflow (environment
setup, `make` targets, pre-commit, Conventional Commits) is summarized
in the root [README.md](../README.md) and
[CONTRIBUTING.md](../CONTRIBUTING.md). The **application configuration
system** (`src/forecastforge/config/`) is implemented and documented in
full below. Everything else in this guide (working with data locally,
debugging pipeline/model code, the release process) remains a skeleton
until the corresponding functionality exists.

## Configuration

ForecastForge's configuration system lives entirely under
`src/forecastforge/config/` and is the dependency every other component
reads settings from. This section describes how it works today.

### Configuration Hierarchy

Configuration is resolved from five layers, listed from **highest to
lowest priority** — a value set in a higher layer always wins over the
same value set in a lower one:

1. **Explicit keyword arguments** passed directly to `Settings(...)`.
   Used by tests and by the singleton factory itself; application code
   should not normally construct `Settings` with overrides.
2. **Environment variables**, prefixed `FORECASTFORGE_` and matching a
   field name (e.g. `FORECASTFORGE_BATCH_SIZE`, `FORECASTFORGE_LOG_LEVEL`).
   The one exception is the deployment environment itself, which uses
   `FORECASTFORGE_ENV` (see [Environment Overrides](#environment-overrides)
   below).
3. **A `.env` file** in the project root (see `.env.example` for the
   full list of supported variables). Loaded via `python-dotenv`
   through `pydantic-settings`; if no `.env` file exists, this layer is
   silently skipped.
4. **TOML configuration files** — `configs/settings.toml` (the base
   layer) merged with `configs/<environment>.toml` (an overlay that
   only needs to declare the keys that differ from the base). See
   [How Configuration Files Work](#how-configuration-files-work) below.
5. **Field defaults** declared directly on the `Settings` model in
   `src/forecastforge/config/settings.py`. This is the final fallback
   if no other layer provides a value.

```
init kwargs  >  env vars  >  .env file  >  TOML (base + overlay)  >  field defaults
 (highest)                                                              (lowest)
```

Obtain the resolved configuration via the process-wide singleton rather
than constructing `Settings` yourself, so parsing/validation happens
exactly once per process:

```python
from forecastforge.config import get_settings

settings = get_settings()
print(settings.environment, settings.batch_size, settings.data_dir)
```

If you need to force re-parsing (for example, after changing an
environment variable mid-process — mainly relevant in tests), use
`reload_settings()` or `get_settings.cache_clear()`.

### How Configuration Files Work

`configs/settings.toml` holds environment-agnostic defaults, organized
into four sections:

| Section | Maps to |
|---|---|
| `[project]` | `app_name`, `environment`, `random_seed` |
| `[logging]` | `log_level` |
| `[paths]` | `data_dir`, `models_dir`, `artifacts_dir`, `mlruns_dir` |
| `[defaults]` | `default_asset_symbol`, `default_timeframe`, `sequence_length`, `batch_size`, `num_workers`, `device_preference` |

`configs/development.toml`, `configs/testing.toml`, and
`configs/production.toml` are **overlays**: each only declares the keys
that differ from `settings.toml`. The active overlay is chosen by the
resolved deployment environment (see below) and is merged section by
section on top of the base file — a key present in the overlay wins;
any key the overlay omits keeps its value from `settings.toml`.

All relative paths in `[paths]` are resolved to absolute paths against
the project root (see `forecastforge.config.paths.get_project_root`),
never against the current working directory, so behavior is consistent
regardless of where a command is invoked from.

### Environment Overrides

The **deployment environment** (`development` / `testing` /
`production`) is special: it has to be known *before* the TOML layer
can be loaded, since it determines which overlay file to read. It is
resolved independently by
`forecastforge.config.environment.get_environment()`, which reads the
`FORECASTFORGE_ENV` environment variable and defaults to `development`
if unset:

```bash
export FORECASTFORGE_ENV=testing   # or: production, development
```

An unrecognized value raises a clear `ValueError` rather than silently
falling back, so a typo'd environment name fails fast instead of
quietly loading the wrong overlay.

Every other setting can be overridden individually with an environment
variable named `FORECASTFORGE_<FIELD_NAME>` (uppercased), for example:

```bash
export FORECASTFORGE_BATCH_SIZE=64
export FORECASTFORGE_LOG_LEVEL=DEBUG
export FORECASTFORGE_DATA_DIR=/mnt/external/forecastforge-data
```

See `.env.example` for the complete, documented list of supported
variables — copy it to `.env` and uncomment what you need rather than
exporting variables manually in every shell session.

### Related Utilities

The `config` package also provides several utilities that sit alongside
`Settings` rather than inside it:

- **`forecastforge.config.paths`** — path resolution helpers
  (`get_project_root`, `get_data_dir`, `resolve_path`, etc.) used to
  compute the absolute directories `Settings` exposes, and reusable
  anywhere else a project-relative path is needed.
- **`forecastforge.config.runtime`** — `get_runtime_info()` returns a
  snapshot of the current execution context (Python version, OS,
  hostname, whether running under pytest/CI). This is distinct from the
  deployment environment above: it describes *what machine and process*
  ForecastForge is running as, not *which configuration profile* is
  active.
- **`forecastforge.config.device`** — `detect_device()` /
  `resolve_device()` provide a `DeviceType` abstraction (`cpu`, `cuda`,
  `mps`) over PyTorch's device APIs, without requiring PyTorch to be
  installed. `Settings.resolved_device()` is a convenience wrapper that
  resolves `settings.device_preference` this way.
- **`forecastforge.config.seed`** — `set_seed(seed)` seeds Python's
  `random` module, NumPy's global RNG, and PyTorch's CPU/CUDA/MPS
  generators. NumPy is a core dependency; PyTorch is optional and is
  skipped when absent.

## Planned Contents

The following sections of this guide remain unwritten until the
corresponding functionality exists:

- **Environment setup** — detailed walkthrough of `scripts/bootstrap.sh`
  / `scripts/bootstrap.ps1`, virtual environment management, and
  troubleshooting common setup issues.
- **Running the quality gates** — detailed usage of `make lint`,
  `make format`, `make typecheck`, `make test`, and `make run-precommit`,
  including how to interpret common failures.
- **Working with data locally** — how to point ForecastForge at sample or
  local datasets under `data/`.
- **Debugging and profiling** — recommended tools and workflows once
  pipeline and model code exist.
- **Release process** — how versions are cut, how `CHANGELOG.md` is
  maintained, and how releases are published.

This document will grow alongside the codebase; sections above will be
filled in as the underlying functionality is implemented.
