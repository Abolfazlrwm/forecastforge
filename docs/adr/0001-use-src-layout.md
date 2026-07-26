# 0001. Use src-layout for the Python package

## Status

Accepted

## Context

ForecastForge needs a Python package layout that will remain robust as
the project grows to include multiple contributors, a test suite, and
eventually multiple deployable components (ingestion, feature
engineering, training, serving, monitoring).

Two common layouts were considered:

1. **Flat layout** — the package lives directly at the repository root
   (e.g. `forecastforge/__init__.py` next to `pyproject.toml`).
2. **src-layout** — the package lives under a `src/` directory (e.g.
   `src/forecastforge/__init__.py`), separate from repository-root files
   such as configuration, documentation, and tooling.

The flat layout is common in small scripts and tutorials, but has a
well-documented failure mode: because the repository root is often on
`sys.path` (e.g. when running `pytest` or a script from the repo root),
it is possible to accidentally import the package from the source tree
even when it has not been installed, or to import a stale version
instead of the installed one. This masks packaging errors — a
package that "works" only because of the current working directory,
and silently breaks once installed elsewhere (e.g. in a Docker image,
a CI runner, or another contributor's machine) — which is the opposite
of what a project intending to be maintained by multiple contributors
needs.

ForecastForge is explicitly intended to be a production-grade,
multi-contributor open-source project with an eventual multi-component
architecture (ingestion, feature engineering, modeling, serving,
monitoring — see `docs/architecture.md`). It will be installed via
`pip install -e .` in local development, installed non-editable in CI
and containers, and potentially packaged for distribution.

## Decision

ForecastForge will use the **src-layout**: all importable package code
lives under `src/forecastforge/`, and nothing at the repository root is
directly importable as part of the package.

This is reflected in:

- `pyproject.toml`: `[tool.hatch.build.targets.wheel] packages = ["src"]`
  and `[tool.pytest.ini_options] pythonpath = ["src"]`.
- The package only being importable after installation (editable or
  otherwise), never by accident via the current working directory.
- `src/forecastforge/py.typed` marking the installed package as typed,
  consistent with strict MyPy configuration.

## Consequences

**Positive:**

- Tests and tooling exercise the *installed* package, not the source
  tree, catching packaging mistakes (missing files, incorrect package
  discovery) before they reach CI consumers or production containers.
- Prevents accidental imports of uninstalled or stale code, which is
  especially important once ForecastForge has multiple subpackages
  (`data`, `features`, `models`, `serving`, etc.) that could otherwise
  be imported inconsistently depending on the current working
  directory.
- Keeps the repository root reserved for project-level concerns
  (configuration, documentation, CI, Docker, scripts) rather than mixing
  them with importable Python modules.
- Aligns with current Python packaging community guidance (e.g. the
  Python Packaging Authority's own recommendation) and with the tooling
  already adopted in this repository (Hatchling, pytest `pythonpath`
  configuration).

**Negative / trade-offs:**

- Requires an editable or regular install (`pip install -e .`) before
  the package can be imported locally; it cannot be run directly via
  `python forecastforge/...` from the repository root. This is
  mitigated by `scripts/bootstrap.sh` / `scripts/bootstrap.ps1` and the
  `make install` target, which perform this install automatically.
- Slightly deeper import paths in the source tree
  (`src/forecastforge/...` instead of `forecastforge/...`), which is a
  minor, well-understood cost relative to the reliability benefit.

No further action is required as a result of this decision; it is
already reflected in the current repository structure and tooling
configuration.
