# Contributing to ForecastForge

Thank you for your interest in contributing to ForecastForge. This
document describes the conventions and workflow used to keep the
project consistent and maintainable across many contributors.

> **Note on current scope:** Phases 0 (repository foundation), 1 (data
> layer, schemas, and validation), and 2 (feature engineering) are
> completed and merged to `main`. Phase 3 (models and evaluation) is next.
> See the [Roadmap Summary](README.md#roadmap-summary) in `README.md` for
> the full plan.

## Code of Conduct

By participating in this project, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Branching Strategy

ForecastForge uses a two-tier branch model:

- `main` is the release trunk. It only ever contains code that has been
  reviewed, has a green CI run, and is considered releasable. It is
  updated exclusively by merging `develop` (or a hotfix branch) through
  a pull request.
- `develop` is the integration branch. It is the default target for
  feature work and is expected to be green at all times.
- All work happens on short-lived topic branches created from `develop`,
  named with a type prefix, e.g.:
  - `feature/data-ingestion-interfaces`
  - `fix/logging-config-path`
  - `docs/architecture-diagram`
  - `chore/bump-ruff`
  - `refactor/config-sources`

Topic branches are merged into `develop` via pull request. Direct pushes
to either `main` or `develop` are discouraged, even for maintainers.
History is never rewritten on shared branches: no force-pushes and no
rebasing of published commits.

### Promotion to `main`

When `develop` reaches a coherent milestone (the end of a numbered
phase), it is merged into `main` through a pull request. The merge is a
**regular merge commit** (not a squash) so that every phase commit is
preserved on `main` for bisectability and history clarity.

Release cadence:
1. Feature branches are created from `develop`, reviewed via PR, and
   merged into `develop`.
2. At the end of each phase, open a PR `develop -> main` titled
   `chore(release): <phase description>`.
3. Wait for CI green on the PR, then merge with a merge commit.
4. `main` must never lag behind `develop` by more than one phase.

`VERSION` is bumped, `CHANGELOG.md` is updated, and an annotated
`vX.Y.Z` tag is created on the merge commit as part of the release
process.

## Commit Convention

ForecastForge follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
Each commit message should be structured as:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Common types used in this project:

| Type | Use for |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation-only changes |
| `style` | Formatting changes with no code behavior change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A performance improvement |
| `test` | Adding or correcting tests |
| `build` | Changes to the build system or dependencies |
| `ci` | Changes to CI configuration |
| `chore` | Other changes that don't modify src or test files |

Examples:

```
feat(data): add CSV ingestion connector interface
fix(config): correct default logging level fallback
docs(adr): record decision to use src-layout
```

Breaking changes are indicated with a `!` after the type/scope (e.g.
`feat(api)!: rename forecast endpoint`) and a `BREAKING CHANGE:` footer
explaining the impact.

## Pull Request Checklist

Before opening a pull request, please confirm:

- [ ] The branch is up to date with `develop` and targets `develop`.
- [ ] `make lint` passes with no errors.
- [ ] `make format` has been run (code is Ruff-formatted).
- [ ] `make typecheck` passes with no MyPy errors.
- [ ] `make test` passes locally.
- [ ] `make run-precommit` passes (all pre-commit hooks).
- [ ] New or changed behavior has corresponding tests.
- [ ] Public functions, classes, and modules have docstrings and type
      annotations.
- [ ] Documentation (`README.md`, `docs/`, ADRs) is updated if the
      change affects architecture, workflow, or public behavior.
- [ ] Commit messages follow the Conventional Commits format above.
- [ ] The PR description explains **what** changed and **why**, and
      links any related issue.

## Coding Standards

- **Python version:** 3.12+. Use modern syntax (e.g. `X | None` instead
  of `Optional[X]`, built-in generics instead of `typing.List`/`Dict`).
- **Style and linting:** enforced by [Ruff](https://github.com/astral-sh/ruff)
  per the configuration in `pyproject.toml`. Run `make format` before
  committing; do not hand-format code that Ruff would reformat.
- **Typing:** all new code must be fully type-annotated. MyPy runs in
  strict mode (`pyproject.toml`); untyped `def`s and implicit `Any` are
  not permitted outside of `tests/`.
- **Docstrings:** every public module, class, and function should have a
  concise docstring describing its purpose. Follow the style already
  used in `src/forecastforge/__init__.py` and the subpackage
  `__init__.py` files.
- **Package layout:** new functionality belongs in the appropriate
  `src/forecastforge/` subpackage (`config`, `core`, `common`, `utils`,
  `data`, `features`, `models`, `evaluation`, `tracking`, `serving`,
  `monitoring`, `pipelines`, `schemas`) rather than a new top-level
  module. If you're unsure which subpackage a change belongs in, ask in
  the pull request or a preceding issue.
- **Tests:** new behavior should ship with tests under `tests/`, mirroring
  the structure of `src/forecastforge/`. Tests should be deterministic
  and not depend on network access or external services unless
  explicitly mocked.
- **Architectural decisions:** non-trivial, hard-to-reverse design
  choices (framework selection, storage format, API design) should be
  recorded as an ADR in `docs/adr/`, following the format used in
  `docs/adr/0001-use-src-layout.md`.
- **Dependencies:** avoid adding new dependencies without discussion;
  prefer the standard library where practical. When a new dependency is
  genuinely needed, explain the rationale in the pull request.

## Getting Help

If anything in this guide is unclear, please open an issue — improving
this document is itself a welcome contribution.
