<!--
  Thanks for contributing to ForecastForge. Keep the description focused on
  what changed and why; the checklist below is what reviewers will verify.
-->

## What does this change?

<!-- A short summary of the change. Link the issue it closes, if any. -->

Closes #

## Why?

<!--
  The motivation: the bug, the limitation, or the use case. Reviewers should
  be able to judge the approach without reading the diff first.
-->

## How was it verified?

<!--
  Commands you ran and what they reported. For behaviour changes, describe the
  test that now covers it. If a number appears in the docs or README, say
  which command produces it.
-->

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy src` passes
- [ ] `pytest` passes
- [ ] New or changed behaviour has tests
- [ ] Public modules, classes, and functions have docstrings and type hints
- [ ] Documentation (`README.md`, `docs/`, ADRs) is updated if behaviour or
      architecture changed
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] The branch targets `develop`, not `main`

## Notes for reviewers

<!-- Anything you are unsure about, or would like a second opinion on. -->
