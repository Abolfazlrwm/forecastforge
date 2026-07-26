"""Deployment environment detection for ForecastForge.

Defines the set of supported deployment environments (development,
testing, production) and the logic used to detect which one is active.

This module is intentionally independent of
:mod:`forecastforge.config.settings`: the active environment must be
known *before* `Settings` can decide which per-environment TOML overlay
(`configs/<environment>.toml`) to load, so resolving it here avoids a
circular dependency between the two modules.
"""

from __future__ import annotations

import os
from enum import StrEnum

__all__ = ["Environment", "get_environment"]

_ENVIRONMENT_ENV_VAR = "FORECASTFORGE_ENV"
_DEFAULT_ENVIRONMENT = "development"


class Environment(StrEnum):
    """Supported ForecastForge deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Resolve the currently active deployment environment.

    Reads the ``FORECASTFORGE_ENV`` environment variable (case-
    insensitive). Defaults to :data:`Environment.DEVELOPMENT` when the
    variable is unset.

    Raises:
        ValueError: If ``FORECASTFORGE_ENV`` is set to a value that is
            not one of ``development``, ``testing``, or ``production``.
    """
    raw = os.environ.get(_ENVIRONMENT_ENV_VAR, _DEFAULT_ENVIRONMENT).strip().lower()
    try:
        return Environment(raw)
    except ValueError:
        valid = ", ".join(member.value for member in Environment)
        raise ValueError(
            f"Invalid {_ENVIRONMENT_ENV_VAR}={raw!r}; expected one of: {valid}"
        ) from None
