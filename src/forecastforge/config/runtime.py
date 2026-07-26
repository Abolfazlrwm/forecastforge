"""Runtime execution context detection for ForecastForge.

Distinct from :mod:`forecastforge.config.environment` (which answers
*which deployment stage* is configured — development, testing, or
production), this module answers *what machine and process*
ForecastForge is currently executing in/as: the interpreter, platform,
host, and whether the process is running under a test runner or CI.
This is intended for diagnostics and logging, not for branching core
application behavior.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from functools import lru_cache

__all__ = ["RuntimeInfo", "get_runtime_info"]

_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL")


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """An immutable snapshot of the current runtime execution context."""

    python_version: str
    platform_system: str
    platform_release: str
    hostname: str
    process_id: int
    is_ci: bool
    is_pytest: bool
    is_interactive: bool


def _detect_is_ci() -> bool:
    """Return True if common CI environment indicators are present."""
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


def _detect_is_pytest() -> bool:
    """Return True if the current process is running under pytest."""
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _detect_is_interactive() -> bool:
    """Return True if attached to an interactive terminal session."""
    if hasattr(sys, "ps1"):
        return True
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        return False


@lru_cache(maxsize=1)
def get_runtime_info() -> RuntimeInfo:
    """Return a cached snapshot of the current runtime execution context.

    The result is cached for the lifetime of the process. Call
    ``get_runtime_info.cache_clear()`` (e.g. in tests) to force a fresh
    snapshot.
    """
    return RuntimeInfo(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        hostname=platform.node(),
        process_id=os.getpid(),
        is_ci=_detect_is_ci(),
        is_pytest=_detect_is_pytest(),
        is_interactive=_detect_is_interactive(),
    )
