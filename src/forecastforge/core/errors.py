"""Exception hierarchy shared across ForecastForge.

Every error the package raises deliberately derives from
:class:`ForecastForgeError`, so callers can catch library failures without
also catching unrelated built-ins. Errors that describe invalid *input* also
inherit from the matching built-in (``ValueError``, ``ImportError``) so that
they behave as a reader would expect.
"""

from __future__ import annotations

__all__ = [
    "DataError",
    "DataValidationError",
    "ForecastForgeError",
    "MissingOptionalDependencyError",
]


class ForecastForgeError(Exception):
    """Base class for every exception raised deliberately by ForecastForge."""


class DataError(ForecastForgeError):
    """Raised when data cannot be loaded, parsed, or produced."""


class DataValidationError(DataError, ValueError):
    """Raised when a dataset violates the time-series contract.

    Inherits from :class:`ValueError` so that ``except ValueError`` — the
    conventional way to handle rejected input — keeps working.
    """


class MissingOptionalDependencyError(ForecastForgeError, ImportError):
    """Raised when a feature needs an optional dependency that is not installed.

    Inherits from :class:`ImportError` so that callers treating a missing
    module conventionally still catch it. The message always names the extra
    to install.
    """
