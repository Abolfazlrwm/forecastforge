"""Core domain types, interfaces, and shared abstractions.

Currently provides the exception hierarchy every other subpackage raises
from; see :mod:`forecastforge.core.errors`.
"""

from __future__ import annotations

from forecastforge.core.errors import (
    DataError,
    DataValidationError,
    ForecastForgeError,
    MissingOptionalDependencyError,
)

__all__ = [
    "DataError",
    "DataValidationError",
    "ForecastForgeError",
    "MissingOptionalDependencyError",
]
