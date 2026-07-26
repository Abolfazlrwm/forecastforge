"""ForecastForge: a production-grade, open-source forecasting platform.

This package is currently in the repository-bootstrap phase. No modeling,
feature engineering, or data ingestion functionality has been implemented
yet. This module intentionally exposes only package metadata so that the
package is importable and its version can be introspected by tooling
(pip, build backends, CI, documentation generators, etc.).
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__: str = metadata.version("forecastforge")
except metadata.PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g. running from a source checkout without
    # `pip install -e .`). Fall back to a development sentinel version.
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
