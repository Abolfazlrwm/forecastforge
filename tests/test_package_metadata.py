"""Smoke tests for the ForecastForge package skeleton.

These tests intentionally validate only packaging/metadata concerns. They
exist to give the CI pipeline, coverage tooling, and contributors a working
example test to model future tests on. No modeling or data-processing
behavior is exercised here yet.
"""

from __future__ import annotations

import forecastforge


def test_package_is_importable() -> None:
    """The top-level package must import without side effects or errors."""
    assert forecastforge is not None


def test_package_exposes_a_version_string() -> None:
    """The package must expose a non-empty semantic version string."""
    assert isinstance(forecastforge.__version__, str)
    assert forecastforge.__version__ != ""
