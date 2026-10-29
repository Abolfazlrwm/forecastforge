"""Tests for forecastforge.core.errors."""

from __future__ import annotations

import pytest

from forecastforge.core.errors import (
    DataError,
    DataValidationError,
    ForecastForgeError,
    MissingOptionalDependencyError,
)


def test_all_errors_share_one_base_class() -> None:
    """Callers can catch every library failure with a single except clause."""
    for error in (DataError, DataValidationError, MissingOptionalDependencyError):
        assert issubclass(error, ForecastForgeError)


def test_data_validation_error_is_a_value_error() -> None:
    """Invalid input should be catchable the conventional way."""
    with pytest.raises(ValueError, match="bad input"):
        raise DataValidationError("bad input")


def test_data_validation_error_is_a_data_error() -> None:
    with pytest.raises(DataError):
        raise DataValidationError("bad input")


def test_missing_dependency_error_is_an_import_error() -> None:
    with pytest.raises(ImportError, match="install"):
        raise MissingOptionalDependencyError("install forecastforge[data] to use this feature")


def test_base_error_is_not_a_builtin() -> None:
    """A bare ForecastForgeError must not be silently swallowed by except ValueError."""
    assert not issubclass(ForecastForgeError, ValueError)
