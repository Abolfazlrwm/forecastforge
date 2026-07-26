"""Tests for forecastforge.config.runtime."""

from __future__ import annotations

import dataclasses

import pytest

from forecastforge.config.runtime import RuntimeInfo, get_runtime_info


def test_get_runtime_info_returns_a_populated_snapshot() -> None:
    info = get_runtime_info()
    assert isinstance(info, RuntimeInfo)
    assert info.python_version
    assert info.platform_system
    assert info.process_id > 0


def test_get_runtime_info_detects_pytest() -> None:
    # This test itself runs under pytest, so detection must reflect that.
    assert get_runtime_info().is_pytest is True


def test_get_runtime_info_is_cached() -> None:
    assert get_runtime_info() is get_runtime_info()


def test_runtime_info_is_frozen() -> None:
    info = get_runtime_info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.python_version = "9.9.9"  # type: ignore[misc]
