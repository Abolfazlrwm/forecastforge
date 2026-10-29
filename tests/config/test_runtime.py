"""Tests for forecastforge.config.runtime."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from forecastforge.config import runtime
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


def test_detect_is_interactive_true_in_an_interactive_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REPL sets ``sys.ps1``; a non-interactive process never does."""
    monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
    assert runtime._detect_is_interactive() is True


def test_detect_is_interactive_false_without_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "ps1", raising=False)
    monkeypatch.setattr(sys, "stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
    assert runtime._detect_is_interactive() is False
