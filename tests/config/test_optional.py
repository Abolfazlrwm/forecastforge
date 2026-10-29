"""Tests for forecastforge.config._optional."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecastforge.config._optional import optional_import


def test_optional_import_returns_installed_module() -> None:
    assert optional_import("json") is json


def test_optional_import_returns_none_for_missing_module() -> None:
    assert optional_import("forecastforge_missing_optional_dependency") is None


def test_optional_import_returns_none_when_module_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module that exists but fails to import is treated as unavailable."""
    (tmp_path / "ff_broken_module.py").write_text(
        "raise ImportError('simulated broken native build')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert optional_import("ff_broken_module") is None
