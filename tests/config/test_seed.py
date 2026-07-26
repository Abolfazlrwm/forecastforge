"""Tests for forecastforge.config.seed."""

from __future__ import annotations

import builtins
import os
import random

import pytest

from forecastforge.config.seed import set_seed


def test_set_seed_makes_python_random_reproducible() -> None:
    set_seed(1234)
    first = [random.random() for _ in range(5)]
    set_seed(1234)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_different_seeds_diverge() -> None:
    set_seed(1)
    first = [random.random() for _ in range(5)]
    set_seed(2)
    second = [random.random() for _ in range(5)]
    assert first != second


def test_set_seed_sets_pythonhashseed_env_var() -> None:
    set_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_set_seed_handles_missing_optional_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_seed must not raise even when numpy/torch are unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"numpy", "torch"}:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    set_seed(7)  # must not raise
