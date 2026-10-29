"""Tests for forecastforge.config.seed."""

from __future__ import annotations

import builtins
import os
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from forecastforge.config.seed import set_seed


def test_set_seed_makes_python_random_reproducible() -> None:
    """Seeding must determine the whole global RNG state, not just a prefix.

    Comparing the generator state is a stronger assertion than comparing a
    few drawn values: it proves that every subsequent draw is fixed, rather
    than that the first five happen to match.
    """
    set_seed(1234)
    first = random.getstate()
    set_seed(1234)
    assert random.getstate() == first


def test_set_seed_different_seeds_diverge() -> None:
    set_seed(1)
    first = random.getstate()
    set_seed(2)
    assert random.getstate() != first


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


def test_set_seed_seeds_numpy_and_every_torch_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the optional libraries are present, every RNG they expose is seeded.

    A stub is required because neither library is a test dependency, so the
    numpy/CUDA/MPS branches would otherwise never execute in CI.
    """
    calls: list[tuple[str, int]] = []

    numpy = ModuleType("numpy")
    numpy.random = SimpleNamespace(  # type: ignore[attr-defined]
        seed=lambda seed: calls.append(("numpy", seed))
    )

    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: calls.append(("torch", seed))  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        manual_seed_all=lambda seed: calls.append(("cuda", seed)),
    )
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: True)
    )
    torch.mps = SimpleNamespace(  # type: ignore[attr-defined]
        manual_seed=lambda seed: calls.append(("mps", seed))
    )

    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "torch", torch)

    set_seed(7)

    assert calls == [("numpy", 7), ("torch", 7), ("cuda", 7), ("mps", 7)]


def test_set_seed_skips_accelerator_generators_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU-only torch build must not attempt CUDA or MPS seeding."""
    calls: list[str] = []

    numpy = ModuleType("numpy")
    numpy.random = SimpleNamespace(seed=lambda seed: None)  # type: ignore[attr-defined]

    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: calls.append("torch")  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: False)
    )

    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "torch", torch)

    set_seed(11)

    assert calls == ["torch"]
