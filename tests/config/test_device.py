"""Tests for forecastforge.config.device."""

from __future__ import annotations

import builtins
from collections.abc import Generator

import pytest

from forecastforge.config.device import DeviceType, detect_device, resolve_device


@pytest.fixture(autouse=True)
def _clear_device_cache() -> Generator[None, None, None]:
    detect_device.cache_clear()
    yield
    detect_device.cache_clear()


def test_detect_device_falls_back_to_cpu_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PyTorch cannot be imported, detection must degrade to CPU."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert detect_device() == DeviceType.CPU


def test_detect_device_returns_a_concrete_device_type() -> None:
    # Regardless of what hardware/backends are actually available in the
    # environment running this test, detection must never return AUTO.
    assert detect_device() in {DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS}


def test_detect_device_result_is_cached() -> None:
    assert detect_device() is detect_device()


def test_resolve_device_auto_matches_detect_device() -> None:
    assert resolve_device(DeviceType.AUTO) == detect_device()


def test_resolve_device_none_matches_detect_device() -> None:
    assert resolve_device(None) == detect_device()


def test_resolve_device_cpu_always_honored() -> None:
    assert resolve_device(DeviceType.CPU) == DeviceType.CPU


def test_resolve_device_accepts_string_preference() -> None:
    assert resolve_device("cpu") == DeviceType.CPU


def test_resolve_device_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forecastforge.config.device.detect_device", lambda: DeviceType.CPU)
    # Requesting CUDA on a machine detect_device reports as CPU-only
    # should fall back to the actually-available device, not return an
    # unusable DeviceType.CUDA.
    assert resolve_device(DeviceType.CUDA) == DeviceType.CPU


def test_resolve_device_honors_available_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forecastforge.config.device.detect_device", lambda: DeviceType.CUDA)
    assert resolve_device(DeviceType.CUDA) == DeviceType.CUDA
