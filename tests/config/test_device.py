"""Tests for forecastforge.config.device."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Generator
from types import ModuleType, SimpleNamespace

import pytest

from forecastforge.config.device import DeviceType, detect_device, resolve_device


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool | None) -> None:
    """Install a minimal torch stand-in so accelerator branches can be exercised.

    PyTorch is deliberately not a test dependency, so without a stub the CUDA
    and MPS code paths would never run on a CPU-only CI machine, and a bug in
    them would only surface on a contributor's workstation.

    Args:
        monkeypatch: The active pytest monkeypatch fixture.
        cuda: Value returned by ``torch.cuda.is_available()``.
        mps: Value returned by ``torch.backends.mps.is_available()``, or
            ``None`` to model a torch build that has no MPS backend at all.
    """
    fake = ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    fake.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=None if mps is None else SimpleNamespace(is_available=lambda: mps)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


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


def test_detect_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cuda=True, mps=True)
    assert detect_device() == DeviceType.CUDA


def test_detect_device_falls_back_to_mps_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cuda=False, mps=True)
    assert detect_device() == DeviceType.MPS


def test_detect_device_returns_cpu_when_no_accelerator_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch, cuda=False, mps=False)
    assert detect_device() == DeviceType.CPU


def test_detect_device_tolerates_torch_without_an_mps_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch builds compiled without MPS expose no `torch.backends.mps`."""
    _install_fake_torch(monkeypatch, cuda=False, mps=None)
    assert detect_device() == DeviceType.CPU
