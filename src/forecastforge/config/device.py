"""Compute device detection for ForecastForge.

Provides a backend-agnostic abstraction over the compute devices
ForecastForge may eventually run on (CPU, NVIDIA CUDA, Apple MPS)
without requiring any specific ML framework to be installed at this
stage of the project. PyTorch is treated as an optional dependency: if
it is not installed, detection gracefully falls back to CPU rather than
raising an error.

This module performs *detection* only — it does not construct or
configure any model, tensor, or training component.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from forecastforge.config._optional import optional_import

__all__ = ["DeviceType", "detect_device", "resolve_device"]


class DeviceType(StrEnum):
    """Supported (or requested) compute device types.

    ``AUTO`` is a valid *preference* (e.g. in configuration) meaning
    "pick the best available device"; it is never returned by
    :func:`detect_device`, which always resolves to a concrete device.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@lru_cache(maxsize=1)
def detect_device() -> DeviceType:
    """Detect the best available compute device on this machine.

    Preference order: CUDA > Apple MPS > CPU. Returns
    :data:`DeviceType.CPU` if PyTorch is not installed, or if it is
    installed but no accelerator backend is available. Never returns
    :data:`DeviceType.AUTO`.

    The result is cached for the lifetime of the process, since
    available hardware does not change at runtime. Call
    ``detect_device.cache_clear()`` (e.g. in tests) to force
    re-detection.
    """
    torch = optional_import("torch")
    if torch is None:
        return DeviceType.CPU

    if torch.cuda.is_available():
        return DeviceType.CUDA

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return DeviceType.MPS

    return DeviceType.CPU


def resolve_device(preference: DeviceType | str | None = None) -> DeviceType:
    """Resolve a requested device preference to a concrete, available device.

    Args:
        preference: The requested device. ``None`` or
            :data:`DeviceType.AUTO` triggers auto-detection via
            :func:`detect_device`. An explicit request for
            :data:`DeviceType.CUDA` or :data:`DeviceType.MPS` is honored
            only if that backend is actually available; otherwise this
            function falls back to the auto-detected device rather than
            returning a device that cannot be used.
            :data:`DeviceType.CPU` is always honored, since it is always
            available.

    Returns:
        A concrete, usable :class:`DeviceType` (never ``AUTO``).
    """
    if preference is None:
        return detect_device()

    requested = DeviceType(preference)

    if requested is DeviceType.AUTO:
        return detect_device()

    if requested is DeviceType.CPU:
        return DeviceType.CPU

    available = detect_device()
    if requested == available:
        return requested

    return available
