"""Configuration loading and validation for ForecastForge.

This is the entry point other ForecastForge packages should use to
access application configuration, path resolution, environment/runtime
detection, compute device detection, and reproducibility seeding. See
``docs/development.md`` for a description of the configuration
hierarchy and file layout.

Typical usage::

    from forecastforge.config import get_settings, set_seed

    settings = get_settings()
    set_seed(settings.random_seed)
"""

from __future__ import annotations

from forecastforge.config.device import DeviceType, detect_device, resolve_device
from forecastforge.config.environment import Environment, get_environment
from forecastforge.config.paths import (
    get_artifacts_dir,
    get_configs_dir,
    get_data_dir,
    get_external_data_dir,
    get_interim_data_dir,
    get_mlruns_dir,
    get_models_dir,
    get_processed_data_dir,
    get_project_root,
    get_raw_data_dir,
    resolve_path,
)
from forecastforge.config.runtime import RuntimeInfo, get_runtime_info
from forecastforge.config.seed import set_seed
from forecastforge.config.settings import Settings, get_settings, reload_settings

__all__ = [
    "DeviceType",
    "Environment",
    "RuntimeInfo",
    "Settings",
    "detect_device",
    "get_artifacts_dir",
    "get_configs_dir",
    "get_data_dir",
    "get_environment",
    "get_external_data_dir",
    "get_interim_data_dir",
    "get_mlruns_dir",
    "get_models_dir",
    "get_processed_data_dir",
    "get_project_root",
    "get_raw_data_dir",
    "get_runtime_info",
    "get_settings",
    "reload_settings",
    "resolve_device",
    "resolve_path",
    "set_seed",
]
