"""Tests for forecastforge.config.settings."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from forecastforge.config.device import DeviceType
from forecastforge.config.environment import Environment
from forecastforge.config.settings import Settings, get_settings, reload_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Generator[None, None, None]:
    """Ensure the Settings singleton does not leak state across tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_forecastforge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean FORECASTFORGE_* environment.

    Without this, environment variables set by the developer's shell (or
    by a previous test, if monkeypatch's teardown were ever bypassed)
    could silently change which configuration layer "wins" in a given
    test.
    """
    for key in list(os.environ):
        if key.startswith("FORECASTFORGE_"):
            monkeypatch.delenv(key, raising=False)


def test_get_settings_returns_a_singleton() -> None:
    assert get_settings() is get_settings()


def test_reload_settings_returns_a_fresh_but_equivalent_instance() -> None:
    first = get_settings()
    second = reload_settings()
    assert first is not second
    assert first == second


def test_settings_defaults_are_internally_valid() -> None:
    settings = Settings()
    assert settings.app_name
    assert isinstance(settings.environment, Environment)
    assert settings.data_dir.is_absolute()
    assert settings.models_dir.is_absolute()
    assert settings.artifacts_dir.is_absolute()
    assert settings.mlruns_dir.is_absolute()
    assert settings.sequence_length > 0
    assert settings.batch_size > 0
    assert settings.num_workers >= 0
    assert isinstance(settings.device_preference, DeviceType)


def test_toml_configuration_overrides_field_default() -> None:
    # configs/development.toml sets logging.level = "DEBUG", which must
    # override the Settings field default of "INFO" when no environment
    # variable is set (FORECASTFORGE_ENV defaults to "development").
    settings = Settings()
    assert settings.log_level == "DEBUG"


def test_environment_variable_overrides_toml_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORECASTFORGE_APP_NAME", "overridden-app")
    monkeypatch.setenv("FORECASTFORGE_BATCH_SIZE", "128")
    monkeypatch.setenv("FORECASTFORGE_LOG_LEVEL", "ERROR")
    settings = Settings()
    assert settings.app_name == "overridden-app"
    assert settings.batch_size == 128
    # "ERROR" from the env var must win over "DEBUG" from
    # configs/development.toml.
    assert settings.log_level == "ERROR"


def test_forecastforge_env_selects_deployment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORECASTFORGE_ENV", "production")
    settings = Settings()
    assert settings.environment is Environment.PRODUCTION
    # configs/production.toml sets logging.level = "INFO" (same as the
    # field default), so this also exercises the production overlay
    # being selected rather than development's.
    assert settings.log_level == "INFO"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_LOG_LEVEL", "NOT_A_LEVEL")
    with pytest.raises(ValueError, match="log_level"):
        Settings()


def test_log_level_is_case_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_LOG_LEVEL", "warning")
    assert Settings().log_level == "WARNING"


def test_data_dir_env_override_is_resolved_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_dir = tmp_path / "custom-data"
    monkeypatch.setenv("FORECASTFORGE_DATA_DIR", str(custom_dir))
    settings = Settings()
    assert settings.data_dir == custom_dir


def test_sequence_length_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_SEQUENCE_LENGTH", "0")
    with pytest.raises(ValueError, match="greater than 0"):
        Settings()


def test_num_workers_rejects_negative_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_NUM_WORKERS", "-1")
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        Settings()


def test_device_preference_accepts_valid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_DEVICE_PREFERENCE", "cuda")
    assert Settings().device_preference is DeviceType.CUDA


def test_resolved_device_delegates_to_device_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_DEVICE_PREFERENCE", "cpu")
    settings = Settings()
    assert settings.resolved_device() == DeviceType.CPU


def test_settings_is_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValueError):
        settings.app_name = "mutated"  # type: ignore[misc]
