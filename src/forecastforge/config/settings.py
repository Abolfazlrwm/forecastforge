"""Central application settings for ForecastForge.

:class:`Settings` is the single source of truth for application
configuration. It is a :class:`pydantic_settings.BaseSettings`
subclass, which provides:

- strong typing and validation (pydantic v2)
- environment variable overrides (prefixed ``FORECASTFORGE_``)
- ``.env`` file loading (via python-dotenv, through pydantic-settings)
- custom TOML file loading and merging (via :class:`_TomlConfigSource`
  below)

Configuration precedence, from highest to lowest priority:

  1. Explicit keyword arguments passed to ``Settings(...)``.
  2. Environment variables (``FORECASTFORGE_<FIELD_NAME>``).
  3. A ``.env`` file in the project root.
  4. TOML configuration files: ``configs/settings.toml`` merged with
     ``configs/<environment>.toml`` (the environment overlay wins on
     conflicting keys).
  5. The field defaults declared on :class:`Settings` below.

Use :func:`get_settings` to obtain the process-wide singleton instance
rather than constructing ``Settings()`` directly, so configuration is
parsed and validated exactly once per process. See
``docs/development.md`` for a full description of this hierarchy.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, PositiveInt, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from forecastforge.config.device import DeviceType, resolve_device
from forecastforge.config.environment import Environment, get_environment
from forecastforge.config.paths import get_configs_dir, get_project_root, resolve_path

__all__ = ["Settings", "get_settings", "reload_settings"]

_ENV_PREFIX = "FORECASTFORGE_"
_BASE_CONFIG_FILENAME = "settings.toml"


class _TomlConfigSource(PydanticBaseSettingsSource):
    """A pydantic-settings source that loads and merges TOML config files.

    Reads ``configs/settings.toml`` as a base layer, then merges
    ``configs/<environment>.toml`` on top of it, where the environment
    is resolved via :func:`forecastforge.config.environment.get_environment`
    — kept independent of the `Settings` model under construction to
    avoid a circular dependency.

    Only the nested TOML sections consumed by :class:`Settings`
    (``project``, ``logging``, ``paths``, ``defaults``) are read; each
    is flattened into the flat field names `Settings` expects.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        configs_dir: Path | None = None,
    ) -> None:
        super().__init__(settings_cls)
        self._configs_dir = configs_dir or get_configs_dir()

    def _load_toml(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("rb") as fh:
            return tomllib.load(fh)

    def _merge(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {**base}
        for section, values in overlay.items():
            if isinstance(values, dict) and isinstance(merged.get(section), dict):
                merged[section] = {**merged[section], **values}
            else:
                merged[section] = values
        return merged

    def _flatten(self, raw: dict[str, Any]) -> dict[str, Any]:
        project = raw.get("project", {})
        logging_section = raw.get("logging", {})
        paths_section = raw.get("paths", {})
        defaults_section = raw.get("defaults", {})

        flat: dict[str, Any] = {
            "app_name": project.get("name"),
            "environment": project.get("environment"),
            "random_seed": project.get("random_seed"),
            "log_level": logging_section.get("level"),
            "data_dir": paths_section.get("data_dir"),
            "models_dir": paths_section.get("models_dir"),
            "artifacts_dir": paths_section.get("artifacts_dir"),
            "mlruns_dir": paths_section.get("mlruns_dir"),
            "default_asset_symbol": defaults_section.get("asset_symbol"),
            "default_timeframe": defaults_section.get("timeframe"),
            "sequence_length": defaults_section.get("sequence_length"),
            "batch_size": defaults_section.get("batch_size"),
            "num_workers": defaults_section.get("num_workers"),
            "device_preference": defaults_section.get("device_preference"),
        }
        return {key: value for key, value in flat.items() if value is not None}

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Required by the PydanticBaseSettingsSource ABC. Unused in
        # practice because __call__ below builds the full settings dict
        # in one pass rather than resolving fields individually.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        base = self._load_toml(self._configs_dir / _BASE_CONFIG_FILENAME)
        environment = get_environment()
        overlay = self._load_toml(self._configs_dir / f"{environment.value}.toml")
        return self._flatten(self._merge(base, overlay))


class Settings(BaseSettings):
    """Application-wide, validated ForecastForge configuration.

    Do not construct this directly in application code — use
    :func:`get_settings` so configuration is parsed once and shared.
    Direct construction (``Settings()`` or ``Settings(**overrides)``) is
    intended for tests and for the singleton factory itself.
    """

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file=str(get_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
        frozen=True,
    )

    app_name: str = Field(
        default="forecastforge",
        description="Application/package name, used in logs and diagnostics.",
    )
    environment: Environment = Field(
        default_factory=get_environment,
        validation_alias="FORECASTFORGE_ENV",
        description="Active deployment environment.",
    )
    random_seed: int = Field(default=42, description="Default global random seed.")
    log_level: str = Field(default="INFO", description="Root logging level.")

    data_dir: Path = Field(default=Path("data"), description="Root data directory.")
    models_dir: Path = Field(
        default=Path("models"), description="Trained model artifact directory."
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts"), description="General run-artifact directory."
    )
    mlruns_dir: Path = Field(
        default=Path("mlruns"), description="Experiment-tracking store directory."
    )

    default_asset_symbol: str = Field(
        default="SPY", description="Default forecasting asset/instrument symbol."
    )
    default_timeframe: str = Field(
        default="1d", description="Default forecasting timeframe/bar size."
    )

    sequence_length: PositiveInt = Field(
        default=64, description="Default input sequence length for windowed models."
    )
    batch_size: PositiveInt = Field(
        default=32, description="Default training/inference batch size."
    )
    num_workers: int = Field(default=4, ge=0, description="Default data-loading worker count.")

    device_preference: DeviceType = Field(
        default=DeviceType.AUTO, description="Preferred compute device."
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(f"log_level must be one of {sorted(valid_levels)}, got {value!r}")
        return normalized

    @field_validator("data_dir", "models_dir", "artifacts_dir", "mlruns_dir", mode="after")
    @classmethod
    def _resolve_absolute(cls, value: Path) -> Path:
        return resolve_path(value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority, highest to lowest: init kwargs > env vars > .env
        # file > TOML files. Field defaults are pydantic-settings'
        # implicit final fallback and need no explicit source here.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _TomlConfigSource(settings_cls),
        )

    def resolved_device(self) -> DeviceType:
        """Resolve :attr:`device_preference` to a concrete, available device.

        Convenience wrapper around
        :func:`forecastforge.config.device.resolve_device`.
        """
        return resolve_device(self.device_preference)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Configuration is parsed and validated once per process; subsequent
    calls return the cached instance, avoiding repeated environment
    variable lookups and TOML file parsing.
    """
    return Settings()


def reload_settings() -> Settings:
    """Clear the cached :class:`Settings` singleton and reconstruct it.

    Primarily useful in tests that mutate environment variables or
    config files and need :func:`get_settings` to reflect the change on
    its next call.
    """
    get_settings.cache_clear()
    return get_settings()
