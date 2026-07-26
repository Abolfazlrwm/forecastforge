"""Filesystem path resolution utilities for ForecastForge.

Every filesystem path used elsewhere in the codebase should be derived
from the functions in this module rather than hardcoded, so the project
can be relocated, containerized, or installed without source changes.

Path resolution never touches the filesystem beyond checking for marker
files (it does not create directories) — callers are responsible for
creating a directory before writing to it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "get_artifacts_dir",
    "get_configs_dir",
    "get_data_dir",
    "get_external_data_dir",
    "get_interim_data_dir",
    "get_mlruns_dir",
    "get_models_dir",
    "get_processed_data_dir",
    "get_project_root",
    "get_raw_data_dir",
    "resolve_path",
]

_PROJECT_ROOT_ENV_VAR = "FORECASTFORGE_PROJECT_ROOT"
_ROOT_MARKER_FILES = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return the absolute path to the ForecastForge project root.

    Resolution order:

      1. The ``FORECASTFORGE_PROJECT_ROOT`` environment variable, if set
         — always honored as-is, useful in containers or installed
         (non-editable) deployments where there is no source tree to
         search.
      2. Walking upward from this file's own location until a directory
         containing a root marker (``pyproject.toml`` or ``.git``) is
         found — correct for an editable install
         (``pip install -e .``) of the ``src``-layout package, which is
         the expected local development setup.
      3. The current working directory, as a last resort.

    The result is cached for the lifetime of the process. Call
    ``get_project_root.cache_clear()`` (e.g. in tests) to force
    re-resolution after changing ``FORECASTFORGE_PROJECT_ROOT``.
    """
    env_override = os.environ.get(_PROJECT_ROOT_ENV_VAR)
    if env_override:
        return Path(env_override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKER_FILES):
            return candidate

    return Path.cwd().resolve()  # pragma: no cover - unreachable in this repo


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve ``path`` to an absolute :class:`~pathlib.Path`.

    Args:
        path: A relative or absolute path. ``~`` is expanded.
        base: The directory relative paths are resolved against.
            Defaults to :func:`get_project_root`.

    Returns:
        An absolute path. If ``path`` was already absolute (after ``~``
        expansion), it is returned unchanged; otherwise it is joined to
        ``base`` and resolved (normalizing ``.``/``..`` segments without
        requiring the path to exist).
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ((base or get_project_root()) / candidate).resolve()


def get_configs_dir() -> Path:
    """Return the absolute path to the ``configs/`` directory."""
    return get_project_root() / "configs"


def get_data_dir(data_dir: str | Path = "data") -> Path:
    """Return the absolute path to the root data directory."""
    return resolve_path(data_dir)


def get_raw_data_dir(data_dir: str | Path = "data") -> Path:
    """Return the absolute path to the immutable, as-received data directory."""
    return get_data_dir(data_dir) / "raw"


def get_interim_data_dir(data_dir: str | Path = "data") -> Path:
    """Return the absolute path to the intermediate, partially-processed data directory."""
    return get_data_dir(data_dir) / "interim"


def get_processed_data_dir(data_dir: str | Path = "data") -> Path:
    """Return the absolute path to the final, model-ready data directory."""
    return get_data_dir(data_dir) / "processed"


def get_external_data_dir(data_dir: str | Path = "data") -> Path:
    """Return the absolute path to the third-party/external data directory."""
    return get_data_dir(data_dir) / "external"


def get_models_dir(models_dir: str | Path = "models") -> Path:
    """Return the absolute path to the serialized model artifact directory."""
    return resolve_path(models_dir)


def get_artifacts_dir(artifacts_dir: str | Path = "artifacts") -> Path:
    """Return the absolute path to the general run-artifact directory."""
    return resolve_path(artifacts_dir)


def get_mlruns_dir(mlruns_dir: str | Path = "mlruns") -> Path:
    """Return the absolute path to the experiment-tracking store directory."""
    return resolve_path(mlruns_dir)
