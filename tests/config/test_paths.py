"""Tests for forecastforge.config.paths."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from forecastforge.config import paths


@pytest.fixture(autouse=True)
def _clear_project_root_cache() -> Generator[None, None, None]:
    """Ensure project-root caching does not leak across tests."""
    paths.get_project_root.cache_clear()
    yield
    paths.get_project_root.cache_clear()


def test_get_project_root_is_absolute_and_contains_pyproject() -> None:
    root = paths.get_project_root()
    assert root.is_absolute()
    assert (root / "pyproject.toml").is_file()


def test_get_project_root_is_cached() -> None:
    assert paths.get_project_root() is paths.get_project_root()


def test_get_project_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORECASTFORGE_PROJECT_ROOT", str(tmp_path))
    paths.get_project_root.cache_clear()
    assert paths.get_project_root() == tmp_path.resolve()


def test_resolve_path_returns_absolute_for_relative_input() -> None:
    resolved = paths.resolve_path("data")
    assert resolved.is_absolute()
    assert resolved == paths.get_project_root() / "data"


def test_resolve_path_preserves_already_absolute_input(tmp_path: Path) -> None:
    absolute = tmp_path / "somewhere"
    assert paths.resolve_path(absolute) == absolute


def test_resolve_path_expands_user_home() -> None:
    resolved = paths.resolve_path("~")
    assert resolved.is_absolute()
    assert "~" not in str(resolved)


def test_resolve_path_honors_explicit_base(tmp_path: Path) -> None:
    resolved = paths.resolve_path("subdir", base=tmp_path)
    assert resolved == (tmp_path / "subdir").resolve()


def test_data_subdirectory_helpers_nest_under_data_dir() -> None:
    data_dir = paths.get_data_dir()
    assert paths.get_raw_data_dir() == data_dir / "raw"
    assert paths.get_interim_data_dir() == data_dir / "interim"
    assert paths.get_processed_data_dir() == data_dir / "processed"
    assert paths.get_external_data_dir() == data_dir / "external"


def test_get_configs_dir_points_at_configs_directory() -> None:
    configs_dir = paths.get_configs_dir()
    assert configs_dir == paths.get_project_root() / "configs"
    assert (configs_dir / "settings.toml").is_file()


def test_artifact_directory_helpers_point_at_their_directories() -> None:
    root = paths.get_project_root()
    assert paths.get_models_dir() == root / "models"
    assert paths.get_artifacts_dir() == root / "artifacts"
    assert paths.get_mlruns_dir() == root / "mlruns"
