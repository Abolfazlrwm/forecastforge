"""Tests for forecastforge.config.environment."""

from __future__ import annotations

import pytest

from forecastforge.config.environment import Environment, get_environment


def test_get_environment_defaults_to_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORECASTFORGE_ENV", raising=False)
    assert get_environment() is Environment.DEVELOPMENT


def test_get_environment_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_ENV", "production")
    assert get_environment() is Environment.PRODUCTION


def test_get_environment_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_ENV", "TESTING")
    assert get_environment() is Environment.TESTING


def test_get_environment_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_ENV", "  production  ")
    assert get_environment() is Environment.PRODUCTION


def test_get_environment_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTFORGE_ENV", "staging")
    with pytest.raises(ValueError, match="FORECASTFORGE_ENV"):
        get_environment()
