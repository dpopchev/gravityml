"""Tests for the configuration shell -- one assert per test.

Each test isolates the working directory via ``chdir(tmp_path)`` so the real
``gravityml.toml`` / ``.env`` never leak in, clears the ``get_settings`` cache,
and drives sourcing through ``GRAVITYML__`` env vars and a temp TOML file.
"""

from pathlib import Path

import pytest

from gravityml.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Run each test in a clean cwd with a fresh settings cache."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_toml(body: str) -> None:
    Path("gravityml.toml").write_text(body, encoding="utf-8")


# --- artifacts state dir: default + env ---


def test_default_state_is_state():
    assert get_settings().state == Path("state")


def test_env_overrides_default_state(monkeypatch):
    monkeypatch.setenv("GRAVITYML__STATE", "/srv/artifacts")
    assert get_settings().state == Path("/srv/artifacts")


# --- TOML source + env precedence ---


def test_toml_supplies_state_when_no_env():
    _write_toml('state = "from-toml"\n')
    assert get_settings().state == Path("from-toml")


def test_env_always_wins_over_toml(monkeypatch):
    _write_toml('state = "from-toml"\n')
    monkeypatch.setenv("GRAVITYML__STATE", "from-env")
    assert get_settings().state == Path("from-env")


# --- required-columns schema: default + env ---


def test_default_required_columns_is_the_example_schema():
    assert get_settings().required_columns == ("x1", "x2", "y")


def test_env_overrides_required_columns(monkeypatch):
    monkeypatch.setenv("GRAVITYML__REQUIRED_COLUMNS", '["a", "b", "target"]')
    assert get_settings().required_columns == ("a", "b", "target")


# --- derived directories ---


def test_datasets_dir_defaults_under_state():
    assert get_settings().datasets_dir == Path("state") / "datasets"


def test_datasets_dir_follows_state_override(monkeypatch):
    monkeypatch.setenv("GRAVITYML__STATE", "/srv/artifacts")
    assert get_settings().datasets_dir == Path("/srv/artifacts") / "datasets"


def test_reports_dir_defaults_under_state():
    assert get_settings().reports_dir == Path("state") / "reports"


def test_reports_dir_follows_state_override(monkeypatch):
    monkeypatch.setenv("GRAVITYML__STATE", "/srv/artifacts")
    assert get_settings().reports_dir == Path("/srv/artifacts") / "reports"


def test_models_dir_defaults_under_state():
    assert get_settings().models_dir == Path("state") / "models"


def test_models_dir_follows_state_override(monkeypatch):
    monkeypatch.setenv("GRAVITYML__STATE", "/srv/artifacts")
    assert get_settings().models_dir == Path("/srv/artifacts") / "models"


# --- caching ---


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_cache_clear_rereads_env(monkeypatch):
    first = get_settings().state
    monkeypatch.setenv("GRAVITYML__STATE", "/after-clear")
    get_settings.cache_clear()
    assert (first, get_settings().state) == (
        Path("state"),
        Path("/after-clear"),
    )
