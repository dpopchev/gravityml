"""The runtime configuration model -- the typed shape of the environment.

A frozen pydantic-settings model layered from ``gravityml.toml`` UNDER
``GRAVITYML__``-prefixed env vars (env always wins); defining it reads nothing,
instantiation (in the config shell) does.

Starts minimal -- only ``state`` -- and grows one field at a time as contexts
need configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Settings(BaseSettings):
    """Resolved runtime configuration -- one model, two sources (TOML + env).

    Sourced from ``./gravityml.toml`` layered UNDER ``GRAVITYML__`` env vars: env
    ALWAYS wins (see :func:`gravityml.config.get_settings`). Nested sections, once
    added, bind with the same ``__`` delimiter, e.g. ``GRAVITYML__TRAIN__MAX_EPOCHS``.
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAVITYML__",
        env_nested_delimiter="__",
        toml_file="gravityml.toml",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    state: Path = Path("state")
    """Root directory for runtime artifacts (datasets, reports, models); defaults to
    ``state``, following the Linux ``~/.local/state`` convention for reproducible,
    non-essential state."""

    required_columns: tuple[str, ...] = ("x1", "x2", "y")
    """Columns a source frame MUST carry (as floats) to certify as a dataset. The
    schema is configuration, not code: point it at your own data's columns via
    ``GRAVITYML__REQUIRED_COLUMNS='["a","b","target"]'`` or a ``gravityml.toml``
    entry. The default matches the bundled synthetic example (see ``data/``)."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def datasets_dir(self) -> Path:
        """Directory holding persisted datasets; derived as ``<state>/datasets``."""
        return self.state / "datasets"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reports_dir(self) -> Path:
        """Directory holding analytical outputs; derived as ``<state>/reports``."""
        return self.state / "reports"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        """Directory holding persisted models; derived as ``<state>/models``."""
        return self.state / "models"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources so ENV ALWAYS takes precedence over the TOML file.

        First source wins. ``gravityml.toml`` is the lowest-priority real source,
        so any matching ``GRAVITYML__*`` env var (or ``.env`` entry) overrides it.
        A missing ``gravityml.toml`` contributes nothing -- env-only is valid.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
