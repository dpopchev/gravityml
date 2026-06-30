"""The configuration shell -- the one place that reads the environment.

:func:`get_settings` instantiates the frozen :class:`Settings` type, which is when
``gravityml.toml`` and ``GRAVITYML__``-prefixed env vars are actually read (env
wins). The result is cached, so the environment is read once per process; tests
reset it with ``get_settings.cache_clear()``.

Reading the environment is I/O, so it lives here in the imperative shell -- never
in ``domain/`` or ``application/``. Callers depend on ``get_settings()``, never on
``Settings()`` constructed directly.
"""

from __future__ import annotations

from functools import lru_cache

from gravityml.shared_kernel.settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide resolved settings, reading env + TOML once."""
    return Settings()
