"""A structured, serializable failure descriptor shared across contexts.

``ErrorInfo`` is the cross-cutting shape a lower-ring error takes when a
higher-ring error encapsulates it as a ``cause`` -- it carries a stable ``code``
and a human-readable ``message`` without leaking the producing ring's own error
type. This keeps layered errors chainable (the railway analogue of ``raise X from
Y``) while respecting ring import direction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A failure's stable ``code`` plus a human-readable ``message``."""

    code: str
    message: str

    def __str__(self) -> str:
        """The human-readable reason -- what the edge surfaces to a user.

        ``code`` stays for machine/log use via ``repr``; ``str`` is the message
        alone, so an ``f"...: {error}"`` at the CLI edge reads cleanly.
        """
        return self.message


def fmap_error[E](
    into: Callable[[ErrorInfo], E], code: str, where: object = ""
) -> Callable[[object], E]:
    """Wrap a lower-layer error or caught exception into an ``ErrorInfo`` cause and
    lift it to a layer-specific error via ``into``.

    The original failure becomes the chainable ``ErrorInfo`` cause -- its stable
    ``code`` plus a message (``str(error)``, optionally prefixed with ``where`` for
    context such as a path) -- so a higher ring raises its own error (e.g. a save
    failure) WITHOUT naming the lower ring's exception type (disk-full,
    location-unavailable, ...). Use it as the ``fmap_err`` of ``@safe`` in
    ``infrastructure/`` or on a ``Result`` at a ring boundary.
    """

    def wrap(error: object) -> E:
        message = f"{where}: {error}" if where else str(error)
        return into(ErrorInfo(code=code, message=message))

    return wrap
