"""``@safe`` -- the I/O boundary wrapper that turns exceptions into ``Result``.

The wrapped call returns Ok(value), or Err(fmap_err(exc)) for a declared catch
type; anything else propagates. Use in infrastructure/ only.

    >>> @safe(ZeroDivisionError, lambda exc: "div by zero")
    ... def reciprocal(n: int) -> float:
    ...     return 1 / n
    >>> reciprocal(4)
    Ok(value=0.25)
    >>> reciprocal(0)
    Err(error='div by zero')
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from gravityml.shared_kernel.result import Err, Ok, Result


def safe[**P, T, E](
    catch: type[Exception] | tuple[type[Exception], ...],
    fmap_err: Callable[[Exception], E],
) -> Callable[[Callable[P, T]], Callable[P, Result[T, E]]]:
    """Build a decorator wrapping a side-effecting call into a ``Result``.

    ``catch`` is the exception (or tuple) treated as a domain failure; ``fmap_err``
    turns a caught exception into the typed error ``E``.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, Result[T, E]]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E]:
            try:
                return Ok(fn(*args, **kwargs))
            except catch as exc:
                return Err(fmap_err(exc))

        return wrapper

    return decorate
