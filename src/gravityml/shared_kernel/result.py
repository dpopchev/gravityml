"""A Rust-style ``Result`` for railway-oriented error handling.

``Result[T, E]`` is the discriminated union ``Ok[T] | Err[E]``: ``Ok`` carries a
success value, ``Err`` carries an error. The method surface mirrors Rust's
``std::result::Result`` -- querying (``is_ok``/``is_err``), extracting
(``ok``/``err``/``unwrap``/``unwrap_or``/...), and chaining
(``fmap``/``fmap_err``/``and_then``/``or_else``/``inspect``).

The unwrap family raises :class:`UnwrapError` -- the Python analogue of a Rust
panic -- when called on the wrong variant. Prefer the chaining combinators;
reach for ``unwrap`` only at an outer edge where a failure is genuinely
unrecoverable.

    >>> Ok(2).fmap(lambda x: x + 1).unwrap()
    3
    >>> Err("boom").fmap(lambda x: x + 1).unwrap_or(0)
    0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Never


class UnwrapError(Exception):
    """Raised by the unwrap family when invoked on the wrong variant."""


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """The success variant, wrapping a value of type ``T``."""

    value: T

    # --- queries ---
    def is_ok(self) -> bool:
        """Return ``True`` -- this is the success variant.

        >>> Ok(1).is_ok()
        True
        """
        return True

    def is_err(self) -> bool:
        """Return ``False`` -- an ``Ok`` is not an error.

        >>> Ok(1).is_err()
        False
        """
        return False

    # --- extraction ---
    def ok(self) -> T | None:
        """Return the contained value.

        >>> Ok(5).ok()
        5
        """
        return self.value

    def err(self) -> None:
        """Return ``None`` -- an ``Ok`` carries no error.

        >>> Ok(5).err() is None
        True
        """
        return None

    def unwrap(self) -> T:
        """Return the contained value.

        >>> Ok(5).unwrap()
        5
        """
        return self.value

    def unwrap_err(self) -> Never:
        """Raise -- an ``Ok`` has no error to unwrap.

        >>> Ok(5).unwrap_err()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: ...
        """
        raise UnwrapError(f"called unwrap_err on an Ok value: {self.value!r}")

    def unwrap_or(self, default: T) -> T:
        """Return the contained value, ignoring ``default`` (only ``Err`` uses it).

        >>> Ok(5).unwrap_or(0)
        5
        """
        return self.value

    def unwrap_or_else[F](self, op: Callable[[F], T]) -> T:
        """Return the contained value, ignoring ``op`` (only ``Err`` calls it).

        >>> Ok(5).unwrap_or_else(lambda e: 0)
        5
        """
        return self.value

    def expect(self, message: str) -> T:
        """Return the contained value; ``message`` is unused on ``Ok``.

        >>> Ok(5).expect("must be ok")
        5
        """
        return self.value

    def expect_err(self, message: str) -> Never:
        """Raise ``UnwrapError(message)`` -- an ``Ok`` has no error.

        >>> Ok(5).expect_err("want err")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: want err
        """
        raise UnwrapError(message)

    # --- chaining ---
    def fmap[U](self, op: Callable[[T], U]) -> Ok[U]:
        """Apply ``op`` to the value and rewrap in ``Ok``.

        >>> Ok(2).fmap(lambda x: x + 1)
        Ok(value=3)
        """
        return Ok(op(self.value))

    def fmap_err[F](self, op: Callable[[Never], F]) -> Ok[T]:
        """Return self unchanged -- ``fmap_err`` only transforms ``Err``.

        >>> Ok(2).fmap_err(str.upper)
        Ok(value=2)
        """
        return self

    def fmap_or[U](self, default: U, op: Callable[[T], U]) -> U:
        """Apply ``op`` to the value, ignoring ``default``.

        >>> Ok(2).fmap_or(-1, lambda x: x + 1)
        3
        """
        return op(self.value)

    def and_then[U, F](self, op: Callable[[T], Result[U, F]]) -> Result[U, F]:
        """Chain into ``op``, which itself returns a ``Result``.

        >>> Ok(2).and_then(lambda x: Ok(x * 10))
        Ok(value=20)
        """
        return op(self.value)

    def or_else[G](self, op: Callable[[Never], Result[T, G]]) -> Ok[T]:
        """Return self unchanged -- ``or_else`` only recovers ``Err``.

        >>> Ok(2).or_else(lambda e: Ok(0))
        Ok(value=2)
        """
        return self

    def inspect(self, op: Callable[[T], object]) -> Ok[T]:
        """Run ``op`` on the value for its side effect, then return self.

        >>> Ok(2).inspect(lambda x: None)
        Ok(value=2)
        """
        op(self.value)
        return self

    def inspect_err(self, op: Callable[[Never], object]) -> Ok[T]:
        """Return self unchanged -- ``inspect_err`` only observes ``Err``.

        >>> Ok(2).inspect_err(lambda e: None)
        Ok(value=2)
        """
        return self


@dataclass(frozen=True, slots=True)
class Err[E]:
    """The failure variant, wrapping an error of type ``E``."""

    error: E

    # --- queries ---
    def is_ok(self) -> bool:
        """Return ``False`` -- an ``Err`` is not a success.

        >>> Err("e").is_ok()
        False
        """
        return False

    def is_err(self) -> bool:
        """Return ``True`` -- this is the failure variant.

        >>> Err("e").is_err()
        True
        """
        return True

    # --- extraction ---
    def ok(self) -> None:
        """Return ``None`` -- an ``Err`` carries no value.

        >>> Err("e").ok() is None
        True
        """
        return None

    def err(self) -> E | None:
        """Return the contained error.

        >>> Err("e").err()
        'e'
        """
        return self.error

    def unwrap(self) -> Never:
        """Raise -- an ``Err`` has no value to unwrap.

        >>> Err("e").unwrap()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: ...
        """
        raise UnwrapError(f"called unwrap on an Err value: {self.error!r}")

    def unwrap_err(self) -> E:
        """Return the contained error.

        >>> Err("e").unwrap_err()
        'e'
        """
        return self.error

    def unwrap_or[T](self, default: T) -> T:
        """Return ``default`` -- the fallback for an ``Err``.

        >>> Err("e").unwrap_or(0)
        0
        """
        return default

    def unwrap_or_else[T](self, op: Callable[[E], T]) -> T:
        """Return ``op(error)`` -- compute a fallback from the error.

        >>> Err("e").unwrap_or_else(len)
        1
        """
        return op(self.error)

    def expect(self, message: str) -> Never:
        """Raise ``UnwrapError(message)`` -- an ``Err`` has no value.

        >>> Err("e").expect("want ok")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: want ok
        """
        raise UnwrapError(message)

    def expect_err(self, message: str) -> E:
        """Return the contained error; ``message`` is unused on ``Err``.

        >>> Err("e").expect_err("x")
        'e'
        """
        return self.error

    # --- chaining ---
    def fmap[U](self, op: Callable[[Never], U]) -> Err[E]:
        """Return self unchanged -- ``fmap`` only transforms ``Ok``.

        >>> Err("e").fmap(lambda x: x + 1)
        Err(error='e')
        """
        return self

    def fmap_err[F](self, op: Callable[[E], F]) -> Err[F]:
        """Apply ``op`` to the error and rewrap in ``Err``.

        >>> Err("e").fmap_err(str.upper)
        Err(error='E')
        """
        return Err(op(self.error))

    def fmap_or[U](self, default: U, op: Callable[[Never], U]) -> U:
        """Return ``default`` -- ``op`` is only applied on ``Ok``.

        >>> Err("e").fmap_or(-1, lambda x: x + 1)
        -1
        """
        return default

    def and_then[U, F](self, op: Callable[[Never], Result[U, F]]) -> Err[E]:
        """Return self unchanged (short-circuit) -- only ``Ok`` chains.

        >>> Err("e").and_then(lambda x: Ok(x * 10))
        Err(error='e')
        """
        return self

    def or_else[U, G](self, op: Callable[[E], Result[U, G]]) -> Result[U, G]:
        """Recover by chaining into ``op(error)``.

        >>> Err("e").or_else(lambda e: Ok(0))
        Ok(value=0)
        """
        return op(self.error)

    def inspect(self, op: Callable[[Never], object]) -> Err[E]:
        """Return self unchanged -- ``inspect`` only observes ``Ok``.

        >>> Err("e").inspect(lambda x: None)
        Err(error='e')
        """
        return self

    def inspect_err(self, op: Callable[[E], object]) -> Err[E]:
        """Run ``op`` on the error for its side effect, then return self.

        >>> Err("e").inspect_err(lambda e: None)
        Err(error='e')
        """
        op(self.error)
        return self


type Result[T, E] = Ok[T] | Err[E]
