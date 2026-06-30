"""A Rust-style ``Option`` for values that may be absent.

``Option[T]`` is the discriminated union ``Some[T] | Nothing``: ``Some`` carries a
present value, ``Nothing`` marks absence and carries no reason (that is
:class:`~gravityml.shared_kernel.result.Result`'s job). The method surface
mirrors Rust's ``std::option::Option`` -- querying (``is_some``/``is_nothing``),
extracting (``unwrap``/``unwrap_or``/...), transforming
(``fmap``/``and_then``/``filter``/``or_else``), and bridging to ``Result``
(``ok_or``/``ok_or_else``).

The variant is named ``Nothing`` (Haskell), not Rust's ``None``, to avoid the
clash with Python's built-in ``None``; the rest follows Rust's ``Some``.

    >>> Some(2).fmap(lambda x: x + 1).unwrap()
    3
    >>> Nothing().fmap(lambda x: x + 1).unwrap_or(0)
    0
    >>> from_optional(None).ok_or("missing")
    Err(error='missing')
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Never

from gravityml.shared_kernel.result import Err, Ok, UnwrapError


@dataclass(frozen=True, slots=True)
class Some[T]:
    """The present variant, wrapping a value of type ``T``."""

    value: T

    # --- queries ---
    def is_some(self) -> bool:
        """Return ``True`` -- this is the present variant.

        >>> Some(1).is_some()
        True
        """
        return True

    def is_nothing(self) -> bool:
        """Return ``False`` -- a ``Some`` is not absent.

        >>> Some(1).is_nothing()
        False
        """
        return False

    # --- extraction ---
    def unwrap(self) -> T:
        """Return the contained value.

        >>> Some(5).unwrap()
        5
        """
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the contained value, ignoring ``default`` (only ``Nothing`` uses it).

        >>> Some(5).unwrap_or(0)
        5
        """
        return self.value

    def unwrap_or_else(self, op: Callable[[], T]) -> T:
        """Return the contained value, ignoring ``op`` (only ``Nothing`` calls it).

        >>> Some(5).unwrap_or_else(lambda: 0)
        5
        """
        return self.value

    def expect(self, message: str) -> T:
        """Return the contained value; ``message`` is unused on ``Some``.

        >>> Some(5).expect("must be present")
        5
        """
        return self.value

    # --- transform / chain ---
    def fmap[U](self, op: Callable[[T], U]) -> Some[U]:
        """Apply ``op`` to the value and rewrap in ``Some``.

        >>> Some(2).fmap(lambda x: x + 1)
        Some(value=3)
        """
        return Some(op(self.value))

    def fmap_or[U](self, default: U, op: Callable[[T], U]) -> U:
        """Apply ``op`` to the value, ignoring ``default``.

        >>> Some(2).fmap_or(-1, lambda x: x + 1)
        3
        """
        return op(self.value)

    def and_then[U](self, op: Callable[[T], Option[U]]) -> Option[U]:
        """Chain into ``op``, which itself returns an ``Option``.

        >>> Some(2).and_then(lambda x: Some(x * 10))
        Some(value=20)
        """
        return op(self.value)

    def or_else(self, op: Callable[[], Option[T]]) -> Some[T]:
        """Return self unchanged -- ``or_else`` only recovers ``Nothing``.

        >>> Some(2).or_else(lambda: Some(0))
        Some(value=2)
        """
        return self

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Keep the value if ``predicate`` holds, else collapse to ``Nothing``.

        >>> Some(4).filter(lambda x: x > 2)
        Some(value=4)
        >>> Some(1).filter(lambda x: x > 2)
        Nothing()
        """
        return self if predicate(self.value) else Nothing()

    def inspect(self, op: Callable[[T], object]) -> Some[T]:
        """Run ``op`` on the value for its side effect, then return self.

        >>> Some(2).inspect(lambda x: None)
        Some(value=2)
        """
        op(self.value)
        return self

    # --- interop with Result ---
    def ok_or[E](self, error: E) -> Ok[T]:
        """Convert to ``Ok`` -- ``error`` is unused on ``Some``.

        >>> Some(2).ok_or("missing")
        Ok(value=2)
        """
        return Ok(self.value)

    def ok_or_else[E](self, op: Callable[[], E]) -> Ok[T]:
        """Convert to ``Ok`` -- ``op`` is unused on ``Some``.

        >>> Some(2).ok_or_else(lambda: "missing")
        Ok(value=2)
        """
        return Ok(self.value)


@dataclass(frozen=True, slots=True)
class Nothing:
    """The empty variant -- absence of a value, carrying no reason."""

    # --- queries ---
    def is_some(self) -> bool:
        """Return ``False`` -- a ``Nothing`` holds no value.

        >>> Nothing().is_some()
        False
        """
        return False

    def is_nothing(self) -> bool:
        """Return ``True`` -- this is the empty variant.

        >>> Nothing().is_nothing()
        True
        """
        return True

    # --- extraction ---
    def unwrap(self) -> Never:
        """Raise -- a ``Nothing`` has no value to unwrap.

        >>> Nothing().unwrap()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: ...
        """
        raise UnwrapError("called unwrap on a Nothing value")

    def unwrap_or[T](self, default: T) -> T:
        """Return ``default`` -- the fallback for a ``Nothing``.

        >>> Nothing().unwrap_or(0)
        0
        """
        return default

    def unwrap_or_else[T](self, op: Callable[[], T]) -> T:
        """Return ``op()`` -- compute a fallback from nothing.

        >>> Nothing().unwrap_or_else(lambda: 7)
        7
        """
        return op()

    def expect(self, message: str) -> Never:
        """Raise ``UnwrapError(message)`` -- a ``Nothing`` has no value.

        >>> Nothing().expect("want a value")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        UnwrapError: want a value
        """
        raise UnwrapError(message)

    # --- transform / chain ---
    def fmap[U](self, op: Callable[[Never], U]) -> Nothing:
        """Return self unchanged -- ``fmap`` only transforms ``Some``.

        >>> Nothing().fmap(lambda x: x + 1)
        Nothing()
        """
        return self

    def fmap_or[U](self, default: U, op: Callable[[Never], U]) -> U:
        """Return ``default`` -- ``op`` is only applied on ``Some``.

        >>> Nothing().fmap_or(-1, lambda x: x + 1)
        -1
        """
        return default

    def and_then[U](self, op: Callable[[Never], Option[U]]) -> Nothing:
        """Return self unchanged (short-circuit) -- only ``Some`` chains.

        >>> Nothing().and_then(lambda x: Some(x * 10))
        Nothing()
        """
        return self

    def or_else[T](self, op: Callable[[], Option[T]]) -> Option[T]:
        """Recover by chaining into ``op()``.

        >>> Nothing().or_else(lambda: Some(0))
        Some(value=0)
        """
        return op()

    def filter(self, predicate: Callable[[Never], bool]) -> Nothing:
        """Return self unchanged -- there is no value to test.

        >>> Nothing().filter(lambda x: x > 2)
        Nothing()
        """
        return self

    def inspect(self, op: Callable[[Never], object]) -> Nothing:
        """Return self unchanged -- ``inspect`` only observes ``Some``.

        >>> Nothing().inspect(lambda x: None)
        Nothing()
        """
        return self

    # --- interop with Result ---
    def ok_or[E](self, error: E) -> Err[E]:
        """Convert to ``Err(error)`` -- supply the reason the value was absent.

        >>> Nothing().ok_or("missing")
        Err(error='missing')
        """
        return Err(error)

    def ok_or_else[E](self, op: Callable[[], E]) -> Err[E]:
        """Convert to ``Err(op())`` -- compute the reason lazily.

        >>> Nothing().ok_or_else(lambda: "missing")
        Err(error='missing')
        """
        return Err(op())


type Option[T] = Some[T] | Nothing


def from_optional[T](value: T | None) -> Option[T]:
    """Lift a nullable into an ``Option``: ``None`` -> ``Nothing``, else ``Some``.

    The boundary adapter for ``T | None`` shapes (``dict.get``, a pydantic
    ``X | None`` field) entering the railway.

    >>> from_optional(5)
    Some(value=5)
    >>> from_optional(None)
    Nothing()
    """
    return Nothing() if value is None else Some(value)
