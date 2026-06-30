"""Tests for the Rust-style Option type -- one assert per test."""

import pytest

from gravityml.shared_kernel.option import Nothing, Some, from_optional
from gravityml.shared_kernel.result import Err, Ok, UnwrapError

# --- queries ---


def test_some_is_some_true():
    assert Some(1).is_some() is True


def test_some_is_nothing_false():
    assert Some(1).is_nothing() is False


def test_nothing_is_nothing_true():
    assert Nothing().is_nothing() is True


def test_nothing_is_some_false():
    assert Nothing().is_some() is False


# --- extraction: Some ---


def test_some_unwrap_returns_value():
    assert Some(5).unwrap() == 5


def test_some_unwrap_or_returns_value():
    assert Some(5).unwrap_or(0) == 5


def test_some_unwrap_or_else_returns_value():
    assert Some(5).unwrap_or_else(lambda: 0) == 5


def test_some_expect_returns_value():
    assert Some(5).expect("must be present") == 5


# --- extraction: Nothing ---


def test_nothing_unwrap_raises():
    with pytest.raises(UnwrapError):
        Nothing().unwrap()


def test_nothing_unwrap_or_returns_default():
    assert Nothing().unwrap_or(0) == 0


def test_nothing_unwrap_or_else_calls_op():
    assert Nothing().unwrap_or_else(lambda: 7) == 7


def test_nothing_expect_raises_with_message():
    with pytest.raises(UnwrapError, match="want a value"):
        Nothing().expect("want a value")


# --- transform / chain: Some ---


def test_some_fmap_applies_op():
    assert Some(2).fmap(lambda x: x + 1) == Some(3)


def test_some_fmap_or_applies_op():
    assert Some(2).fmap_or(-1, lambda x: x + 1) == 3


def test_some_and_then_chains():
    assert Some(2).and_then(lambda x: Some(x * 10)) == Some(20)


def test_some_or_else_returns_self():
    assert Some(2).or_else(lambda: Some(0)) == Some(2)


def test_some_filter_keeps_when_predicate_true():
    assert Some(4).filter(lambda x: x > 2) == Some(4)


def test_some_filter_drops_when_predicate_false():
    assert Some(1).filter(lambda x: x > 2) == Nothing()


def test_some_inspect_returns_self():
    seen: list[int] = []
    assert Some(2).inspect(seen.append) == Some(2)
    assert seen == [2]


# --- transform / chain: Nothing ---


def test_nothing_fmap_returns_self():
    assert Nothing().fmap(lambda x: x + 1) == Nothing()


def test_nothing_fmap_or_returns_default():
    assert Nothing().fmap_or(-1, lambda x: x + 1) == -1


def test_nothing_and_then_short_circuits():
    assert Nothing().and_then(lambda x: Some(x * 10)) == Nothing()


def test_nothing_or_else_recovers():
    assert Nothing().or_else(lambda: Some(0)) == Some(0)


def test_nothing_filter_returns_self():
    assert Nothing().filter(lambda x: x > 2) == Nothing()


# --- interop with Result ---


def test_some_ok_or_returns_ok():
    assert Some(2).ok_or("missing") == Ok(2)


def test_some_ok_or_else_returns_ok():
    assert Some(2).ok_or_else(lambda: "missing") == Ok(2)


def test_nothing_ok_or_returns_err():
    assert Nothing().ok_or("missing") == Err("missing")


def test_nothing_ok_or_else_computes_err():
    assert Nothing().ok_or_else(lambda: "missing") == Err("missing")


# --- from_optional lifter ---


def test_from_optional_value_is_some():
    assert from_optional(5) == Some(5)


def test_from_optional_none_is_nothing():
    assert from_optional(None) == Nothing()
