"""Tests for the Rust-style Result type -- one assert per test."""

import pytest

from gravityml.shared_kernel.result import Err, Ok, UnwrapError

# --- queries ---


def test_ok_is_ok_true():
    assert Ok(1).is_ok() is True


def test_ok_is_err_false():
    assert Ok(1).is_err() is False


def test_err_is_err_true():
    assert Err("e").is_err() is True


def test_err_is_ok_false():
    assert Err("e").is_ok() is False


# --- extraction: Ok ---


def test_ok_ok_returns_value():
    assert Ok(5).ok() == 5


def test_ok_err_returns_none():
    assert Ok(5).err() is None


def test_ok_unwrap_returns_value():
    assert Ok(5).unwrap() == 5


def test_ok_unwrap_or_returns_value():
    assert Ok(5).unwrap_or(0) == 5


def test_ok_unwrap_or_else_returns_value():
    assert Ok(5).unwrap_or_else(lambda e: 0) == 5


def test_ok_expect_returns_value():
    assert Ok(5).expect("boom") == 5


def test_ok_unwrap_err_raises():
    with pytest.raises(UnwrapError):
        Ok(5).unwrap_err()


def test_ok_expect_err_raises():
    with pytest.raises(UnwrapError):
        Ok(5).expect_err("must be err")


# --- extraction: Err ---


def test_err_err_returns_error():
    assert Err("e").err() == "e"


def test_err_ok_returns_none():
    assert Err("e").ok() is None


def test_err_unwrap_err_returns_error():
    assert Err("e").unwrap_err() == "e"


def test_err_unwrap_or_returns_default():
    assert Err("e").unwrap_or(0) == 0


def test_err_unwrap_or_else_calls_op():
    assert Err("e").unwrap_or_else(len) == 1


def test_err_expect_err_returns_error():
    assert Err("e").expect_err("boom") == "e"


def test_err_unwrap_raises():
    with pytest.raises(UnwrapError):
        Err("e").unwrap()


def test_err_expect_raises():
    with pytest.raises(UnwrapError):
        Err("e").expect("must be ok")


# --- chaining: fmap / fmap_err ---


def test_ok_fmap_transforms_value():
    assert Ok(2).fmap(lambda x: x + 1) == Ok(3)


def test_err_fmap_is_passthrough():
    assert Err("e").fmap(lambda x: x + 1) == Err("e")


def test_err_fmap_err_transforms_error():
    assert Err("e").fmap_err(str.upper) == Err("E")


def test_ok_fmap_err_is_passthrough():
    assert Ok(2).fmap_err(str.upper) == Ok(2)


# --- chaining: and_then / or_else ---


def test_ok_and_then_chains_into_ok():
    assert Ok(2).and_then(lambda x: Ok(x * 10)) == Ok(20)


def test_ok_and_then_can_return_err():
    assert Ok(2).and_then(lambda x: Err("bad")) == Err("bad")


def test_err_and_then_short_circuits():
    assert Err("e").and_then(lambda x: Ok(x * 10)) == Err("e")


def test_err_or_else_recovers():
    assert Err("e").or_else(lambda e: Ok(0)) == Ok(0)


def test_ok_or_else_is_passthrough():
    assert Ok(2).or_else(lambda e: Ok(0)) == Ok(2)


# --- chaining: fmap_or ---


def test_ok_fmap_or_applies_op():
    assert Ok(2).fmap_or(-1, lambda x: x + 1) == 3


def test_err_fmap_or_returns_default():
    assert Err("e").fmap_or(-1, lambda x: x + 1) == -1


# --- chaining: inspect / inspect_err ---


def test_ok_inspect_runs_side_effect():
    seen: list[int] = []
    Ok(7).inspect(seen.append)
    assert seen == [7]


def test_err_inspect_is_noop():
    seen: list[int] = []
    Err("e").inspect(lambda x: seen.append(1))
    assert seen == []


def test_err_inspect_err_runs_side_effect():
    seen: list[str] = []
    Err("e").inspect_err(seen.append)
    assert seen == ["e"]


def test_ok_inspect_err_is_noop():
    seen: list[int] = []
    Ok(7).inspect_err(lambda e: seen.append(1))
    assert seen == []


# --- discrimination via match/case ---


def test_match_case_matches_ok():
    match Ok(1):
        case Ok(value):
            assert value == 1
        case _:
            pytest.fail("Ok did not match")


def test_match_case_matches_err():
    match Err("x"):
        case Err(error):
            assert error == "x"
        case _:
            pytest.fail("Err did not match")
