"""Tests for the datasets read side -- univariate stats query + projection.

The projection is verified against a 0..100 ramp, where linear-interpolated
percentiles are exact (p1=1, p25=25, p50=50, p99=99) -- a real anchor for the
numbers, not a pandas-vs-pandas tautology. The query handler is exercised with a
hand-written fake ``find`` port (one assert per test).
"""

import math

import pandas as pd

from gravityml.datasets.application import (
    ColumnStats,
    GetUnivariateStats,
    UnivariateStats,
    handle_get_univariate_stats,
    to_univariate_stats,
)
from gravityml.datasets.domain import DEFAULT_DATASET_ID, DatasetNotFound
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok

_COLUMNS = ("x1", "x2", "y")


def _ramp_frame() -> pd.DataFrame:
    """Every column is the float ramp 0,1,...,100 (clean percentiles)."""
    ramp = [float(n) for n in range(101)]
    return pd.DataFrame({name: ramp for name in _COLUMNS})


def _x1_stats() -> ColumnStats:
    stats = to_univariate_stats(_ramp_frame())
    return next(c for c in stats.columns if c.column == "x1")


# --- projection: shape ---


def test_projection_covers_every_stored_column():
    stats = to_univariate_stats(_ramp_frame())
    assert tuple(c.column for c in stats.columns) == _COLUMNS


# --- projection: exact values on the ramp ---


def test_projection_minimum():
    assert _x1_stats().minimum == 0.0


def test_projection_maximum():
    assert _x1_stats().maximum == 100.0


def test_projection_mean():
    assert _x1_stats().mean == 50.0


def test_projection_median_equals_p50():
    stats = _x1_stats()
    assert (stats.median, stats.p50) == (50.0, 50.0)


def test_projection_percentiles_are_exact_on_the_ramp():
    stats = _x1_stats()
    assert (stats.p1, stats.p5, stats.p25, stats.p75, stats.p95, stats.p99) == (
        1.0,
        5.0,
        25.0,
        75.0,
        95.0,
        99.0,
    )


def test_projection_n_unique():
    assert _x1_stats().n_unique == 101


def test_projection_symmetric_ramp_has_zero_skew():
    assert math.isclose(_x1_stats().skewness, 0.0, abs_tol=1e-9)


# --- query handler: railway over the find port ---


def test_query_maps_found_frame_to_stats():
    outcome = handle_get_univariate_stats(
        lambda _: Ok(_ramp_frame()),
        GetUnivariateStats(DEFAULT_DATASET_ID),
    )
    assert isinstance(outcome.unwrap(), UnivariateStats)


def test_query_propagates_not_found():
    not_found = DatasetNotFound(DEFAULT_DATASET_ID, ErrorInfo("x", "absent"))
    outcome = handle_get_univariate_stats(
        lambda _: Err(not_found),
        GetUnivariateStats(DEFAULT_DATASET_ID),
    )
    assert outcome.unwrap_err() is not_found
