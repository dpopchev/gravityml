"""Datasets bounded context -- application layer.

CQRS as ordered sections in one module:

(1) Commands -- write side. :func:`handle_make_dataset` certifies a raw frame
    through the domain smart constructor (against the injected required columns)
    and persists it via the injected :data:`SaveDatasetFn`.
(2) Queries -- read side. :func:`handle_get_univariate_stats` binds a
    :data:`FindFrameFn` read port that returns the stored frame (NO aggregate
    hydration, NO certification) and projects it to a stats DTO.
(3) Projections -- pure ``to_*`` transforms: a stored frame -> a read DTO.

Pure orchestration: no I/O lives here -- ports arrive as injected callables, and
fallible steps compose on the railway (no branching on a ``Result``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, SupportsFloat, cast

import pandas as pd

from gravityml.datasets.domain import (
    DEFAULT_DATASET_ID,
    Dataset,
    DatasetError,
    DatasetId,
    DatasetNotFound,
    DatasetNotSaved,
    make_dataset,
)
from gravityml.shared_kernel.result import Result


# ============================ Commands (write side) ============================


@dataclass(frozen=True)
class MakeDataset:
    """Command: certify a raw frame and persist it as a dataset.

    ``frame`` is an edge primitive -- a foreign ``pandas.DataFrame`` that the
    handler re-certifies into the domain before anything else touches it.
    """

    frame: pd.DataFrame
    dataset_id: DatasetId = DEFAULT_DATASET_ID


type SaveError = DatasetNotSaved
"""Why the write port failed to persist a certified dataset -- the domain's
chained save failure (carries an ``ErrorInfo`` cause)."""

type MakeDatasetError = DatasetError | SaveError
"""Either the frame failed to certify, or persistence rejected it."""

type SaveDatasetFn = Callable[[Dataset], Result[Dataset, SaveError]]
"""Injected write port: persist a certified dataset, returning it or an error."""


def _widen(error: MakeDatasetError) -> MakeDatasetError:
    """Identity converter that lifts a branch error into the handler's union.

    ``Result``'s error parameter is invariant, so a ``DatasetError`` rail and a
    ``SaveError`` rail do not merge on their own -- each is widened here before
    joining the single ``MakeDatasetError`` channel.
    """
    return error


def handle_make_dataset(
    save: SaveDatasetFn,
    required_columns: tuple[str, ...],
    cmd: MakeDataset,
) -> Result[Dataset, MakeDatasetError]:
    """Certify the command's frame against `required_columns`, then persist it.

    Certification short-circuits the railway on a ``DatasetError``; only a
    certified ``Dataset`` reaches ``save``, whose ``SaveError`` joins the same
    error channel.
    """

    def _persist(dataset: Dataset) -> Result[Dataset, MakeDatasetError]:
        return save(dataset).fmap_err(_widen)

    return (
        make_dataset(cmd.frame, required_columns, cmd.dataset_id)
        .fmap_err(_widen)
        .and_then(_persist)
    )


# ============================= Queries (read side) =============================


@dataclass(frozen=True)
class GetUnivariateStats:
    """Query: univariate statistics for a stored dataset."""

    dataset_id: DatasetId = DEFAULT_DATASET_ID


type FindFrameFn = Callable[[DatasetId], Result[pd.DataFrame, DatasetNotFound]]
"""Read port: load a stored dataset's frame by id -- NO certification, NO
aggregate hydration. The read-model bypass (returns a primitive frame)."""

type GetUnivariateStatsError = DatasetNotFound
"""Why the univariate-stats query failed -- the stored dataset could not be read."""


def handle_get_univariate_stats(
    find: FindFrameFn,
    query: GetUnivariateStats,
) -> Result[UnivariateStats, GetUnivariateStatsError]:
    """Read the stored frame and project its univariate statistics."""
    return find(query.dataset_id).fmap(to_univariate_stats)


# ======================= Projections (pure read transforms) =====================


@dataclass(frozen=True)
class ColumnStats:
    """Univariate statistics for one numeric column.

    Conventions (stated for reproducibility): ``std`` is the SAMPLE standard
    deviation (ddof=1); ``skewness`` is the bias-corrected Fisher-Pearson sample
    skewness; ``kurtosis`` is Fisher (EXCESS) kurtosis (a normal distribution ->
    0); ``median`` equals ``p50``.
    """

    column: str
    n_unique: int
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class UnivariateStats:
    """Univariate statistics for every column, in column order."""

    columns: tuple[ColumnStats, ...]


_PERCENTILES: Final[tuple[float, ...]] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def to_univariate_stats(frame: pd.DataFrame) -> UnivariateStats:
    """Project a stored frame to per-column univariate statistics (pure).

    Tabulates EVERY column the stored frame carries: a certified dataset holds
    exactly its required columns, so the projection needs no fixed column list.
    """
    return UnivariateStats(tuple(_column_stats(frame[name]) for name in frame.columns))


def _as_float(value: object) -> float:
    """Narrow a pandas scalar (typed loosely in the stubs) to a Python ``float``."""
    return float(cast(SupportsFloat, value))


def _column_stats(series: pd.Series) -> ColumnStats:
    """Compute the tabulated univariate statistics for a single column."""
    quantiles = series.quantile(list(_PERCENTILES))
    return ColumnStats(
        column=str(series.name),
        n_unique=int(series.nunique()),
        minimum=_as_float(series.min()),
        maximum=_as_float(series.max()),
        mean=_as_float(series.mean()),
        median=_as_float(series.median()),
        std=_as_float(series.std()),
        p1=_as_float(quantiles.loc[0.01]),
        p5=_as_float(quantiles.loc[0.05]),
        p25=_as_float(quantiles.loc[0.25]),
        p50=_as_float(quantiles.loc[0.5]),
        p75=_as_float(quantiles.loc[0.75]),
        p95=_as_float(quantiles.loc[0.95]),
        p99=_as_float(quantiles.loc[0.99]),
        skewness=_as_float(series.skew()),
        kurtosis=_as_float(series.kurt()),
    )
