"""Datasets bounded context -- functional core.

Domain types for a tabular dataset: a ``DatasetId`` identity, the ``Dataset``
value object wrapping a validated ``pandas.DataFrame``, and the smart constructor
that certifies the frame before it enters the domain -- every column the caller
declares ``required`` must be present and float-typed.

The required columns are NOT hardcoded here: they are passed in (sourced from
configuration at the composition root), so the same core serves any tabular
dataset. Pure and total: no I/O, no exceptions raised -- failures travel as a
``Result``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NewType

import pandas as pd
from pandas.api.types import is_float_dtype

from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result

DatasetId = NewType("DatasetId", str)
"""Opaque dataset identity -- a plain string at the edge, distinct inside the core."""

DEFAULT_DATASET_ID: Final = DatasetId("example")
"""Identity assigned to a dataset when the caller supplies none."""


@dataclass(frozen=True)
class MissingColumns:
    """Required columns absent from the candidate frame."""

    columns: tuple[str, ...]

    def __str__(self) -> str:
        return f"required columns absent: {', '.join(self.columns)}"


@dataclass(frozen=True)
class NonFloatColumns:
    """Required columns present but not of a float dtype."""

    columns: tuple[str, ...]

    def __str__(self) -> str:
        return f"required columns not float-typed: {', '.join(self.columns)}"


type DatasetError = MissingColumns | NonFloatColumns
"""Why a candidate frame failed to certify -- matched structurally at the edge."""


@dataclass(frozen=True)
class DatasetNotSaved:
    """A certified dataset failed to persist.

    The application-visible save failure: it names the ``dataset_id`` and
    ENCAPSULATES the producing ring's underlying error as ``cause`` (an
    :class:`~gravityml.shared_kernel.error.ErrorInfo`), so nothing is lost when an
    infrastructure error is lifted onto the railway. Infrastructure constructs it;
    the application names it in a handler's error union.
    """

    dataset_id: DatasetId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class DatasetNotFound:
    """A stored dataset could not be read back.

    The application-visible read failure (mirror of :class:`DatasetNotSaved`): it
    names the ``dataset_id`` and encapsulates the storage error as ``cause``.
    Infrastructure constructs it on a failed read; the read side names it.
    """

    dataset_id: DatasetId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True, eq=False)
class Dataset:
    """A certified dataset: an identity plus its validated frame.

    Build it through :func:`make_dataset`; a value reaching here is guaranteed to
    carry EXACTLY the ``required_columns`` it was certified against, as floats --
    any other source columns are dropped. ``eq=False`` keeps identity semantics --
    element-wise equality of wrapped frames is ill-defined.
    """

    frame: pd.DataFrame
    dataset_id: DatasetId = DEFAULT_DATASET_ID


def make_dataset(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    dataset_id: DatasetId = DEFAULT_DATASET_ID,
) -> Result[Dataset, DatasetError]:
    """Certify `frame` against `required_columns`, slim to them, and wrap it.

    On success the wrapped frame holds EXACTLY ``required_columns`` (in order);
    columns the source carried beyond them are dropped.
    """

    def _select(certified: pd.DataFrame) -> pd.DataFrame:
        return certified[list(required_columns)]

    def _wrap(certified: pd.DataFrame) -> Dataset:
        return Dataset(frame=certified, dataset_id=dataset_id)

    return _certify_float_columns(frame, required_columns).fmap(_select).fmap(_wrap)


def _certify_float_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...]
) -> Result[pd.DataFrame, DatasetError]:
    """Pass the frame through only if every required column is present and float."""
    missing = tuple(name for name in required_columns if name not in frame.columns)
    if missing:
        return Err(MissingColumns(missing))
    non_float = tuple(
        name for name in required_columns if not is_float_dtype(frame[name])
    )
    if non_float:
        return Err(NonFloatColumns(non_float))
    return Ok(frame)
