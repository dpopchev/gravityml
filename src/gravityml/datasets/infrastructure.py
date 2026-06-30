"""Datasets bounded context -- imperative shell.

Persists a certified ``Dataset`` as a Parquet file under a configured directory.
Parquet is chosen because it embeds the Arrow schema -- the certified float
columns survive a round trip exactly, so a reloaded dataset re-certifies -- and
lets us stamp the ``dataset_id`` into the file's key-value metadata.

This ring implements the application's save port STRUCTURALLY (DIP): it depends
only on ``domain`` types and never imports ``application``. All I/O is funnelled
through ``@safe``; a write failure is wrapped by ``fmap_error`` into the ``cause``
(an ``ErrorInfo``) of the domain's :class:`DatasetNotSaved`, so nothing is lost on
the way out.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gravityml.datasets.domain import (
    Dataset,
    DatasetId,
    DatasetNotFound,
    DatasetNotSaved,
)
from gravityml.shared_kernel.error import fmap_error
from gravityml.shared_kernel.result import Result
from gravityml.shared_kernel.safe import safe

type SaveDatasetFn = Callable[[Dataset], Result[Dataset, DatasetNotSaved]]
"""Concrete shape of the save port -- structurally matches the application's
``SaveDatasetFn`` without importing it."""

DATASET_ID_METADATA_KEY: Final = b"gravityml.dataset_id"
"""Parquet footer key under which the dataset identity is stamped."""

WRITE_FAILED_CODE: Final = "dataset.storage.write_failed"
"""``ErrorInfo`` code for a failed dataset write."""

_CAUGHT_WRITE_ERRORS: Final = (OSError, pa.ArrowException)
"""Exceptions treated as a storage failure -- filesystem and Arrow/Parquet errors."""


def _write_parquet(path: Path, dataset: Dataset) -> Path:
    """Write the frame to ``path`` as Parquet, stamping the dataset identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(dataset.frame, preserve_index=False)
    stamped = table.replace_schema_metadata(
        {DATASET_ID_METADATA_KEY: str(dataset.dataset_id).encode()}
    )
    pq.write_table(stamped, path)
    return path


def make_save_dataset(datasets_dir: Path) -> SaveDatasetFn:
    """Build a save port persisting datasets as ``<datasets_dir>/<id>.parquet``.

    The returned callable is the concrete adapter the composition root injects
    where the application expects a ``SaveDatasetFn``.
    """

    def save(dataset: Dataset) -> Result[Dataset, DatasetNotSaved]:
        path = datasets_dir / f"{dataset.dataset_id}.parquet"
        guarded = safe(
            _CAUGHT_WRITE_ERRORS,
            fmap_error(
                lambda cause: DatasetNotSaved(dataset.dataset_id, cause),
                WRITE_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_write_parquet)(path, dataset).fmap(lambda _: dataset)

    return save


# --- read side: find the stored frame (no certification) ---

type FindFrameFn = Callable[[DatasetId], Result[pd.DataFrame, DatasetNotFound]]
"""Concrete shape of the read port -- structurally matches the application's
``FindFrameFn`` without importing it."""

READ_FAILED_CODE: Final = "dataset.storage.read_failed"
"""``ErrorInfo`` code for a failed dataset read."""

_CAUGHT_READ_ERRORS: Final = (OSError, pa.ArrowException)
"""Exceptions treated as a read failure -- filesystem and Arrow/Parquet errors."""


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a stored dataset frame; the embedded Arrow schema restores dtypes."""
    return pd.read_parquet(path)


def make_find_frame(datasets_dir: Path) -> FindFrameFn:
    """Build a read port loading the stored frame at ``<datasets_dir>/<id>.parquet``.

    The read side bypasses certification: it returns the stored frame as-is (the
    trust boundary is the write side), suitable for read-only projections.
    """

    def find(dataset_id: DatasetId) -> Result[pd.DataFrame, DatasetNotFound]:
        path = datasets_dir / f"{dataset_id}.parquet"
        guarded = safe(
            _CAUGHT_READ_ERRORS,
            fmap_error(
                lambda cause: DatasetNotFound(dataset_id, cause),
                READ_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_read_parquet)(path)

    return find
