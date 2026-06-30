"""Tests for the datasets infrastructure -- one assert per test.

The save port is built by ``make_save_dataset(datasets_dir)`` and persists a
certified dataset as Parquet. Each test points it at a ``tmp_path`` directory; the
failure tests force an OS error and assert the chained ``cause``.
"""

import pandas as pd
import pyarrow.parquet as pq

from gravityml.datasets.domain import DEFAULT_DATASET_ID, make_dataset
from gravityml.datasets.infrastructure import make_find_frame, make_save_dataset
from gravityml.shared_kernel.error import ErrorInfo

_REQUIRED = ("x1", "x2", "y")

# --- hand-written test data ---


def _dataset():
    """A certified dataset with every required column as float."""
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [0.1, 0.2],
            "y": [1.4, 1.5],
        }
    )
    return make_dataset(frame, _REQUIRED).unwrap()


# --- happy path: persist as Parquet ---


def test_save_returns_ok_with_the_dataset(tmp_path):
    save = make_save_dataset(tmp_path)
    dataset = _dataset()
    assert save(dataset).unwrap() is dataset


def test_save_writes_parquet_file(tmp_path):
    save = make_save_dataset(tmp_path)
    save(_dataset())
    assert (tmp_path / f"{DEFAULT_DATASET_ID}.parquet").is_file()


def test_save_creates_missing_directory(tmp_path):
    nested = tmp_path / "datasets" / "store"
    make_save_dataset(nested)(_dataset())
    assert (nested / f"{DEFAULT_DATASET_ID}.parquet").is_file()


def test_saved_parquet_preserves_float_dtypes(tmp_path):
    make_save_dataset(tmp_path)(_dataset())
    back = pd.read_parquet(tmp_path / f"{DEFAULT_DATASET_ID}.parquet")
    assert all(str(dtype) == "float64" for dtype in back.dtypes)


def test_saved_parquet_embeds_dataset_id(tmp_path):
    make_save_dataset(tmp_path)(_dataset())
    metadata = pq.read_table(tmp_path / f"{DEFAULT_DATASET_ID}.parquet").schema.metadata
    assert metadata[b"gravityml.dataset_id"] == DEFAULT_DATASET_ID.encode()


# --- failure: write error wraps the infra cause ---


def test_save_failure_returns_err(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert make_save_dataset(blocker)(_dataset()).is_err()


def test_save_failure_carries_dataset_id(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    error = make_save_dataset(blocker)(_dataset()).unwrap_err()
    assert error.dataset_id == DEFAULT_DATASET_ID


def test_save_failure_encapsulates_cause_error_info(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    error = make_save_dataset(blocker)(_dataset()).unwrap_err()
    assert isinstance(error.cause, ErrorInfo)


def test_save_failure_cause_has_write_failed_code(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    error = make_save_dataset(blocker)(_dataset()).unwrap_err()
    assert error.cause.code == "dataset.storage.write_failed"


# --- read side: find the stored frame (no certification) ---


def test_find_reads_back_the_saved_frame(tmp_path):
    make_save_dataset(tmp_path)(_dataset())
    found = make_find_frame(tmp_path)(DEFAULT_DATASET_ID)
    assert list(found.unwrap().columns) == ["x1", "x2", "y"]


def test_find_missing_dataset_is_err(tmp_path):
    assert make_find_frame(tmp_path)(DEFAULT_DATASET_ID).is_err()


def test_find_missing_dataset_carries_dataset_id(tmp_path):
    error = make_find_frame(tmp_path)(DEFAULT_DATASET_ID).unwrap_err()
    assert error.dataset_id == DEFAULT_DATASET_ID


def test_find_missing_dataset_cause_has_read_failed_code(tmp_path):
    error = make_find_frame(tmp_path)(DEFAULT_DATASET_ID).unwrap_err()
    assert error.cause.code == "dataset.storage.read_failed"
