"""Tests for the datasets application layer -- one assert per test.

The handler is pure orchestration: persistence is injected as a ``SaveDatasetFn``
and the required-column schema is injected too, so each test binds a hand-written
fake save (recording spy, success, or failure) and asserts a single property of
the railway.
"""

import pandas as pd

from gravityml.datasets.application import MakeDataset, handle_make_dataset
from gravityml.datasets.domain import (
    DEFAULT_DATASET_ID,
    Dataset,
    DatasetNotSaved,
    MissingColumns,
)
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result

_REQUIRED = ("x1", "x2", "y")
_DISK_FULL = ErrorInfo("dataset.storage.write_failed", "disk full")

# --- hand-written test data ---


def _valid_frame() -> pd.DataFrame:
    """A frame carrying every required column as a float -- certifies cleanly."""
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [0.1, 0.2],
            "y": [1.4, 1.5],
        }
    )


def _save_ok(dataset: Dataset) -> Result[Dataset, DatasetNotSaved]:
    """A save port that persists successfully, echoing the dataset back."""
    return Ok(dataset)


def _save_fails(dataset: Dataset) -> Result[Dataset, DatasetNotSaved]:
    """A save port that always reports a persistence failure."""
    return Err(DatasetNotSaved(dataset.dataset_id, _DISK_FULL))


class _SaveSpy:
    """Records the dataset handed to it, then persists successfully."""

    def __init__(self) -> None:
        self.calls: list[Dataset] = []

    def __call__(self, dataset: Dataset) -> Result[Dataset, DatasetNotSaved]:
        self.calls.append(dataset)
        return Ok(dataset)


# --- certify + persist (happy path) ---


def test_valid_frame_returns_ok():
    cmd = MakeDataset(_valid_frame())
    assert handle_make_dataset(_save_ok, _REQUIRED, cmd).is_ok() is True


def test_valid_frame_returns_certified_dataset():
    cmd = MakeDataset(_valid_frame())
    result = handle_make_dataset(_save_ok, _REQUIRED, cmd)
    assert isinstance(result.unwrap(), Dataset)


def test_certified_dataset_carries_default_id():
    cmd = MakeDataset(_valid_frame())
    result = handle_make_dataset(_save_ok, _REQUIRED, cmd)
    assert result.unwrap().dataset_id == DEFAULT_DATASET_ID


def test_save_invoked_once_when_certified():
    spy = _SaveSpy()
    handle_make_dataset(spy, _REQUIRED, MakeDataset(_valid_frame()))
    assert len(spy.calls) == 1


def test_save_receives_certified_dataset():
    spy = _SaveSpy()
    handle_make_dataset(spy, _REQUIRED, MakeDataset(_valid_frame()))
    assert isinstance(spy.calls[0], Dataset)


# --- certification failure short-circuits before save ---


def test_uncertified_frame_returns_err():
    cmd = MakeDataset(pd.DataFrame({"x1": [1.0]}))
    assert handle_make_dataset(_save_ok, _REQUIRED, cmd).is_err() is True


def test_uncertified_frame_reports_missing_columns():
    cmd = MakeDataset(pd.DataFrame({"x1": [1.0]}))
    result = handle_make_dataset(_save_ok, _REQUIRED, cmd)
    assert result.err() == MissingColumns(("x2", "y"))


def test_uncertified_frame_never_invokes_save():
    spy = _SaveSpy()
    handle_make_dataset(spy, _REQUIRED, MakeDataset(pd.DataFrame()))
    assert spy.calls == []


# --- persistence failure propagates on the error rail ---


def test_save_failure_returns_err():
    cmd = MakeDataset(_valid_frame())
    assert handle_make_dataset(_save_fails, _REQUIRED, cmd).is_err() is True


def test_save_failure_propagates_save_error():
    cmd = MakeDataset(_valid_frame())
    result = handle_make_dataset(_save_fails, _REQUIRED, cmd)
    assert result.err() == DatasetNotSaved(DEFAULT_DATASET_ID, _DISK_FULL)
