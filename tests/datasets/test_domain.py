"""Tests for the datasets domain -- certifying a frame against required columns.

The smart constructor certifies that the caller's required columns are present and
float-typed, keeps EXACTLY those columns (dropping any extras), and wraps the frame
as a ``Dataset``. The required columns are passed in, so the core is bound to no
single dataset's schema.
"""

import pandas as pd

from gravityml.datasets.domain import (
    DEFAULT_DATASET_ID,
    Dataset,
    DatasetId,
    DatasetNotFound,
    DatasetNotSaved,
    MissingColumns,
    NonFloatColumns,
    make_dataset,
)
from gravityml.shared_kernel.error import ErrorInfo

_REQUIRED = ("x1", "x2", "y")


def _frame(rows: int = 2) -> pd.DataFrame:
    """A frame carrying every required column as a float -- certifies cleanly."""
    return pd.DataFrame(
        {
            "x1": [1.0] * rows,
            "x2": [0.1] * rows,
            "y": [float(n) for n in range(rows)],
        }
    )


def test_valid_frame_certifies_as_dataset():
    dataset = make_dataset(_frame(), _REQUIRED).unwrap()
    assert isinstance(dataset, Dataset)


def test_certified_dataset_carries_default_id():
    dataset = make_dataset(_frame(), _REQUIRED).unwrap()
    assert dataset.dataset_id == DEFAULT_DATASET_ID


def test_certified_frame_holds_exactly_the_required_columns():
    dataset = make_dataset(_frame(), _REQUIRED).unwrap()
    assert tuple(dataset.frame.columns) == _REQUIRED


def test_extra_source_columns_are_dropped():
    frame = _frame().assign(extra=[3.0, 4.0], note=[5.0, 6.0])
    dataset = make_dataset(frame, _REQUIRED).unwrap()
    assert tuple(dataset.frame.columns) == _REQUIRED


def test_missing_required_column_is_err():
    frame = _frame().drop(columns=["y"])
    assert make_dataset(frame, _REQUIRED).err() == MissingColumns(("y",))


def test_non_float_required_column_is_err():
    frame = _frame().assign(x2=["a", "b"])
    assert make_dataset(frame, _REQUIRED).err() == NonFloatColumns(("x2",))


def test_missing_columns_str_names_the_columns():
    assert str(MissingColumns(("x2", "y"))) == "required columns absent: x2, y"


def test_non_float_columns_str_names_the_columns():
    assert str(NonFloatColumns(("y",))) == "required columns not float-typed: y"


def test_dataset_not_saved_str_is_the_cause_message():
    not_saved = DatasetNotSaved(
        DatasetId("ds"), ErrorInfo("store.write_failed", "disk full")
    )
    assert str(not_saved) == "disk full"


def test_dataset_not_found_str_is_the_cause_message():
    not_found = DatasetNotFound(
        DatasetId("ds"), ErrorInfo("store.read_failed", "absent")
    )
    assert str(not_found) == "absent"
