"""Tests for the CLI delivery infrastructure -- one assert per test.

Covers the argv grammar (argparse -> typed input model) and the source reader
(path -> DataFrame), including the unsupported-format and missing-file failures.
"""

from pathlib import Path

import pandas as pd

from gravityml.cli.domain import (
    ArchiveModel,
    DefineModel,
    DescribeDataset,
    EvaluateModel,
    ListModels,
    PrepareDataset,
    Predict,
    TrainModel,
)
from gravityml.cli.infrastructure import parse_args, read_source, write_report


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"x1": [1.0, 2.0]})


# --- argv grammar ---


def test_parse_prepare_dataset_builds_typed_command():
    assert parse_args(["prepare-dataset", "example", "-i", "in.csv"]) == (
        PrepareDataset("example", Path("in.csv"))
    )


def test_parse_accepts_long_input_flag():
    command = parse_args(["prepare-dataset", "example", "--input", "data.parquet"])
    assert command.source == Path("data.parquet")


def test_parse_describe_dataset_builds_typed_command():
    assert parse_args(["describe-dataset", "example"]) == (DescribeDataset("example"))


# The training binding define-model now requires (data + run sizes); recipe defaults
# (network/activation/optimizer/loss) and the val/seed/accelerator defaults apply.
_DEFINE_BINDING = [
    "--dataset",
    "d1",
    "--feature-cols",
    "x",
    "--target-cols",
    "y",
    "--max-epochs",
    "10",
    "--batch-size",
    "4",
]


def test_parse_define_model_accepts_a_scaler_flag():
    command = parse_args(
        [
            "define-model",
            "m1",
            "--input-dim",
            "3",
            "--target-dim",
            "1",
            "--scaler",
            "standard",
            *_DEFINE_BINDING,
        ]
    )
    assert command.scaler == "standard"


def test_parse_define_model_required_args_apply_recipe_and_binding_defaults():
    assert parse_args(
        [
            "define-model",
            "m1",
            "--input-dim",
            "3",
            "--target-dim",
            "1",
            *_DEFINE_BINDING,
        ]
    ) == DefineModel(
        model_id="m1",
        network_name="sequential-mlp",
        input_dim=3,
        hidden_dims=(),
        target_dim=1,
        activation="relu",
        optimizer_name="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        loss_name="mse",
        dataset_id="d1",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.0,
        max_epochs=10,
        batch_size=4,
        seed=0,
        accelerator="auto",
    )


def test_parse_define_model_parses_hidden_dims_csv():
    command = parse_args(
        [
            "define-model",
            "m1",
            "--input-dim",
            "3",
            "--target-dim",
            "1",
            "--hidden-dims",
            "16,8",
            *_DEFINE_BINDING,
        ]
    )
    assert command.hidden_dims == (16, 8)


def test_parse_define_model_overrides_optimizer():
    command = parse_args(
        [
            "define-model",
            "m1",
            "--input-dim",
            "3",
            "--target-dim",
            "1",
            "--optimizer",
            "sgd",
            *_DEFINE_BINDING,
        ]
    )
    assert command.optimizer_name == "sgd"


def test_parse_define_model_parses_feature_cols_csv():
    command = parse_args(
        [
            "define-model",
            "m1",
            "--input-dim",
            "3",
            "--target-dim",
            "1",
            "--dataset",
            "d1",
            "--feature-cols",
            "a,b",
            "--target-cols",
            "y",
            "--max-epochs",
            "10",
            "--batch-size",
            "4",
        ]
    )
    assert command.feature_columns == ("a", "b")


def test_parse_train_model_carries_only_the_model_id():
    assert parse_args(["train-model", "m1"]) == TrainModel(model_id="m1")


def test_parse_evaluate_model_builds_typed_command():
    assert parse_args(
        [
            "evaluate-model",
            "m1",
            "--dataset",
            "d1",
            "--feature-cols",
            "a,b",
            "--target-cols",
            "y",
        ]
    ) == EvaluateModel(
        model_id="m1",
        dataset_id="d1",
        feature_columns=("a", "b"),
        target_columns=("y",),
    )


def test_parse_archive_model_builds_typed_command():
    assert parse_args(["archive-model", "m1"]) == ArchiveModel(model_id="m1")


def test_parse_predict_builds_typed_command():
    assert parse_args(["predict", "m1", "-i", "in.csv"]) == Predict(
        model_id="m1", source=Path("in.csv")
    )


def test_parse_list_models_builds_typed_command():
    assert parse_args(["list-models"]) == ListModels()


# --- source reader ---


def test_read_source_reads_csv(tmp_path):
    path = tmp_path / "in.csv"
    _frame().to_csv(path, index=False)
    assert read_source(path).unwrap().shape[0] == 2


def test_read_source_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("nope", encoding="utf-8")
    assert read_source(path).unwrap_err().code == "cli.source.unsupported"


def test_read_source_missing_file_is_err(tmp_path):
    assert read_source(tmp_path / "absent.csv").is_err()


# --- report writer ---


def test_write_report_writes_content_creating_dirs(tmp_path):
    path = tmp_path / "reports" / "out.csv"
    write_report(path, "a,b\n1,2")
    assert path.read_text(encoding="utf-8") == "a,b\n1,2"


def test_write_report_returns_the_written_path(tmp_path):
    path = tmp_path / "out.csv"
    assert write_report(path, "x").unwrap() == path


def test_write_report_failure_is_err(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert write_report(blocker / "out.csv", "x").is_err()
