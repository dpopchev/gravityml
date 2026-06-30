"""End-to-end tests for the gravityml entry point -- one assert per test.

Drives ``main`` exactly as ``uv run gravityml prepare-dataset ...`` would, with
the state dir pointed at ``tmp_path`` via env so the Parquet artifact lands under
``<state>/datasets``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from gravityml.__main__ import main
from gravityml.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAVITYML__STATE", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_valid_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [0.1, 0.2],
            "y": [1.4, 1.5],
        }
    ).to_csv(path, index=False)


def test_prepare_dataset_returns_ok_code(tmp_path):
    source = tmp_path / "src.csv"
    _write_valid_csv(source)
    assert main(["prepare-dataset", "example", "-i", str(source)]) == 0


def test_prepare_dataset_writes_parquet_under_state(tmp_path):
    source = tmp_path / "src.csv"
    _write_valid_csv(source)
    main(["prepare-dataset", "example", "-i", str(source)])
    assert (tmp_path / "datasets" / "example.parquet").is_file()


def test_invalid_dataset_returns_failure_code(tmp_path):
    source = tmp_path / "src.csv"
    pd.DataFrame({"x1": [1.0]}).to_csv(source, index=False)  # missing required columns
    assert main(["prepare-dataset", "example", "-i", str(source)]) == 1


def test_describe_after_prepare_returns_ok(tmp_path):
    source = tmp_path / "src.csv"
    _write_valid_csv(source)
    main(["prepare-dataset", "example", "-i", str(source)])
    assert main(["describe-dataset", "example"]) == 0


def test_describe_writes_report_under_state(tmp_path):
    source = tmp_path / "src.csv"
    _write_valid_csv(source)
    main(["prepare-dataset", "example", "-i", str(source)])
    main(["describe-dataset", "example"])
    report = tmp_path / "reports" / "example.univariate-stats.csv"
    assert report.is_file()


def test_describe_report_has_csv_header(tmp_path):
    source = tmp_path / "src.csv"
    _write_valid_csv(source)
    main(["prepare-dataset", "example", "-i", str(source)])
    main(["describe-dataset", "example"])
    report = tmp_path / "reports" / "example.univariate-stats.csv"
    assert report.read_text(encoding="utf-8").startswith("column,n_unique,")


def test_describe_missing_dataset_returns_failure(tmp_path):
    assert main(["describe-dataset", "example"]) == 1


# --- define-model: certify a training-ready model and persist the manifest ---


def _define_argv(
    model_id: str,
    *,
    input_dim: str = "3",
    target_dim: str = "1",
    dataset_id: str = "d1",
) -> list[str]:
    """A full define invocation -- the recipe plus the now-required training binding."""
    return [
        "define-model",
        model_id,
        "--input-dim",
        input_dim,
        "--target-dim",
        target_dim,
        "--dataset",
        dataset_id,
        "--feature-cols",
        "x",
        "--target-cols",
        "y",
        "--max-epochs",
        "2",
        "--batch-size",
        "4",
        "--accelerator",
        "cpu",
    ]


def test_define_model_returns_ok_code(tmp_path):
    assert main(_define_argv("m1")) == 0


def test_define_model_writes_manifest_under_state(tmp_path):
    main(_define_argv("m1"))
    assert (tmp_path / "models" / "m1.json").is_file()


def test_define_model_invalid_dimension_returns_failure(tmp_path):
    assert main(_define_argv("m1", input_dim="0")) == 1


# --- train-model: train a defined model on a stored dataset ---


def _write_line_parquet(path: Path) -> None:
    """A noiseless line y = 2x + 1 over 16 points, stored as the find port reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [float(i) for i in range(16)]
    pd.DataFrame({"x": rows, "y": [2.0 * x + 1.0 for x in rows]}).to_parquet(path)


def _train_argv(model_id: str) -> list[str]:
    """Train invocation -- just the id; the binding was fixed at define time."""
    return ["train-model", model_id]


def test_train_after_define_returns_ok(tmp_path):
    main(_define_argv("lin", input_dim="1"))
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    assert main(_train_argv("lin")) == 0


def test_train_writes_weights_under_state(tmp_path):
    main(_define_argv("lin", input_dim="1"))
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    main(_train_argv("lin"))
    assert (tmp_path / "models" / "lin.safetensors").is_file()


def test_train_missing_model_returns_failure(tmp_path):
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    assert main(_train_argv("ghost")) == 1


# --- evaluate-model: evaluate a trained model on a stored dataset ---


def _evaluate_argv(model_id: str, dataset_id: str) -> list[str]:
    return [
        "evaluate-model",
        model_id,
        "--dataset",
        dataset_id,
        "--feature-cols",
        "x",
        "--target-cols",
        "y",
    ]


def test_evaluate_after_train_returns_ok(tmp_path):
    main(_define_argv("lin", input_dim="1"))
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    main(_train_argv("lin"))
    assert main(_evaluate_argv("lin", "d1")) == 0


def test_evaluate_writes_evaluated_manifest(tmp_path):
    main(_define_argv("lin", input_dim="1"))
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    main(_train_argv("lin"))
    main(_evaluate_argv("lin", "d1"))
    manifest = json.loads((tmp_path / "models" / "lin.json").read_text())
    assert manifest["status"] == "evaluated"


def test_evaluate_untrained_model_returns_failure(tmp_path):
    main(_define_argv("lin", input_dim="1"))
    _write_line_parquet(tmp_path / "datasets" / "d1.parquet")
    assert main(_evaluate_argv("lin", "d1")) == 1


# --- archive-model: retire a stored model (tombstone its current state) ---


def test_archive_after_define_returns_ok(tmp_path):
    main(_define_argv("m1"))
    assert main(["archive-model", "m1"]) == 0


def test_archive_writes_archived_manifest(tmp_path):
    main(_define_argv("m1"))
    main(["archive-model", "m1"])
    manifest = json.loads((tmp_path / "models" / "m1.json").read_text())
    assert manifest["status"] == "archived"


def test_archive_missing_model_returns_failure(tmp_path):
    assert main(["archive-model", "ghost"]) == 1


# --- describe-model: report a stored model's state and hyperparameters ---


def test_describe_after_define_returns_ok(tmp_path):
    main(_define_argv("m1"))
    assert main(["describe-model", "m1"]) == 0


def test_describe_model_prints_status_and_recipe(tmp_path, capsys):
    main(_define_argv("m1"))
    main(["describe-model", "m1"])
    out = capsys.readouterr().out
    assert "status: defined" in out and "network: sequential-mlp" in out


def test_describe_missing_model_returns_failure(tmp_path):
    assert main(["describe-model", "ghost"]) == 1


# --- predict: run a trained model over an input file, writing target predictions ---


def _write_training_parquet(path: Path) -> None:
    """A training frame: x1 feature, y target (noiseless line y = 0.1 * x1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [float(i) for i in range(16)]
    pd.DataFrame({"x1": rows, "y": [0.1 * r for r in rows]}).to_parquet(path)


def _define_predict_argv(model_id: str, dataset_id: str) -> list[str]:
    return [
        "define-model",
        model_id,
        "--input-dim",
        "1",
        "--target-dim",
        "1",
        "--dataset",
        dataset_id,
        "--feature-cols",
        "x1",
        "--target-cols",
        "y",
        "--max-epochs",
        "2",
        "--batch-size",
        "4",
        "--accelerator",
        "cpu",
    ]


def test_predict_after_train_writes_target_predictions(tmp_path):
    main(_define_predict_argv("p", "d3"))
    _write_training_parquet(tmp_path / "datasets" / "d3.parquet")
    main(_train_argv("p"))
    source = tmp_path / "input.csv"
    pd.DataFrame({"x1": [1.0, 2.0]}).to_csv(source, index=False)
    assert main(["predict", "p", "-i", str(source)]) == 0
    out = pd.read_csv(tmp_path / "input.predictions.csv")
    assert "y" in out.columns


# --- list-models: enumerate the stored models ---


def test_list_models_returns_ok_code(tmp_path):
    main(_define_argv("m1"))
    assert main(["list-models"]) == 0


def test_list_models_prints_each_model_id_and_status(tmp_path, capsys):
    main(_define_argv("m1"))
    main(["list-models"])
    assert "m1: defined" in capsys.readouterr().out
