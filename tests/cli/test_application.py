"""Tests for the CLI delivery application -- one assert per test.

``run_prepare_dataset`` is pure orchestration over two injected ports
(``read_source`` and ``prepare``); each test binds hand-written fakes and asserts
a single property of the rendered :class:`CliReport`.
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from gravityml.cli.application import (
    render_model_description,
    render_univariate_stats,
    run_archive_model,
    run_define_model,
    run_describe_dataset,
    run_describe_model,
    run_evaluate_model,
    run_list_models,
    run_predict,
    run_prepare_dataset,
    run_train_model,
)
from gravityml.cli.domain import (
    ArchiveModel,
    DefineModel,
    DescribeDataset,
    DescribeModel,
    EvaluateModel,
    ExitCode,
    ListModels,
    Predict,
    PrepareDataset,
    TrainModel,
)
from gravityml.mlmodel.application import (
    EvaluationSummary,
    ModelDescription,
    ModelsListing,
    ModelSummary,
    TrainingSummary,
)
from gravityml.datasets.application import ColumnStats, UnivariateStats
from gravityml.datasets.domain import DatasetNotFound, MissingColumns
from gravityml.mlmodel.domain import (
    Accelerator,
    ModelId,
    ModelNotFound,
    ModelsNotListed,
    NetworkName,
    NonPositiveLearningRate,
    ScalerName,
)
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok

_COMMAND = PrepareDataset("example", Path("in.csv"))
_SAVED = Path("state/datasets/example.parquet")


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"x1": [1.0]})


# --- success ---


def test_success_reports_ok_exit():
    report = run_prepare_dataset(
        _COMMAND,
        read_source=lambda _: Ok(_frame()),
        prepare=lambda _: Ok(_SAVED),
    )
    assert report.exit_code is ExitCode.OK


def test_success_message_names_the_saved_path():
    report = run_prepare_dataset(
        _COMMAND,
        read_source=lambda _: Ok(_frame()),
        prepare=lambda _: Ok(_SAVED),
    )
    assert str(_SAVED) in report.message


# --- read failure short-circuits before prepare ---


def test_read_failure_reports_failure_exit():
    report = run_prepare_dataset(
        _COMMAND,
        read_source=lambda _: Err(ErrorInfo("cli.source.read_failed", "boom")),
        prepare=lambda _: Ok(_SAVED),
    )
    assert report.exit_code is ExitCode.FAILURE


def test_read_failure_message_mentions_the_source():
    report = run_prepare_dataset(
        _COMMAND,
        read_source=lambda _: Err(ErrorInfo("cli.source.read_failed", "boom")),
        prepare=lambda _: Ok(_SAVED),
    )
    assert "in.csv" in report.message


# --- prepare failure is reported ---


def test_prepare_failure_reports_failure_exit():
    report = run_prepare_dataset(
        _COMMAND,
        read_source=lambda _: Ok(_frame()),
        prepare=lambda _: Err(MissingColumns(("y",))),
    )
    assert report.exit_code is ExitCode.FAILURE


# --- describe-dataset: render and persist the stats table ---

_DESCRIBE = DescribeDataset("example")
_REPORT_PATH = Path("state/reports/example.univariate-stats.csv")


def _stats() -> UnivariateStats:
    column = ColumnStats(
        column="x1",
        n_unique=2,
        minimum=1.0,
        maximum=2.0,
        mean=1.5,
        median=1.5,
        std=0.7,
        p1=1.0,
        p5=1.0,
        p25=1.0,
        p50=1.5,
        p75=2.0,
        p95=2.0,
        p99=2.0,
        skewness=0.0,
        kurtosis=-2.0,
    )
    return UnivariateStats((column,))


class _SaveReportSpy:
    """Records the content handed to it, then reports a written path."""

    def __init__(self) -> None:
        self.content: str | None = None

    def __call__(self, content: str):
        self.content = content
        return Ok(_REPORT_PATH)


def test_describe_success_reports_ok_exit():
    report = run_describe_dataset(
        _DESCRIBE,
        describe=lambda _: Ok(_stats()),
        save_report=lambda _: Ok(_REPORT_PATH),
    )
    assert report.exit_code is ExitCode.OK


def test_describe_success_message_names_saved_path():
    report = run_describe_dataset(
        _DESCRIBE,
        describe=lambda _: Ok(_stats()),
        save_report=lambda _: Ok(_REPORT_PATH),
    )
    assert str(_REPORT_PATH) in report.message


def test_describe_persists_the_rendered_csv():
    spy = _SaveReportSpy()
    run_describe_dataset(_DESCRIBE, describe=lambda _: Ok(_stats()), save_report=spy)
    assert spy.content is not None and "x1" in spy.content


def test_describe_query_failure_reports_failure_exit():
    not_found = DatasetNotFound("example", ErrorInfo("x", "absent"))
    spy = _SaveReportSpy()
    report = run_describe_dataset(
        _DESCRIBE, describe=lambda _: Err(not_found), save_report=spy
    )
    assert (report.exit_code, spy.content) == (ExitCode.FAILURE, None)


def test_describe_write_failure_reports_failure_exit():
    report = run_describe_dataset(
        _DESCRIBE,
        describe=lambda _: Ok(_stats()),
        save_report=lambda _: Err(ErrorInfo("cli.report.write_failed", "disk full")),
    )
    assert report.exit_code is ExitCode.FAILURE


def test_render_starts_with_csv_header():
    assert (
        render_univariate_stats(_stats()).splitlines()[0].startswith("column,n_unique,")
    )


# --- define-model: lift the input, drive the define port, render the outcome ---

_DEFINE_COMMAND = DefineModel(
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
_MANIFEST = Path("state/models/m1.json")


class _DefineSpy:
    """Records the lifted mlmodel command, then reports a written manifest path."""

    def __init__(self) -> None:
        self.request = None

    def __call__(self, request):
        self.request = request
        return Ok(_MANIFEST)


def test_define_success_reports_ok_exit():
    report = run_define_model(_DEFINE_COMMAND, define=lambda _: Ok(_MANIFEST))
    assert report.exit_code is ExitCode.OK


def test_define_success_message_names_manifest_path():
    report = run_define_model(_DEFINE_COMMAND, define=lambda _: Ok(_MANIFEST))
    assert str(_MANIFEST) in report.message


def test_define_lifts_string_flags_into_domain_enums():
    spy = _DefineSpy()
    run_define_model(_DEFINE_COMMAND, define=spy)
    assert spy.request.network_name is NetworkName.SEQUENTIAL_MLP


def test_define_lifts_accelerator_into_domain_enum():
    spy = _DefineSpy()
    run_define_model(_DEFINE_COMMAND, define=spy)
    assert spy.request.accelerator is Accelerator.AUTO


def test_define_lifts_scaler_into_domain_enum():
    spy = _DefineSpy()
    run_define_model(replace(_DEFINE_COMMAND, scaler="standard"), define=spy)
    assert spy.request.scaler is ScalerName.STANDARD


def test_define_failure_reports_failure_exit():
    report = run_define_model(
        _DEFINE_COMMAND, define=lambda _: Err(NonPositiveLearningRate(0.0))
    )
    assert report.exit_code is ExitCode.FAILURE


# --- train-model: lift the input, drive the train port, render the outcome ---

_TRAIN_COMMAND = TrainModel(model_id="m1")
_TRAIN_MANIFEST = Path("state/models/m1.json")


class _TrainSpy:
    """Records the lifted mlmodel command, then reports a written manifest path."""

    def __init__(self) -> None:
        self.request = None

    def __call__(self, request):
        self.request = request
        return Ok(_TRAIN_MANIFEST)


def test_train_success_reports_ok_exit():
    report = run_train_model(_TRAIN_COMMAND, train=lambda _: Ok(_TRAIN_MANIFEST))
    assert report.exit_code is ExitCode.OK


def test_train_success_message_names_manifest_path():
    report = run_train_model(_TRAIN_COMMAND, train=lambda _: Ok(_TRAIN_MANIFEST))
    assert str(_TRAIN_MANIFEST) in report.message


def test_train_carries_only_the_model_id():
    spy = _TrainSpy()
    run_train_model(_TRAIN_COMMAND, train=spy)
    assert spy.request.model_id == "m1"


def test_train_failure_reports_failure_exit():
    not_found = ModelNotFound(ModelId("m1"), ErrorInfo("x", "absent"))
    report = run_train_model(_TRAIN_COMMAND, train=lambda _: Err(not_found))
    assert report.exit_code is ExitCode.FAILURE


# --- evaluate-model: lift the input, drive the evaluate port, render the outcome ---

_EVALUATE_COMMAND = EvaluateModel(
    model_id="m1",
    dataset_id="d1",
    feature_columns=("x",),
    target_columns=("y",),
)
_EVALUATE_MANIFEST = Path("state/models/m1.json")


def test_evaluate_success_reports_ok_exit():
    report = run_evaluate_model(
        _EVALUATE_COMMAND, evaluate=lambda _: Ok(_EVALUATE_MANIFEST)
    )
    assert report.exit_code is ExitCode.OK


def test_evaluate_success_message_names_manifest_path():
    report = run_evaluate_model(
        _EVALUATE_COMMAND, evaluate=lambda _: Ok(_EVALUATE_MANIFEST)
    )
    assert str(_EVALUATE_MANIFEST) in report.message


def test_evaluate_failure_reports_failure_exit():
    not_found = ModelNotFound(ModelId("m1"), ErrorInfo("x", "absent"))
    report = run_evaluate_model(_EVALUATE_COMMAND, evaluate=lambda _: Err(not_found))
    assert report.exit_code is ExitCode.FAILURE


# --- archive-model: lift the input, drive the archive port, render the outcome ---

_ARCHIVE_COMMAND = ArchiveModel(model_id="m1")
_ARCHIVE_MANIFEST = Path("state/models/m1.json")


def test_archive_success_reports_ok_exit():
    report = run_archive_model(
        _ARCHIVE_COMMAND, archive=lambda _: Ok(_ARCHIVE_MANIFEST)
    )
    assert report.exit_code is ExitCode.OK


def test_archive_success_message_names_manifest_path():
    report = run_archive_model(
        _ARCHIVE_COMMAND, archive=lambda _: Ok(_ARCHIVE_MANIFEST)
    )
    assert str(_ARCHIVE_MANIFEST) in report.message


def test_archive_failure_reports_failure_exit():
    not_found = ModelNotFound(ModelId("m1"), ErrorInfo("x", "absent"))
    report = run_archive_model(_ARCHIVE_COMMAND, archive=lambda _: Err(not_found))
    assert report.exit_code is ExitCode.FAILURE


# --- uniform rendering: a failure's reason reaches the user as a clean string ---


def test_define_failure_message_carries_the_reason():
    report = run_define_model(
        _DEFINE_COMMAND, define=lambda _: Err(NonPositiveLearningRate(0.0))
    )
    assert report.message.endswith("learning rate must be strictly positive: 0.0")


# --- describe-model: drive the describe port, render the state + hyperparameters ---

_DESCRIBE_MODEL = DescribeModel(model_id="demo")


def _model_description() -> ModelDescription:
    return ModelDescription(
        model_id="demo",
        status="trained",
        network_name="sequential-mlp",
        input_dim=3,
        hidden_dims=(16, 8),
        target_dim=2,
        activation="relu",
        optimizer_name="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        loss_name="mse",
        training=TrainingSummary(
            dataset_id="example",
            feature_columns=("x1", "x2"),
            target_columns=("y",),
            val_fraction=0.2,
            max_epochs=50,
            batch_size=64,
            seed=0,
            accelerator="cpu",
            epochs_run=50,
            train_loss=0.01,
            val_loss=0.02,
            best_val_loss=0.02,
            trained_at="2026-06-27T12:00:00",
        ),
        evaluation=None,
        archived_at=None,
    )


def test_describe_model_success_reports_ok_exit():
    report = run_describe_model(
        _DESCRIBE_MODEL, describe=lambda _: Ok(_model_description())
    )
    assert report.exit_code is ExitCode.OK


def test_describe_model_renders_status_and_network():
    report = run_describe_model(
        _DESCRIBE_MODEL, describe=lambda _: Ok(_model_description())
    )
    assert "status: trained" in report.message and "sequential-mlp" in report.message


def test_describe_model_renders_the_training_run():
    rendered = render_model_description(_model_description())
    assert "targets=y" in rendered


def test_describe_model_query_failure_reports_failure_exit():
    not_found = ModelNotFound(ModelId("demo"), ErrorInfo("x", "absent"))
    report = run_describe_model(_DESCRIBE_MODEL, describe=lambda _: Err(not_found))
    assert report.exit_code is ExitCode.FAILURE


def test_describe_model_renders_a_linear_model_with_no_hidden_dims():
    linear = replace(_model_description(), hidden_dims=(), training=None)
    assert "hidden_dims=none" in render_model_description(linear)


def test_describe_model_renders_the_evaluation_run():
    evaluated = replace(
        _model_description(),
        evaluation=EvaluationSummary(
            test_loss=0.03, n_samples=1000, evaluated_at="2026-06-27T13:00:00"
        ),
    )
    assert "evaluation: test_loss=0.03" in render_model_description(evaluated)


def test_describe_model_renders_the_archived_timestamp():
    archived = replace(_model_description(), archived_at="2026-06-27T14:00:00")
    assert "archived_at: 2026-06-27T14:00:00" in render_model_description(archived)


def test_train_failure_message_carries_the_cause():
    not_found = ModelNotFound(ModelId("m1"), ErrorInfo("x", "no stored model"))
    report = run_train_model(_TRAIN_COMMAND, train=lambda _: Err(not_found))
    assert report.message.endswith("no stored model")


# --- predict: read input, run inference, write the target predictions CSV ---

_PREDICT_COMMAND = Predict(model_id="m1", source=Path("in.csv"))
_PREDICTIONS_PATH = Path("state/predictions/m1.csv")


def _predictions() -> pd.DataFrame:
    """A predictions frame carrying the model's target column."""
    return pd.DataFrame({"y": [2.0]})


def test_predict_success_reports_ok_exit():
    report = run_predict(
        _PREDICT_COMMAND,
        read_source=lambda _: Ok(pd.DataFrame({"x1": [1.0]})),
        predict=lambda _: Ok(_predictions()),
        save_predictions=lambda _: Ok(_PREDICTIONS_PATH),
    )
    assert report.exit_code is ExitCode.OK


def test_predict_read_failure_reports_failure_exit():
    report = run_predict(
        _PREDICT_COMMAND,
        read_source=lambda _: Err(ErrorInfo("cli.source.read_failed", "boom")),
        predict=lambda _: Ok(_predictions()),
        save_predictions=lambda _: Ok(_PREDICTIONS_PATH),
    )
    assert report.exit_code is ExitCode.FAILURE


def test_predict_inference_failure_reports_failure_exit():
    not_found = ModelNotFound(ModelId("m1"), ErrorInfo("x", "absent"))
    report = run_predict(
        _PREDICT_COMMAND,
        read_source=lambda _: Ok(pd.DataFrame({"x1": [1.0]})),
        predict=lambda _: Err(not_found),
        save_predictions=lambda _: Ok(_PREDICTIONS_PATH),
    )
    assert report.exit_code is ExitCode.FAILURE


def test_predict_writes_target_predictions_verbatim():
    spy = _SaveReportSpy()
    run_predict(
        _PREDICT_COMMAND,
        read_source=lambda _: Ok(pd.DataFrame({"x1": [1.0]})),
        predict=lambda _: Ok(_predictions()),
        save_predictions=spy,
    )
    header = (spy.content or "").splitlines()[0]
    assert header.split(",") == ["y"]


# --- list-models: render the stored models (id + status) ---


def test_list_models_renders_id_and_status():
    listing = ModelsListing(
        (ModelSummary("a", "trained"), ModelSummary("b", "defined"))
    )
    report = run_list_models(ListModels(), list_models=lambda _: Ok(listing))
    assert report.exit_code is ExitCode.OK and "a: trained" in report.message


def test_list_models_empty_reports_no_models():
    report = run_list_models(ListModels(), list_models=lambda _: Ok(ModelsListing(())))
    assert "no models stored" in report.message


def test_list_models_failure_reports_failure_exit():
    failed = ModelsNotListed(ErrorInfo("x", "boom"))
    report = run_list_models(ListModels(), list_models=lambda _: Err(failed))
    assert report.exit_code is ExitCode.FAILURE
