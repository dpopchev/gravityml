"""CLI delivery context -- application layer.

Drives a core context through its PUBLISHED application API. ``run_prepare_dataset``
takes two injected ports -- ``read_source`` (acquire the input frame) and
``prepare`` (the wired datasets command handler) -- and renders the outcome to a
:class:`CliReport`. Per the delivery rule, the ``Result`` -> output ``match`` lives
here at the edge; the core never branches on a ``Result``.

This context depends only on the datasets PUBLISHED types (the command DTO and its
error union), never on its ``domain`` or ``infrastructure``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd

from gravityml.cli.domain import (
    ArchiveModel,
    CliReport,
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
from gravityml.datasets.application import (
    ColumnStats,
    DatasetId,
    GetUnivariateStats,
    GetUnivariateStatsError,
    MakeDataset,
    MakeDatasetError,
    UnivariateStats,
)
from gravityml.mlmodel.application import (
    Accelerator,
    ActivationName,
    ArchiveMLModel,
    ArchiveMLModelError,
    DefineMLModel,
    DefineMLModelError,
    EvaluateMLModel,
    EvaluateMLModelError,
    EvaluationSummary,
    GetModelDescription,
    GetModelDescriptionError,
    ListModelsError,
    ListStoredModels,
    LossName,
    ModelDescription,
    ModelsListing,
    NetworkName,
    OptimizerName,
    PredictMLModel,
    PredictMLModelError,
    ScalerName,
    TrainingSummary,
    TrainMLModel,
    TrainMLModelError,
)
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result

type ReadSourceFn = Callable[[Path], Result[pd.DataFrame, ErrorInfo]]
"""Injected port: read a source file into a candidate DataFrame, or report why not."""

type PrepareDatasetFn = Callable[[MakeDataset], Result[Path, MakeDatasetError]]
"""Injected port: the wired datasets handler, yielding the saved artifact path."""


def run_prepare_dataset(
    command: PrepareDataset,
    *,
    read_source: ReadSourceFn,
    prepare: PrepareDatasetFn,
) -> CliReport:
    """Read the source, prepare the dataset, and render the outcome for the user."""
    match read_source(command.source):
        case Err(error=read_error):
            return CliReport(
                f"could not read {command.source}: {read_error}",
                ExitCode.FAILURE,
            )
        case Ok(value=frame):
            command_to_run = MakeDataset(frame, DatasetId(command.dataset_id))
            return _render_prepared(command, prepare(command_to_run))


def _render_prepared(
    command: PrepareDataset,
    outcome: Result[Path, MakeDatasetError],
) -> CliReport:
    """Map the datasets handler's ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=saved_path):
            return CliReport(
                f"prepared {command.dataset_id} dataset -> {saved_path}", ExitCode.OK
            )
        case Err(error=prepare_error):
            return CliReport(
                f"failed to prepare {command.dataset_id} dataset: {prepare_error}",
                ExitCode.FAILURE,
            )


type DescribeDatasetFn = Callable[
    [GetUnivariateStats], Result[UnivariateStats, GetUnivariateStatsError]
]
"""Injected port: the wired univariate-stats query."""

type SaveReportFn = Callable[[str], Result[Path, ErrorInfo]]
"""Injected port: persist a rendered report, returning the written path."""

_STATS_HEADER: Final[tuple[str, ...]] = (
    "column",
    "n_unique",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "p1",
    "p5",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "skewness",
    "kurtosis",
)


def run_describe_dataset(
    command: DescribeDataset,
    *,
    describe: DescribeDatasetFn,
    save_report: SaveReportFn,
) -> CliReport:
    """Run the univariate-stats query, render the table, and persist it as CSV."""
    match describe(GetUnivariateStats(DatasetId(command.dataset_id))):
        case Err(error=query_error):
            return CliReport(
                f"could not describe {command.dataset_id} dataset: {query_error}",
                ExitCode.FAILURE,
            )
        case Ok(value=stats):
            return _render_saved(command, save_report(render_univariate_stats(stats)))


def _render_saved(
    command: DescribeDataset,
    outcome: Result[Path, ErrorInfo],
) -> CliReport:
    """Map the report-write ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=report_path):
            return CliReport(
                f"described {command.dataset_id} dataset -> {report_path}", ExitCode.OK
            )
        case Err(error=write_error):
            return CliReport(
                f"could not write {command.dataset_id} stats: {write_error}",
                ExitCode.FAILURE,
            )


def render_univariate_stats(stats: UnivariateStats) -> str:
    """Render the stats table as CSV: a header row plus one row per column."""
    rows = [",".join(_STATS_HEADER)]
    rows.extend(_render_stats_row(column) for column in stats.columns)
    return "\n".join(rows)


def _render_stats_row(column: ColumnStats) -> str:
    """One CSV row of statistics for a single column, in header order."""
    values: tuple[object, ...] = (
        column.column,
        column.n_unique,
        column.minimum,
        column.maximum,
        column.mean,
        column.median,
        column.std,
        column.p1,
        column.p5,
        column.p25,
        column.p50,
        column.p75,
        column.p95,
        column.p99,
        column.skewness,
        column.kurtosis,
    )
    return ",".join(str(value) for value in values)


type DefineModelFn = Callable[[DefineMLModel], Result[Path, DefineMLModelError]]
"""Injected port: the wired mlmodel define handler, yielding the saved manifest path."""


def run_define_model(command: DefineModel, *, define: DefineModelFn) -> CliReport:
    """Lift the argv input into the mlmodel command, define it, and render the outcome.

    The name-valued flags (network / activation / optimizer / loss / accelerator) are
    plain ``str`` at this edge; they are lifted into the mlmodel domain enums here. The
    lift is total -- argparse ``choices=`` already rejected any value outside the enum
    vocabulary -- so no ``Result`` is needed. The training binding (data + run knobs)
    rides along as primitives the handler certifies through the domain.
    """
    request = DefineMLModel(
        model_id=command.model_id,
        network_name=NetworkName(command.network_name),
        input_dim=command.input_dim,
        hidden_dims=command.hidden_dims,
        target_dim=command.target_dim,
        activation=ActivationName(command.activation),
        optimizer_name=OptimizerName(command.optimizer_name),
        learning_rate=command.learning_rate,
        weight_decay=command.weight_decay,
        loss_name=LossName(command.loss_name),
        dataset_id=command.dataset_id,
        feature_columns=command.feature_columns,
        target_columns=command.target_columns,
        val_fraction=command.val_fraction,
        max_epochs=command.max_epochs,
        batch_size=command.batch_size,
        seed=command.seed,
        accelerator=Accelerator(command.accelerator),
        scaler=ScalerName(command.scaler),
    )
    return _render_defined(command, define(request))


def _render_defined(
    command: DefineModel,
    outcome: Result[Path, DefineMLModelError],
) -> CliReport:
    """Map the define handler's ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=manifest_path):
            return CliReport(
                f"defined model {command.model_id} -> {manifest_path}", ExitCode.OK
            )
        case Err(error=define_error):
            return CliReport(
                f"failed to define model {command.model_id}: {define_error}",
                ExitCode.FAILURE,
            )


type TrainModelFn = Callable[[TrainMLModel], Result[Path, TrainMLModelError]]
"""Injected port: the wired mlmodel train handler, yielding the saved manifest path."""


def run_train_model(command: TrainModel, *, train: TrainModelFn) -> CliReport:
    """Lift the argv input into the mlmodel command, train it, and render the outcome.

    The training binding was fixed at define time and lives on the stored model, so
    this carries only the model id -- no flags, no enum lift.
    """
    request = TrainMLModel(model_id=command.model_id)
    return _render_trained(command, train(request))


def _render_trained(
    command: TrainModel,
    outcome: Result[Path, TrainMLModelError],
) -> CliReport:
    """Map the train handler's ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=manifest_path):
            return CliReport(
                f"trained model {command.model_id} -> {manifest_path}", ExitCode.OK
            )
        case Err(error=train_error):
            return CliReport(
                f"failed to train model {command.model_id}: {train_error}",
                ExitCode.FAILURE,
            )


type EvaluateModelFn = Callable[[EvaluateMLModel], Result[Path, EvaluateMLModelError]]
"""Injected port: the wired mlmodel evaluate handler, yielding the saved manifest path."""


def run_evaluate_model(
    command: EvaluateModel, *, evaluate: EvaluateModelFn
) -> CliReport:
    """Lift the argv input into the mlmodel command, evaluate it, and render the outcome.

    No enum lift is needed -- the evaluate command carries only primitive fields.
    """
    request = EvaluateMLModel(
        model_id=command.model_id,
        dataset_id=command.dataset_id,
        feature_columns=command.feature_columns,
        target_columns=command.target_columns,
    )
    return _render_evaluated(command, evaluate(request))


def _render_evaluated(
    command: EvaluateModel,
    outcome: Result[Path, EvaluateMLModelError],
) -> CliReport:
    """Map the evaluate handler's ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=manifest_path):
            return CliReport(
                f"evaluated model {command.model_id} -> {manifest_path}", ExitCode.OK
            )
        case Err(error=evaluate_error):
            return CliReport(
                f"failed to evaluate model {command.model_id}: {evaluate_error}",
                ExitCode.FAILURE,
            )


type ArchiveModelFn = Callable[[ArchiveMLModel], Result[Path, ArchiveMLModelError]]
"""Injected port: the wired mlmodel archive handler, yielding the saved manifest path."""


def run_archive_model(command: ArchiveModel, *, archive: ArchiveModelFn) -> CliReport:
    """Lift the argv input into the mlmodel command, archive it, and render the outcome.

    No enum lift is needed -- the archive command carries only the model identity.
    """
    request = ArchiveMLModel(model_id=command.model_id)
    return _render_archived(command, archive(request))


def _render_archived(
    command: ArchiveModel,
    outcome: Result[Path, ArchiveMLModelError],
) -> CliReport:
    """Map the archive handler's ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=manifest_path):
            return CliReport(
                f"archived model {command.model_id} -> {manifest_path}", ExitCode.OK
            )
        case Err(error=archive_error):
            return CliReport(
                f"failed to archive model {command.model_id}: {archive_error}",
                ExitCode.FAILURE,
            )


type DescribeModelFn = Callable[
    [GetModelDescription], Result[ModelDescription, GetModelDescriptionError]
]
"""Injected port: the wired describe-model query."""


def run_describe_model(
    command: DescribeModel, *, describe: DescribeModelFn
) -> CliReport:
    """Run the describe-model query and render the model's state + hyperparameters."""
    match describe(GetModelDescription(model_id=command.model_id)):
        case Err(error=query_error):
            return CliReport(
                f"could not describe model {command.model_id}: {query_error}",
                ExitCode.FAILURE,
            )
        case Ok(value=description):
            return CliReport(render_model_description(description), ExitCode.OK)


def render_model_description(description: ModelDescription) -> str:
    """Render a model description as readable ``key: value`` lines for stdout.

    The recipe is always shown; ``training`` / ``evaluation`` / ``archived_at`` lines
    appear only when the model has reached that state.
    """
    lines = [
        f"model: {description.model_id}",
        f"status: {description.status}",
        f"network: {description.network_name} "
        f"(input_dim={description.input_dim}, "
        f"hidden_dims={_render_dims(description.hidden_dims)}, "
        f"target_dim={description.target_dim}, "
        f"activation={description.activation})",
        f"optimizer: {description.optimizer_name} "
        f"(learning_rate={description.learning_rate}, "
        f"weight_decay={description.weight_decay})",
        f"loss: {description.loss_name}",
    ]
    if description.training is not None:
        lines.append(_render_training(description.training))
    if description.evaluation is not None:
        lines.append(_render_evaluation(description.evaluation))
    if description.archived_at is not None:
        lines.append(f"archived_at: {description.archived_at}")
    return "\n".join(lines)


def _render_dims(hidden_dims: tuple[int, ...]) -> str:
    """Render the hidden widths as ``16|8``, or ``none`` for a linear model."""
    return "|".join(str(dim) for dim in hidden_dims) if hidden_dims else "none"


def _render_training(training: TrainingSummary) -> str:
    """Render the training-run facts as one ``training: ...`` line."""
    return (
        "training: "
        f"dataset={training.dataset_id}, "
        f"features={','.join(training.feature_columns)}, "
        f"targets={','.join(training.target_columns)}, "
        f"val_fraction={training.val_fraction}, "
        f"max_epochs={training.max_epochs}, "
        f"batch_size={training.batch_size}, "
        f"seed={training.seed}, "
        f"accelerator={training.accelerator}, "
        f"epochs_run={training.epochs_run}, "
        f"train_loss={training.train_loss}, "
        f"val_loss={training.val_loss}, "
        f"best_val_loss={training.best_val_loss}, "
        f"trained_at={training.trained_at}"
    )


def _render_evaluation(evaluation: EvaluationSummary) -> str:
    """Render the evaluation-run facts as one ``evaluation: ...`` line."""
    return (
        "evaluation: "
        f"test_loss={evaluation.test_loss}, "
        f"n_samples={evaluation.n_samples}, "
        f"evaluated_at={evaluation.evaluated_at}"
    )


type PredictModelFn = Callable[
    [PredictMLModel], Result[pd.DataFrame, PredictMLModelError]
]
"""Injected port: the wired mlmodel predict handler, yielding a predictions frame."""


def run_predict(
    command: Predict,
    *,
    read_source: ReadSourceFn,
    predict: PredictModelFn,
    save_predictions: SaveReportFn,
) -> CliReport:
    """Read the input frame, run inference, and write the predictions.

    The mlmodel ``predict`` capability produces the target predictions, which are
    written out verbatim. Per the delivery rule the ``Result`` -> output match lives
    here at the edge; the core never branches on a ``Result``.
    """
    match read_source(command.source):
        case Err(error=read_error):
            return CliReport(
                f"could not read {command.source}: {read_error}", ExitCode.FAILURE
            )
        case Ok(value=frame):
            outcome = predict(PredictMLModel(model_id=command.model_id, frame=frame))
            return _render_predicted(command, outcome, save_predictions)


def _render_predicted(
    command: Predict,
    outcome: Result[pd.DataFrame, PredictMLModelError],
    save_predictions: SaveReportFn,
) -> CliReport:
    """Map the predict ``Result`` to a saved-predictions report."""
    match outcome:
        case Err(error=predict_error):
            return CliReport(
                f"failed to predict with model {command.model_id}: {predict_error}",
                ExitCode.FAILURE,
            )
        case Ok(value=predictions):
            return _render_saved_predictions(
                command, save_predictions(render_predictions(predictions))
            )


def _render_saved_predictions(
    command: Predict,
    outcome: Result[Path, ErrorInfo],
) -> CliReport:
    """Map the predictions-write ``Result`` to a user-facing report."""
    match outcome:
        case Ok(value=path):
            return CliReport(
                f"predicted with model {command.model_id} -> {path}", ExitCode.OK
            )
        case Err(error=write_error):
            return CliReport(
                f"could not write predictions for {command.model_id}: {write_error}",
                ExitCode.FAILURE,
            )


def render_predictions(predictions: pd.DataFrame) -> str:
    """Render a predictions frame as CSV (a header row plus one row per input row)."""
    return predictions.to_csv(index=False)


type ListModelsFn = Callable[[ListStoredModels], Result[ModelsListing, ListModelsError]]
"""Injected port: the wired list-models query, yielding the id + status listing."""


def run_list_models(command: ListModels, *, list_models: ListModelsFn) -> CliReport:
    """Run the list-models query and render the stored models (id + status)."""
    _ = command
    match list_models(ListStoredModels()):
        case Err(error=query_error):
            return CliReport(f"could not list models: {query_error}", ExitCode.FAILURE)
        case Ok(value=listing):
            return CliReport(render_models_listing(listing), ExitCode.OK)


def render_models_listing(listing: ModelsListing) -> str:
    """Render the stored models as ``<id>: <status>`` lines (or a no-models notice)."""
    if not listing.models:
        return "no models stored"
    return "\n".join(f"{model.model_id}: {model.status}" for model in listing.models)
