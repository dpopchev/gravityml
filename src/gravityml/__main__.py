"""The gravityml entry point -- the single top-level composition root.

Wires the cross-context edge (the datasets save adapter into the datasets command
handler) and drives the CLI delivery context. This is the ONLY place that reaches
into a context's infrastructure; every ring below receives its dependencies as
injected callables.

Invoked as ``uv run gravityml ...`` (the ``gravityml`` console script) or
``python -m gravityml``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from gravityml.cli.application import (
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
from gravityml.cli.infrastructure import parse_args, read_source, write_report
from gravityml.config import get_settings
from gravityml.datasets.application import (
    GetUnivariateStats,
    GetUnivariateStatsError,
    MakeDataset,
    MakeDatasetError,
    UnivariateStats,
    handle_get_univariate_stats,
    handle_make_dataset,
)
from gravityml.datasets.infrastructure import (
    make_find_frame,
    make_save_dataset,
)
from gravityml.mlmodel.application import (
    ArchiveMLModel,
    ArchiveMLModelError,
    DefineMLModel,
    DefineMLModelError,
    EvaluateMLModel,
    EvaluateMLModelError,
    GetModelDescription,
    GetModelDescriptionError,
    ListModelsError,
    ListStoredModels,
    ModelDescription,
    ModelsListing,
    PredictMLModel,
    PredictMLModelError,
    TrainMLModel,
    TrainMLModelError,
    handle_archive_model,
    handle_define_ml_model,
    handle_evaluate_model,
    handle_get_model_description,
    handle_list_models,
    handle_predict_model,
    handle_train_model,
)
from gravityml.mlmodel.infrastructure import (
    make_evaluate_model_fn,
    make_find_model_manifest,
    make_list_model_manifests,
    make_load_any_model,
    make_load_frame,
    make_load_model,
    make_load_trained_model,
    make_predict_model_fn,
    make_save_model,
    make_train_model_fn,
    system_now,
)
from gravityml.shared_kernel.result import Result


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, wire the datasets handlers, dispatch, render, return the code."""
    command = parse_args(sys.argv[1:] if argv is None else argv)

    settings = get_settings()
    datasets_dir = settings.datasets_dir
    required_columns = settings.required_columns
    save = make_save_dataset(datasets_dir)
    find = make_find_frame(datasets_dir)

    def prepare(cmd: MakeDataset) -> Result[Path, MakeDatasetError]:
        return handle_make_dataset(save, required_columns, cmd).fmap(
            lambda dataset: datasets_dir / f"{dataset.dataset_id}.parquet"
        )

    def describe(
        query: GetUnivariateStats,
    ) -> Result[UnivariateStats, GetUnivariateStatsError]:
        return handle_get_univariate_stats(find, query)

    models_dir = settings.models_dir
    save_model = make_save_model(models_dir)

    def define(cmd: DefineMLModel) -> Result[Path, DefineMLModelError]:
        return handle_define_ml_model(save_model, cmd).fmap(
            lambda defined: models_dir / f"{defined.model_id}.json"
        )

    load_frame = make_load_frame(find)
    load_model = make_load_model(models_dir)
    train_model = make_train_model_fn(load_frame, models_dir)

    def train(cmd: TrainMLModel) -> Result[Path, TrainMLModelError]:
        return handle_train_model(load_model, train_model, save_model, cmd).fmap(
            lambda trained: models_dir / f"{trained.defined.model_id}.json"
        )

    load_trained = make_load_trained_model(models_dir)
    evaluate_model = make_evaluate_model_fn(load_frame, models_dir)

    def evaluate(cmd: EvaluateMLModel) -> Result[Path, EvaluateMLModelError]:
        return handle_evaluate_model(
            load_trained, evaluate_model, save_model, cmd
        ).fmap(
            lambda evaluated: models_dir / f"{evaluated.trained.defined.model_id}.json"
        )

    load_any = make_load_any_model(models_dir)

    def archive(cmd: ArchiveMLModel) -> Result[Path, ArchiveMLModelError]:
        return handle_archive_model(load_any, system_now, save_model, cmd).fmap(
            lambda _: models_dir / f"{cmd.model_id}.json"
        )

    find_model_manifest = make_find_model_manifest(models_dir)

    def describe_model(
        query: GetModelDescription,
    ) -> Result[ModelDescription, GetModelDescriptionError]:
        return handle_get_model_description(find_model_manifest, query)

    predict_model_fn = make_predict_model_fn(models_dir)

    def predict(cmd: PredictMLModel) -> Result[pd.DataFrame, PredictMLModelError]:
        return handle_predict_model(load_trained, predict_model_fn, cmd)

    list_model_manifests = make_list_model_manifests(models_dir)

    def list_models(query: ListStoredModels) -> Result[ModelsListing, ListModelsError]:
        return handle_list_models(list_model_manifests, query)

    match command:
        case PrepareDataset():
            report = run_prepare_dataset(
                command, read_source=read_source, prepare=prepare
            )
        case DescribeDataset():
            report_path = (
                settings.reports_dir / f"{command.dataset_id}.univariate-stats.csv"
            )
            report = run_describe_dataset(
                command,
                describe=describe,
                save_report=lambda content: write_report(report_path, content),
            )
        case DefineModel():
            report = run_define_model(command, define=define)
        case TrainModel():
            report = run_train_model(command, train=train)
        case EvaluateModel():
            report = run_evaluate_model(command, evaluate=evaluate)
        case ArchiveModel():
            report = run_archive_model(command, archive=archive)
        case DescribeModel():
            report = run_describe_model(command, describe=describe_model)
        case Predict():
            predictions_path = command.source.with_name(
                f"{command.source.stem}.predictions.csv"
            )
            report = run_predict(
                command,
                read_source=read_source,
                predict=predict,
                save_predictions=lambda content: write_report(
                    predictions_path, content
                ),
            )
        case ListModels():
            report = run_list_models(command, list_models=list_models)

    return _emit(report)


def _emit(report: CliReport) -> int:
    """Print the report to the right stream and return the process exit code."""
    stream = sys.stdout if report.exit_code is ExitCode.OK else sys.stderr
    print(report.message, file=stream)
    return int(report.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
