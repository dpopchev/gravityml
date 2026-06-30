"""CLI delivery context -- imperative shell.

Two I/O concerns: the argv grammar (argparse -> the typed :class:`PrepareDataset`
input model) and the source reader (a user-supplied file -> a candidate
DataFrame). The reader is dispatched by suffix and funnels failures through
``@safe`` into a structured :class:`ErrorInfo`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from gravityml.cli.domain import (
    ArchiveModel,
    CliCommand,
    DefineModel,
    DescribeDataset,
    DescribeModel,
    EvaluateModel,
    ListModels,
    PrepareDataset,
    Predict,
    TrainModel,
)
from gravityml.mlmodel.application import (
    Accelerator,
    ActivationName,
    LossName,
    NetworkName,
    OptimizerName,
    ScalerName,
)
from gravityml.shared_kernel.error import ErrorInfo, fmap_error
from gravityml.shared_kernel.result import Err, Result
from gravityml.shared_kernel.safe import safe


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


_READ_BY_SUFFIX: Final[dict[str, Callable[[Path], pd.DataFrame]]] = {
    ".csv": _read_csv,
    ".parquet": _read_parquet,
}

UNSUPPORTED_SOURCE_CODE: Final = "cli.source.unsupported"
SOURCE_READ_FAILED_CODE: Final = "cli.source.read_failed"
REPORT_WRITE_FAILED_CODE: Final = "cli.report.write_failed"
_CAUGHT_READ_ERRORS: Final = (OSError, ValueError)
_CAUGHT_WRITE_ERRORS: Final = (OSError,)

_DEFAULT_LEARNING_RATE: Final = 1e-3
_DEFAULT_WEIGHT_DECAY: Final = 0.0
_DEFAULT_SEED: Final = 0
_DEFAULT_VAL_FRACTION: Final = 0.0


def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated widths string into a tuple; empty -> no hidden layers."""
    stripped = raw.strip()
    if not stripped:
        return ()
    return tuple(int(part) for part in stripped.split(","))


def _parse_columns(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated column-name string into a tuple of names."""
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _add_define_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``define-model`` subcommand: the recipe AND the training binding.

    A model is fully specified here -- the recipe (network / optimizer / loss) plus the
    training binding (data columns + run knobs) -- so ``train-model`` needs only the id.
    Enum-valued flags constrain ``choices`` to the mlmodel name vocabulary so the later
    str -> enum lift in ``run_define_model`` is total.
    """
    define = subcommands.add_parser(
        "define-model",
        help="Certify a training-ready model (recipe + data/run binding) and persist it.",
    )
    define.add_argument("model_id", help="Identity to persist the model under.")
    define.add_argument(
        "--input-dim", type=int, required=True, help="Width of an input row x."
    )
    define.add_argument(
        "--target-dim", type=int, required=True, help="Width of a target row y."
    )
    define.add_argument(
        "--hidden-dims",
        type=_parse_hidden_dims,
        default=(),
        help='Comma-separated hidden widths, e.g. "16,8" (default: none = line regression).',
    )
    define.add_argument(
        "--network",
        dest="network_name",
        choices=[name.value for name in NetworkName],
        default=NetworkName.SEQUENTIAL_MLP.value,
        help="Network architecture.",
    )
    define.add_argument(
        "--activation",
        choices=[name.value for name in ActivationName],
        default=ActivationName.RELU.value,
        help="Activation between layers.",
    )
    define.add_argument(
        "--scaler",
        choices=[name.value for name in ScalerName],
        default=ScalerName.IDENTITY.value,
        help="Feature scaler: standard standardizes inputs, identity leaves them raw.",
    )
    define.add_argument(
        "--optimizer",
        dest="optimizer_name",
        choices=[name.value for name in OptimizerName],
        default=OptimizerName.ADAM.value,
        help="Optimizer.",
    )
    define.add_argument(
        "--learning-rate",
        type=float,
        default=_DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    define.add_argument(
        "--weight-decay",
        type=float,
        default=_DEFAULT_WEIGHT_DECAY,
        help="Optimizer weight decay.",
    )
    define.add_argument(
        "--loss",
        dest="loss_name",
        choices=[name.value for name in LossName],
        default=LossName.MSE.value,
        help="Training objective.",
    )
    define.add_argument(
        "--dataset",
        dest="dataset_id",
        required=True,
        help="Stored dataset id to train on.",
    )
    define.add_argument(
        "--feature-cols",
        dest="feature_columns",
        type=_parse_columns,
        required=True,
        help='Comma-separated input column names, e.g. "x1,x2".',
    )
    define.add_argument(
        "--target-cols",
        dest="target_columns",
        type=_parse_columns,
        required=True,
        help='Comma-separated target column names, e.g. "y" (or "y1,y2").',
    )
    define.add_argument(
        "--max-epochs", type=int, required=True, help="Maximum training epochs."
    )
    define.add_argument(
        "--batch-size", type=int, required=True, help="Mini-batch size."
    )
    define.add_argument(
        "--val-fraction",
        type=float,
        default=_DEFAULT_VAL_FRACTION,
        help="Validation split fraction in [0, 1).",
    )
    define.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED, help="Training RNG seed."
    )
    define.add_argument(
        "--accelerator",
        choices=[name.value for name in Accelerator],
        default=Accelerator.AUTO.value,
        help="Where training runs.",
    )


def _add_train_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``train-model`` subcommand: the model identity only.

    The data binding and run knobs were fixed at define time and live on the stored
    model, so training takes just the id.
    """
    train = subcommands.add_parser(
        "train-model", help="Train a fully-defined model (binding set at define time)."
    )
    train.add_argument("model_id", help="Identity of the defined model to train.")


def _add_evaluate_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``evaluate-model`` subcommand: the model + test-data binding."""
    evaluate = subcommands.add_parser(
        "evaluate-model", help="Evaluate a trained model on a stored dataset."
    )
    evaluate.add_argument("model_id", help="Identity of the trained model to evaluate.")
    evaluate.add_argument(
        "--dataset", dest="dataset_id", required=True, help="Stored (test) dataset id."
    )
    evaluate.add_argument(
        "--feature-cols",
        dest="feature_columns",
        type=_parse_columns,
        required=True,
        help='Comma-separated input column names, e.g. "x1,x2".',
    )
    evaluate.add_argument(
        "--target-cols",
        dest="target_columns",
        type=_parse_columns,
        required=True,
        help='Comma-separated target column names, e.g. "y" (or "y1,y2").',
    )


def _add_archive_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``archive-model`` subcommand: the model identity to tombstone."""
    archive = subcommands.add_parser(
        "archive-model", help="Retire a stored model (tombstone its current state)."
    )
    archive.add_argument("model_id", help="Identity of the model to archive.")


def _add_describe_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``describe-model`` subcommand: the model identity to inspect."""
    describe = subcommands.add_parser(
        "describe-model",
        help="Report a stored model's state and hyperparameters.",
    )
    describe.add_argument("model_id", help="Identity of the model to describe.")


def _add_predict_model_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``predict`` subcommand: a trained model + an input file to infer over."""
    predict = subcommands.add_parser(
        "predict",
        help="Run a trained model over an input file and write the predictions.",
    )
    predict.add_argument("model_id", help="Identity of the trained model to run.")
    predict.add_argument(
        "-i",
        "--input",
        dest="source",
        type=Path,
        required=True,
        help="Path to the input feature file (.csv or .parquet).",
    )


def _add_list_models_parser(subcommands: argparse._SubParsersAction) -> None:
    """Register the ``list-models`` subcommand: no arguments -- it lists the whole store."""
    subcommands.add_parser(
        "list-models", help="List every stored model with its lifecycle status."
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``gravityml`` argv grammar (the delivery's own parser)."""
    parser = argparse.ArgumentParser(
        prog="gravityml", description="Prepare and manage gravityml datasets."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser(
        "prepare-dataset", help="Certify a source file and persist it as a dataset."
    )
    prepare.add_argument(
        "dataset_id",
        help="Identity to persist the prepared dataset under.",
    )
    prepare.add_argument(
        "-i",
        "--input",
        dest="source",
        type=Path,
        required=True,
        help="Path to the source data file (.csv or .parquet).",
    )
    describe = subcommands.add_parser(
        "describe-dataset",
        help="Tabulate univariate statistics for a prepared dataset.",
    )
    describe.add_argument(
        "dataset_id",
        help="Identity of the prepared dataset to describe.",
    )
    _add_define_model_parser(subcommands)
    _add_train_model_parser(subcommands)
    _add_evaluate_model_parser(subcommands)
    _add_archive_model_parser(subcommands)
    _add_describe_model_parser(subcommands)
    _add_predict_model_parser(subcommands)
    _add_list_models_parser(subcommands)
    return parser


def parse_args(argv: Sequence[str]) -> CliCommand:
    """Parse ``argv`` into the typed input model (argparse exits on a usage error)."""
    namespace = build_parser().parse_args(argv)
    match namespace.command:
        case "prepare-dataset":
            return PrepareDataset(
                dataset_id=namespace.dataset_id, source=namespace.source
            )
        case "describe-dataset":
            return DescribeDataset(dataset_id=namespace.dataset_id)
        case "define-model":
            return DefineModel(
                model_id=namespace.model_id,
                network_name=namespace.network_name,
                input_dim=namespace.input_dim,
                hidden_dims=namespace.hidden_dims,
                target_dim=namespace.target_dim,
                activation=namespace.activation,
                optimizer_name=namespace.optimizer_name,
                learning_rate=namespace.learning_rate,
                weight_decay=namespace.weight_decay,
                loss_name=namespace.loss_name,
                dataset_id=namespace.dataset_id,
                feature_columns=namespace.feature_columns,
                target_columns=namespace.target_columns,
                val_fraction=namespace.val_fraction,
                max_epochs=namespace.max_epochs,
                batch_size=namespace.batch_size,
                seed=namespace.seed,
                accelerator=namespace.accelerator,
                scaler=namespace.scaler,
            )
        case "train-model":
            return TrainModel(model_id=namespace.model_id)
        case "evaluate-model":
            return EvaluateModel(
                model_id=namespace.model_id,
                dataset_id=namespace.dataset_id,
                feature_columns=namespace.feature_columns,
                target_columns=namespace.target_columns,
            )
        case "describe-model":
            return DescribeModel(model_id=namespace.model_id)
        case "predict":
            return Predict(model_id=namespace.model_id, source=namespace.source)
        case "list-models":
            return ListModels()
        case _:
            return ArchiveModel(model_id=namespace.model_id)


def read_source(path: Path) -> Result[pd.DataFrame, ErrorInfo]:
    """Read ``path`` into a DataFrame, dispatched by file suffix."""
    reader = _READ_BY_SUFFIX.get(path.suffix)
    if reader is None:
        return Err(
            ErrorInfo(
                UNSUPPORTED_SOURCE_CODE,
                f"unsupported source format: {path.suffix or path.name}",
            )
        )
    guarded = safe(
        _CAUGHT_READ_ERRORS,
        fmap_error(lambda cause: cause, SOURCE_READ_FAILED_CODE, where=path),
    )
    return guarded(reader)(path)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_report(path: Path, content: str) -> Result[Path, ErrorInfo]:
    """Persist a rendered report to ``path`` (creating parent dirs)."""
    guarded = safe(
        _CAUGHT_WRITE_ERRORS,
        fmap_error(lambda cause: cause, REPORT_WRITE_FAILED_CODE, where=path),
    )
    return guarded(_write_text)(path, content)
