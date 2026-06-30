"""CLI delivery context -- the input/output model (this delivery's own core).

A delivery context owns its vocabulary: the argv-derived invocation
(:class:`PrepareDataset`), the exit codes it returns, and the rendered
:class:`CliReport`. Pure data -- no I/O, no argparse, and no reach into a core
context's ``domain`` (that would be a layering breach).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ExitCode(IntEnum):
    """Process exit codes the CLI returns."""

    OK = 0
    FAILURE = 1


@dataclass(frozen=True)
class PrepareDataset:
    """Invocation: read ``source`` and prepare it as the dataset named ``dataset_id``.

    ``dataset_id`` is a free-form identity the user chooses -- the prepared dataset
    is persisted (and later referenced) under it. The schema it is certified
    against is configuration (see ``Settings.required_columns``), not the id.
    """

    dataset_id: str
    source: Path


@dataclass(frozen=True)
class DescribeDataset:
    """Invocation: tabulate univariate statistics for the prepared ``dataset_id``."""

    dataset_id: str


@dataclass(frozen=True)
class DefineModel:
    """Invocation: certify a training-ready model and persist it.

    Carries edge PRIMITIVES only -- argv-derived strings/ints/floats: the recipe
    (network / optimizer / loss) AND the training binding (data columns + run knobs).
    The name-valued fields stay plain ``str`` here (the delivery's own input model
    reaches no core ``domain``); ``run_define_model`` lifts them into the mlmodel
    command's enums, total because argparse ``choices=`` already gates the values.
    """

    model_id: str
    network_name: str
    input_dim: int
    hidden_dims: tuple[int, ...]
    target_dim: int
    activation: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    loss_name: str
    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    val_fraction: float
    max_epochs: int
    batch_size: int
    seed: int
    accelerator: str
    scaler: str = "identity"


@dataclass(frozen=True)
class TrainModel:
    """Invocation: train a fully-defined model -- the identity only.

    The training binding (data + run knobs) was set at define time and lives on the
    stored model, so this invocation carries only the argv-derived model id.
    """

    model_id: str


@dataclass(frozen=True)
class EvaluateModel:
    """Invocation: evaluate a trained model on a stored (test) dataset.

    Carries edge PRIMITIVES only -- the model identity and the data binding as
    argv-derived values. No enum-valued fields, so no lift is needed at the edge.
    """

    model_id: str
    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveModel:
    """Invocation: retire a stored model -- tombstone whatever state it is in.

    Carries only the model identity as an argv-derived primitive; no enum-valued
    fields, so no lift is needed at the edge.
    """

    model_id: str


@dataclass(frozen=True)
class DescribeModel:
    """Invocation: report a stored model's lifecycle state and hyperparameters.

    Carries only the model identity as an argv-derived primitive; the read side
    needs nothing else.
    """

    model_id: str


@dataclass(frozen=True)
class Predict:
    """Invocation: run a trained model over an input file and write the predictions.

    Carries the model identity and the input ``source`` path as argv-derived
    primitives; the trained model already knows which feature columns to read.
    """

    model_id: str
    source: Path


@dataclass(frozen=True)
class ListModels:
    """Invocation: list every stored model (id + lifecycle status). Carries no fields."""


type CliCommand = (
    PrepareDataset
    | DescribeDataset
    | DefineModel
    | TrainModel
    | EvaluateModel
    | ArchiveModel
    | DescribeModel
    | Predict
    | ListModels
)
"""The parsed-invocation union -- one member per subcommand."""


@dataclass(frozen=True)
class CliReport:
    """What the CLI renders: a message for the user plus the process exit code."""

    message: str
    exit_code: ExitCode
