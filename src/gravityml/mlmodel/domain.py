"""mlmodel bounded context -- functional core.

Domain types for the ML-model lifecycle. The first transition, ``DefineMLModel``,
is the entry point: :func:`define_ml_model` validates the network and optimizer
hyperparameters and assembles them into a :class:`DefinedMLModel` -- the UNTRAINED
state of the lifecycle (network + optimizer + loss, no weights yet).

Lightning never appears here. The domain holds only the pure RECIPE -- value
objects naming an architecture, an optimizer, and an objective -- that the
imperative shell later instantiates into a Lightning object graph. Keeping the
recipe pure is how an OOP-heavy framework is adapted to a functional core.

Pure and total: no I/O, no exceptions raised -- failures travel as a ``Result``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType

from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result

ModelId = NewType("ModelId", str)
"""Opaque model identity -- a plain string at the edge, distinct inside the core."""


class NetworkName(StrEnum):
    """Network architectures the domain knows; infra maps each to a torch builder."""

    SEQUENTIAL_MLP = "sequential-mlp"


class ActivationName(StrEnum):
    """Activation functions a network may use between layers."""

    IDENTITY = "identity"
    RELU = "relu"
    TANH = "tanh"


class ScalerName(StrEnum):
    """How a network standardizes its input features before the first layer.

    ``IDENTITY`` feeds features through unchanged (the default -- a no-op). ``STANDARD``
    is a StandardScaler: each feature is mapped to ``(x - mean) / std`` using per-feature
    statistics FITTED on the training data, so a feature on a wildly different scale (e.g.
    a raw ``x ~ 1e15`` against targets ~1) cannot dominate the fit. Infra maps
    each name to a torch module; the fitted statistics ride in the saved weights.
    """

    IDENTITY = "identity"
    STANDARD = "standard"


class OptimizerName(StrEnum):
    """Optimizers the domain knows; infra maps each to a torch optimizer."""

    ADAM = "adam"
    SGD = "sgd"


class LossName(StrEnum):
    """Loss functions the domain knows; infra maps each to a torch criterion."""

    MSE = "mse"
    L1 = "l1"


class Accelerator(StrEnum):
    """Where training runs; infra maps each to a Lightning ``accelerator``."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass(frozen=True)
class NetworkSpec:
    """A network architecture as pure data -- the recipe for an ``nn.Sequential``.

    The model is generic: ``input_dim`` (the width of an input ``x``) and
    ``target_dim`` (the width of a target ``y``) are configuration, not fixed to
    any one dataset. At inference the network's output has ``target_dim`` units --
    that output IS the prediction. ``hidden_dims`` may be empty: a single
    ``input_dim -> target_dim`` linear map is a line regression.
    """

    name: NetworkName
    input_dim: int
    hidden_dims: tuple[int, ...]
    target_dim: int
    activation: ActivationName
    scaler: ScalerName


@dataclass(frozen=True)
class OptimizerSpec:
    """An optimizer and its hyperparameters as pure data."""

    name: OptimizerName
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class LossSpec:
    """The training objective -- a loss function chosen by name."""

    name: LossName


@dataclass(frozen=True)
class DefinedMLModel:
    """The untrained but TRAINING-READY lifecycle state: no weights yet.

    Embeds the full recipe (network + optimizer + loss) AND the training binding
    (``trainer`` run knobs + ``data`` partition) -- so a defined model is the single
    place a model is fully specified, and ``train`` needs only the injected outcome.
    ``trainer`` / ``data`` are forward references (defined below); ``from __future__
    import annotations`` keeps them lazy.
    """

    model_id: ModelId
    network: NetworkSpec
    optimizer: OptimizerSpec
    loss: LossSpec
    trainer: TrainerSpec
    data: DataSpec


@dataclass(frozen=True)
class CheckpointRef:
    """A pure reference to persisted weights -- the ``.ckpt`` file lives in infra."""

    uri: str


@dataclass(frozen=True)
class TrainingMetrics:
    """The summary metrics one training run produced."""

    epochs_run: int
    train_loss: float
    val_loss: float
    best_val_loss: float


@dataclass(frozen=True)
class TrainerSpec:
    """The hyperparameters of a single training run -- the train command's values."""

    max_epochs: int
    batch_size: int
    seed: int
    accelerator: Accelerator


@dataclass(frozen=True)
class DataSpec:
    """Which data a run trains on, and how it is partitioned.

    ``dataset_id`` references a foreign datasets-context identity as a plain value
    (no cross-context import); ``val_fraction`` is the held-out validation share.
    ``target_columns`` is multi-target -- a single-target binding is a 1-tuple, and a
    multi-output net (e.g. ``(y1, y2)``) names each output column.
    """

    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    val_fraction: float


@dataclass(frozen=True)
class TrainingOutcome:
    """The impure result of one Lightning fit, fed back into the pure fold.

    The shell produces it (weights written, metrics measured, clock read); the
    domain only records it -- which is how the :func:`train` fold stays pure.
    """

    checkpoint: CheckpointRef
    metrics: TrainingMetrics
    trained_at: datetime


@dataclass(frozen=True)
class TrainingRun:
    """The persisted record of one training run: its outcome only.

    The trainer + data binding now lives on the :class:`DefinedMLModel` embedded in
    the trained model (the single source), so the run records just what the fit
    produced -- checkpoint, metrics, timestamp -- never a duplicate of the binding.
    """

    checkpoint: CheckpointRef
    metrics: TrainingMetrics
    trained_at: datetime


@dataclass(frozen=True)
class TrainedMLModel:
    """The trained lifecycle state: a defined model plus one training run.

    Embeds :class:`DefinedMLModel` so identity and recipe stay reachable and a
    model cannot be trained without first being defined.
    """

    defined: DefinedMLModel
    run: TrainingRun


@dataclass(frozen=True)
class EvaluationMetrics:
    """The summary metrics one evaluation pass produced."""

    test_loss: float
    n_samples: int


@dataclass(frozen=True)
class EvaluationOutcome:
    """The impure result of one evaluation pass, fed back into the pure fold.

    Mirror of :class:`TrainingOutcome`: the shell produces it (model run on held-out
    data, metrics measured, clock read); the domain only records it -- which is how
    the :func:`evaluate` fold stays pure.
    """

    metrics: EvaluationMetrics
    evaluated_at: datetime


@dataclass(frozen=True)
class EvaluationRun:
    """The persisted record of one evaluation run: its data binding and its outcome."""

    data: DataSpec
    metrics: EvaluationMetrics
    evaluated_at: datetime


@dataclass(frozen=True)
class EvaluatedMLModel:
    """The evaluated lifecycle state: a trained model plus one evaluation run.

    Embeds :class:`TrainedMLModel` so identity, recipe, and the training run stay
    reachable and a model cannot be evaluated without first being trained.
    """

    trained: TrainedMLModel
    run: EvaluationRun


type LiveMLModel = DefinedMLModel | TrainedMLModel | EvaluatedMLModel
"""The live (non-tombstoned) lifecycle states -- the states a model can still
transition out of. Archiving consumes one of these."""


@dataclass(frozen=True)
class ArchiveRecord:
    """The persisted record of an archival -- when a model was retired."""

    archived_at: datetime


@dataclass(frozen=True)
class ArchivedMLModel:
    """The terminal tombstone state: a live model retired at a point in time.

    Reachable from ANY live state -- it embeds the whole ``previous`` model so the
    identity, recipe, and any training / evaluation runs stay reachable after
    archival. The tombstone is terminal: ``previous`` is a :data:`LiveMLModel`, so a
    tombstone can never itself be archived again (the type is the guard).
    """

    previous: LiveMLModel
    record: ArchiveRecord


type MLModel = LiveMLModel | ArchivedMLModel
"""The full lifecycle union -- a model is exactly one of its states, live or
tombstoned. Persistence ranges over this whole union."""


@dataclass(frozen=True)
class NonPositiveDimensions:
    """One or more network dimensions were not strictly positive."""

    dimensions: tuple[int, ...]

    def __str__(self) -> str:
        values = ", ".join(str(d) for d in self.dimensions)
        return f"network dimensions must be strictly positive: {values}"


@dataclass(frozen=True)
class NonPositiveLearningRate:
    """The optimizer's learning rate was not strictly positive."""

    learning_rate: float

    def __str__(self) -> str:
        return f"learning rate must be strictly positive: {self.learning_rate}"


type ModelSpecError = NonPositiveDimensions | NonPositiveLearningRate
"""Why a candidate model definition failed to certify."""


@dataclass(frozen=True)
class NonPositiveTrainingSize:
    """One or more training sizes (epochs / batch) were not strictly positive."""

    sizes: tuple[int, ...]

    def __str__(self) -> str:
        values = ", ".join(str(s) for s in self.sizes)
        return f"training sizes must be strictly positive: {values}"


@dataclass(frozen=True)
class ValFractionOutOfRange:
    """The validation fraction was not in the half-open interval ``[0, 1)``."""

    val_fraction: float

    def __str__(self) -> str:
        return f"validation fraction must be in [0, 1): {self.val_fraction}"


type TrainSpecError = NonPositiveTrainingSize | ValFractionOutOfRange
"""Why a candidate training run failed to certify (trainer or data inputs)."""

type DefineModelError = ModelSpecError | TrainSpecError
"""Why a training-ready model definition failed to certify -- a recipe (network /
optimizer) input or a binding (trainer / data) input was invalid."""


@dataclass(frozen=True)
class ModelNotSaved:
    """An mlmodel aggregate failed to persist.

    The application-visible save failure: it names the ``model_id`` and
    ENCAPSULATES the producing ring's underlying error as ``cause`` (an
    :class:`~gravityml.shared_kernel.error.ErrorInfo`), so nothing is lost when an
    infrastructure error is lifted onto the railway. Infrastructure constructs it;
    the application names it in a handler's error union.
    """

    model_id: ModelId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class ModelNotFound:
    """A stored mlmodel aggregate could not be read back.

    Mirror of :class:`ModelNotSaved`: names the ``model_id`` and encapsulates the
    storage error as ``cause`` (an ``ErrorInfo``).
    """

    model_id: ModelId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class ModelsNotListed:
    """Enumerating the stored models failed inside the read adapter.

    Names no single model -- the listing read failed as a whole -- and encapsulates the
    storage error as ``cause`` (an ``ErrorInfo``), so the failure stays chainable across
    the ring boundary (mirror of :class:`ModelNotFound` without an id).
    """

    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class TrainingFailed:
    """A training run failed inside the Lightning adapter.

    Names the ``model_id`` and encapsulates the framework error as ``cause`` so the
    failure stays chainable across the ring boundary.
    """

    model_id: ModelId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class EvaluationFailed:
    """An evaluation pass failed inside the Lightning adapter.

    Mirror of :class:`TrainingFailed`: names the ``model_id`` and encapsulates the
    framework error as ``cause`` so the failure stays chainable across the boundary.
    """

    model_id: ModelId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


@dataclass(frozen=True)
class PredictionFailed:
    """A prediction (inference) pass failed inside the torch adapter.

    Mirror of :class:`EvaluationFailed`: names the ``model_id`` and encapsulates the
    framework error as ``cause`` so the failure stays chainable across the boundary.
    """

    model_id: ModelId
    cause: ErrorInfo

    def __str__(self) -> str:
        return self.cause.message


def make_network_spec(
    name: NetworkName,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    target_dim: int,
    activation: ActivationName,
    scaler: ScalerName = ScalerName.IDENTITY,
) -> Result[NetworkSpec, ModelSpecError]:
    """Certify the layer dimensions are strictly positive, then wrap the spec.

    ``scaler`` defaults to ``IDENTITY`` (no feature scaling) so existing callers are
    unchanged; ``STANDARD`` opts the network into input standardization.
    """
    non_positive = tuple(
        dim for dim in (input_dim, *hidden_dims, target_dim) if dim <= 0
    )
    if non_positive:
        return Err(NonPositiveDimensions(non_positive))
    return Ok(NetworkSpec(name, input_dim, hidden_dims, target_dim, activation, scaler))


def make_optimizer_spec(
    name: OptimizerName,
    learning_rate: float,
    weight_decay: float,
) -> Result[OptimizerSpec, ModelSpecError]:
    """Certify the learning rate is strictly positive, then wrap the spec."""
    if learning_rate <= 0:
        return Err(NonPositiveLearningRate(learning_rate))
    return Ok(OptimizerSpec(name, learning_rate, weight_decay))


def _widen_define(error: DefineModelError) -> DefineModelError:
    """Identity converter lifting a branch error onto ``define_ml_model``'s union.

    ``Result``'s error parameter is invariant, so the recipe rails (``ModelSpecError``)
    and the binding rails (``TrainSpecError``) do not merge on their own -- each is
    widened here onto the single ``DefineModelError`` channel.
    """
    return error


def define_ml_model(
    *,
    model_id: str,
    network_name: NetworkName,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    target_dim: int,
    activation: ActivationName,
    optimizer_name: OptimizerName,
    learning_rate: float,
    weight_decay: float,
    loss_name: LossName,
    dataset_id: str,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    val_fraction: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    accelerator: Accelerator,
    scaler_name: ScalerName = ScalerName.IDENTITY,
) -> Result[DefinedMLModel, DefineModelError]:
    """Validate the recipe + binding and assemble a training-ready ``DefinedMLModel``.

    The network, optimizer, trainer, and data specs are certified on a single error
    channel in that order; the first failure short-circuits, so an invalid network
    never reaches the optimizer check and an invalid trainer never reaches the data
    check.
    """

    def _assemble(
        network: NetworkSpec,
        optimizer: OptimizerSpec,
        trainer: TrainerSpec,
        data: DataSpec,
    ) -> DefinedMLModel:
        return DefinedMLModel(
            ModelId(model_id), network, optimizer, LossSpec(loss_name), trainer, data
        )

    def _with_trainer(
        network: NetworkSpec, optimizer: OptimizerSpec, trainer: TrainerSpec
    ) -> Result[DefinedMLModel, DefineModelError]:
        return (
            make_data_spec(dataset_id, feature_columns, target_columns, val_fraction)
            .fmap_err(_widen_define)
            .fmap(lambda data: _assemble(network, optimizer, trainer, data))
        )

    def _with_optimizer(
        network: NetworkSpec, optimizer: OptimizerSpec
    ) -> Result[DefinedMLModel, DefineModelError]:
        return (
            make_trainer_spec(max_epochs, batch_size, seed, accelerator)
            .fmap_err(_widen_define)
            .and_then(lambda trainer: _with_trainer(network, optimizer, trainer))
        )

    def _with_network(network: NetworkSpec) -> Result[DefinedMLModel, DefineModelError]:
        return (
            make_optimizer_spec(optimizer_name, learning_rate, weight_decay)
            .fmap_err(_widen_define)
            .and_then(lambda optimizer: _with_optimizer(network, optimizer))
        )

    return (
        make_network_spec(
            network_name, input_dim, hidden_dims, target_dim, activation, scaler_name
        )
        .fmap_err(_widen_define)
        .and_then(_with_network)
    )


def make_trainer_spec(
    max_epochs: int,
    batch_size: int,
    seed: int,
    accelerator: Accelerator,
) -> Result[TrainerSpec, TrainSpecError]:
    """Certify the training sizes are strictly positive, then wrap the spec."""
    non_positive = tuple(size for size in (max_epochs, batch_size) if size <= 0)
    if non_positive:
        return Err(NonPositiveTrainingSize(non_positive))
    return Ok(TrainerSpec(max_epochs, batch_size, seed, accelerator))


def make_data_spec(
    dataset_id: str,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    val_fraction: float,
) -> Result[DataSpec, TrainSpecError]:
    """Certify the validation fraction is in ``[0, 1)``, then wrap the spec."""
    if not 0 <= val_fraction < 1:
        return Err(ValFractionOutOfRange(val_fraction))
    return Ok(DataSpec(dataset_id, feature_columns, target_columns, val_fraction))


def train(defined: DefinedMLModel, outcome: TrainingOutcome) -> TrainedMLModel:
    """Fold the injected outcome into a :class:`TrainedMLModel`.

    Total and pure: the trainer + data binding lives on ``defined`` (the single
    source), so this records only the impure ``outcome`` (weights, metrics,
    timestamp) the Lightning adapter produced. The input type being ``DefinedMLModel``
    is the guard that a model is trained only once defined.
    """
    run = TrainingRun(
        checkpoint=outcome.checkpoint,
        metrics=outcome.metrics,
        trained_at=outcome.trained_at,
    )
    return TrainedMLModel(defined=defined, run=run)


def evaluate(
    trained: TrainedMLModel,
    data: DataSpec,
    outcome: EvaluationOutcome,
) -> EvaluatedMLModel:
    """Fold one evaluation run's data binding and injected outcome into an
    :class:`EvaluatedMLModel`.

    Total and pure (mirror of :func:`train`): the impure ``outcome`` (metrics,
    timestamp) was produced by the Lightning adapter; here it is only recorded. The
    input type being ``TrainedMLModel`` is the guard that a model is evaluated only
    once trained.
    """
    run = EvaluationRun(
        data=data,
        metrics=outcome.metrics,
        evaluated_at=outcome.evaluated_at,
    )
    return EvaluatedMLModel(trained=trained, run=run)


def archive(model: LiveMLModel, archived_at: datetime) -> ArchivedMLModel:
    """Fold any live state and the injected archival time into a tombstone.

    Total and pure (mirror of :func:`train` / :func:`evaluate`): ``archived_at`` is
    the shell's clock value -- the domain forbids reading the clock itself -- recorded
    here unchanged. The input type being :data:`LiveMLModel` is the guard that a
    tombstone is never archived again, and embedding ``model`` whole preserves the
    identity and every prior run.
    """
    return ArchivedMLModel(
        previous=model, record=ArchiveRecord(archived_at=archived_at)
    )
