"""mlmodel bounded context -- application layer.

CQRS as ordered sections in one module:

(1) Commands -- write side. :func:`handle_define_ml_model` certifies a model's
    hyperparameters through the domain and persists the resulting ``DefinedMLModel``;
    train / evaluate / archive drive the rest of the lifecycle.

(2) Queries -- read side. :func:`handle_get_model_description` reads a stored manifest
    through the :data:`FindModelManifestFn` read port (no re-certification) and projects
    it -- the read-model bypass, never the ``load_*`` aggregate ports.

(3) Projections -- pure ``to_*`` transforms: a trusted manifest dict ->
    :class:`ModelDescription`.

Pure orchestration: no I/O lives here -- ports arrive as injected callables, and
fallible steps compose on the railway (no branching on a ``Result``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import pandas as pd

from gravityml.mlmodel.domain import (
    Accelerator,
    ActivationName,
    ArchivedMLModel,
    DataSpec,
    DefinedMLModel,
    EvaluatedMLModel,
    EvaluationFailed,
    EvaluationOutcome,
    LiveMLModel,
    LossName,
    MLModel,
    ModelId,
    ModelNotFound,
    ModelNotSaved,
    ModelsNotListed,
    ModelSpecError,
    NetworkName,
    OptimizerName,
    PredictionFailed,
    ScalerName,
    TrainedMLModel,
    TrainingFailed,
    TrainingOutcome,
    TrainSpecError,
    archive,
    define_ml_model,
    evaluate,
    make_data_spec,
    train,
)
from gravityml.shared_kernel.result import Result

# ============================ Commands (write side) ============================


@dataclass(frozen=True)
class DefineMLModel:
    """Command: certify a training-ready model and persist it.

    Carries edge primitives -- the recipe (network / optimizer / loss) AND the
    training binding (data columns + run knobs) -- that the handler re-certifies
    through the domain before anything else touches them. The binding lives here, at
    define time, so the later train command needs only the model id.
    """

    model_id: str
    network_name: NetworkName
    input_dim: int
    hidden_dims: tuple[int, ...]
    target_dim: int
    activation: ActivationName
    optimizer_name: OptimizerName
    learning_rate: float
    weight_decay: float
    loss_name: LossName
    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    val_fraction: float
    max_epochs: int
    batch_size: int
    seed: int
    accelerator: Accelerator
    scaler: ScalerName = ScalerName.IDENTITY


type SaveError = ModelNotSaved
"""Why the write port failed to persist an aggregate -- the domain's chained save
failure (carries an ``ErrorInfo`` cause)."""

type DefineMLModelError = ModelSpecError | TrainSpecError | SaveError
"""Either the recipe or the binding failed to certify, or persistence rejected the
model."""

type SaveModelFn = Callable[[MLModel], Result[MLModel, SaveError]]
"""Injected write port: persist any lifecycle state of the aggregate, returning it
or an error."""


def _widen(error: DefineMLModelError) -> DefineMLModelError:
    """Identity converter that lifts a branch error into the handler's union.

    ``Result``'s error parameter is invariant, so a ``ModelSpecError`` rail and a
    ``SaveError`` rail do not merge on their own -- each is widened here before
    joining the single ``DefineMLModelError`` channel.
    """
    return error


def handle_define_ml_model(
    save: SaveModelFn,
    cmd: DefineMLModel,
) -> Result[DefinedMLModel, DefineMLModelError]:
    """Certify the command's hyperparameters, then persist on success.

    Certification short-circuits the railway on a ``ModelSpecError``; only a
    certified ``DefinedMLModel`` reaches ``save``, whose ``SaveError`` joins the
    same error channel.
    """

    def _persist(
        defined: DefinedMLModel,
    ) -> Result[DefinedMLModel, DefineMLModelError]:
        return save(defined).fmap(lambda _: defined).fmap_err(_widen)

    return (
        define_ml_model(
            model_id=cmd.model_id,
            network_name=cmd.network_name,
            input_dim=cmd.input_dim,
            hidden_dims=cmd.hidden_dims,
            target_dim=cmd.target_dim,
            activation=cmd.activation,
            optimizer_name=cmd.optimizer_name,
            learning_rate=cmd.learning_rate,
            weight_decay=cmd.weight_decay,
            loss_name=cmd.loss_name,
            dataset_id=cmd.dataset_id,
            feature_columns=cmd.feature_columns,
            target_columns=cmd.target_columns,
            val_fraction=cmd.val_fraction,
            max_epochs=cmd.max_epochs,
            batch_size=cmd.batch_size,
            seed=cmd.seed,
            accelerator=cmd.accelerator,
            scaler_name=cmd.scaler,
        )
        .fmap_err(_widen)
        .and_then(_persist)
    )


@dataclass(frozen=True)
class TrainMLModel:
    """Command: train a fully-defined model -- the identity only.

    The training binding (data + run knobs) lives on the stored ``DefinedMLModel``
    (set at define time), so this command carries only the model id to run.
    """

    model_id: str


type LoadModelFn = Callable[[ModelId], Result[DefinedMLModel, ModelNotFound]]
"""Injected read-for-write port: hydrate the defined aggregate to train it. Typed
to the ``Defined`` state -- the happy path; a stored model in another lifecycle
state is a practical miscall handled when it arises."""

type TrainModelFn = Callable[[DefinedMLModel], Result[TrainingOutcome, TrainingFailed]]
"""Injected port: run one training run (the Lightning adapter) for the fully-defined
model -- it reads the trainer + data binding off the model -- returning the impure
outcome (checkpoint + metrics + timestamp) or a chained failure."""

type TrainMLModelError = ModelNotFound | TrainingFailed | ModelNotSaved
"""Why training failed: the model was not found, the fit failed, or the trained model
did not persist. The binding was certified at define time, so no spec error arises here."""


def _widen_train(error: TrainMLModelError) -> TrainMLModelError:
    """Identity converter lifting a branch error into the train handler's union."""
    return error


def handle_train_model(
    load: LoadModelFn,
    train_model: TrainModelFn,
    save: SaveModelFn,
    cmd: TrainMLModel,
) -> Result[TrainedMLModel, TrainMLModelError]:
    """Load the fully-defined model, run training, fold the outcome, and persist.

    The happy path: ``load`` -> ``train_model`` (reads the binding off the model) ->
    the pure ``train`` fold -> ``save``. Each fallible step joins one error channel;
    an infrastructure failure arrives already wrapped as an ``ErrorInfo`` cause.
    """

    def _persist(
        trained: TrainedMLModel,
    ) -> Result[TrainedMLModel, TrainMLModelError]:
        return save(trained).fmap(lambda _: trained).fmap_err(_widen_train)

    def _fit(defined: DefinedMLModel) -> Result[TrainedMLModel, TrainMLModelError]:
        return (
            train_model(defined)
            .fmap_err(_widen_train)
            .fmap(lambda outcome: train(defined, outcome))
            .and_then(_persist)
        )

    return load(ModelId(cmd.model_id)).fmap_err(_widen_train).and_then(_fit)


@dataclass(frozen=True)
class EvaluateMLModel:
    """Command: evaluate a trained model on a (test) dataset.

    Carries the data binding as edge primitives; the handler certifies it through the
    domain smart constructor before evaluation. Evaluation uses the WHOLE referenced
    dataset (no validation split), so ``val_fraction`` is fixed to 0.0 internally.
    """

    model_id: str
    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


type LoadTrainedModelFn = Callable[[ModelId], Result[TrainedMLModel, ModelNotFound]]
"""Injected read-for-write port: hydrate the trained aggregate to evaluate it. Typed
to the ``Trained`` state -- the guard that only a trained model is evaluated."""

type EvaluateModelFn = Callable[
    [TrainedMLModel, DataSpec], Result[EvaluationOutcome, EvaluationFailed]
]
"""Injected port: run one evaluation pass (the Lightning adapter), returning the
impure outcome (metrics + timestamp) or a chained failure."""

type EvaluateMLModelError = (
    ModelNotFound | TrainSpecError | EvaluationFailed | ModelNotSaved
)
"""Why evaluation failed: the model was not found, its data spec did not certify, the
evaluation pass failed, or the evaluated model did not persist."""

_EVALUATION_VAL_FRACTION: Final = 0.0
"""Evaluation reads the whole referenced dataset -- there is no validation split."""


def _widen_evaluate(error: EvaluateMLModelError) -> EvaluateMLModelError:
    """Identity converter lifting a branch error into the evaluate handler's union."""
    return error


def handle_evaluate_model(
    load: LoadTrainedModelFn,
    evaluate_model: EvaluateModelFn,
    save: SaveModelFn,
    cmd: EvaluateMLModel,
) -> Result[EvaluatedMLModel, EvaluateMLModelError]:
    """Load the trained model, run evaluation, fold the outcome, and persist.

    The happy path (mirror of :func:`handle_train_model`): ``load`` -> certify the
    data spec -> ``evaluate_model`` -> the pure :func:`evaluate` fold -> ``save``.
    Each fallible step joins one error channel; an infrastructure failure arrives
    already wrapped as an ``ErrorInfo`` cause.
    """

    def _persist(
        evaluated: EvaluatedMLModel,
    ) -> Result[EvaluatedMLModel, EvaluateMLModelError]:
        return save(evaluated).fmap(lambda _: evaluated).fmap_err(_widen_evaluate)

    def _run(
        trained: TrainedMLModel, data: DataSpec
    ) -> Result[EvaluatedMLModel, EvaluateMLModelError]:
        return (
            evaluate_model(trained, data)
            .fmap_err(_widen_evaluate)
            .fmap(lambda outcome: evaluate(trained, data, outcome))
            .and_then(_persist)
        )

    def _with_data(
        trained: TrainedMLModel,
    ) -> Result[EvaluatedMLModel, EvaluateMLModelError]:
        return (
            make_data_spec(
                cmd.dataset_id,
                cmd.feature_columns,
                cmd.target_columns,
                _EVALUATION_VAL_FRACTION,
            )
            .fmap_err(_widen_evaluate)
            .and_then(lambda data: _run(trained, data))
        )

    return load(ModelId(cmd.model_id)).fmap_err(_widen_evaluate).and_then(_with_data)


@dataclass(frozen=True)
class ArchiveMLModel:
    """Command: retire a stored model -- tombstone whatever live state it is in.

    Carries only the model identity; the handler hydrates whichever live state is
    persisted and folds it into a tombstone with the injected archival time.
    """

    model_id: str


type LoadAnyModelFn = Callable[[ModelId], Result[LiveMLModel, ModelNotFound]]
"""Injected read-for-write port: hydrate WHICHEVER live state is stored, to archive it.
Typed to the live union -- archival is reachable from any live state. A missing or
already-tombstoned model surfaces as ``ModelNotFound``."""

type NowFn = Callable[[], datetime]
"""Injected clock port: supplies the archival timestamp. The domain may not read the
clock, so the impure 'now' is injected and recorded by the pure :func:`archive`."""

type ArchiveMLModelError = ModelNotFound | ModelNotSaved
"""Why archival failed: the model was not found (missing or already tombstoned), or the
tombstone did not persist."""


def _widen_archive(error: ArchiveMLModelError) -> ArchiveMLModelError:
    """Identity converter lifting a branch error into the archive handler's union."""
    return error


def handle_archive_model(
    load: LoadAnyModelFn,
    now: NowFn,
    save: SaveModelFn,
    cmd: ArchiveMLModel,
) -> Result[ArchivedMLModel, ArchiveMLModelError]:
    """Load whichever live state is stored, tombstone it, and persist.

    The happy path (mirror of :func:`handle_evaluate_model`): ``load`` -> the pure
    :func:`archive` fold with the injected clock -> ``save``. ``load`` rejects a missing
    or already-archived model as ``ModelNotFound``, short-circuiting before the clock is
    read or anything is saved.
    """

    def _persist(
        archived: ArchivedMLModel,
    ) -> Result[ArchivedMLModel, ArchiveMLModelError]:
        return save(archived).fmap(lambda _: archived).fmap_err(_widen_archive)

    return (
        load(ModelId(cmd.model_id))
        .fmap_err(_widen_archive)
        .fmap(lambda model: archive(model, now()))
        .and_then(_persist)
    )


# ============================ Predict (inference) =============================
#
# Read-side inference: load the trained AGGREGATE to EXECUTE it (not to project a
# stored view), run the injected forward-pass adapter, and return the predictions
# frame. No state transition, no save -- distinct from the write commands above and
# the view queries below.


@dataclass(frozen=True)
class PredictMLModel:
    """Request: run a trained model's forward pass over an input feature frame.

    Carries the model id and the raw input ``frame`` as an edge value (a foreign
    DataFrame crossing the boundary, like the datasets context's make-dataset command);
    the adapter projects the model's ``feature_columns`` from it.
    """

    model_id: str
    frame: pd.DataFrame


type PredictModelFn = Callable[
    [TrainedMLModel, pd.DataFrame], Result[pd.DataFrame, PredictionFailed]
]
"""Injected port: run inference (the torch adapter) -- a forward pass over the input
frame's feature columns, returning a predictions frame keyed by the model's
``target_columns`` -- or a chained failure."""

type PredictMLModelError = ModelNotFound | PredictionFailed
"""Why prediction failed: the trained model was not found, or the forward pass failed."""


def _widen_predict(error: PredictMLModelError) -> PredictMLModelError:
    """Identity converter lifting a branch error into the predict handler's union."""
    return error


def handle_predict_model(
    load: LoadTrainedModelFn,
    predict_model: PredictModelFn,
    query: PredictMLModel,
) -> Result[pd.DataFrame, PredictMLModelError]:
    """Load the trained model and run inference over the input frame.

    The happy path: ``load`` (the trained aggregate -- the recipe + binding needed to
    rebuild and run the net) -> ``predict_model`` (the forward pass). No fold, no save:
    prediction reads the model and returns a predictions frame without mutating state.
    """
    return (
        load(ModelId(query.model_id))
        .fmap_err(_widen_predict)
        .and_then(
            lambda trained: predict_model(trained, query.frame).fmap_err(_widen_predict)
        )
    )


# ============================= Queries (read side) =============================


@dataclass(frozen=True)
class GetModelDescription:
    """Query: the stored lifecycle state + recipe of a model, by id.

    Carries the id as an edge ``str``; the handler lifts it into a ``ModelId``
    (mirror of the command handlers) before touching the read port.
    """

    model_id: str


type FindModelManifestFn = Callable[[ModelId], Result[dict[str, Any], ModelNotFound]]
"""Read port: load a stored model's manifest dict by id -- NO certification, NO
aggregate hydration. The read-model bypass (returns the primitive manifest dict for
projection), distinct from the ``load_*`` ports that re-certify into an aggregate."""

type GetModelDescriptionError = ModelNotFound
"""Why the describe-model query failed -- the manifest could not be read."""


def handle_get_model_description(
    find: FindModelManifestFn,
    query: GetModelDescription,
) -> Result[ModelDescription, GetModelDescriptionError]:
    """Read the stored manifest and project its display description."""
    return find(ModelId(query.model_id)).fmap(to_model_description)


@dataclass(frozen=True)
class ListStoredModels:
    """Query: enumerate every stored model (id + lifecycle status). Carries no fields."""


type FindAllModelManifestsFn = Callable[
    [], Result[list[dict[str, Any]], ModelsNotListed]
]
"""Read port: load EVERY stored model's manifest dict -- NO certification, NO aggregate
hydration. The read-model bypass over the whole model store (an empty store -> [])."""

type ListModelsError = ModelsNotListed
"""Why the list-models query failed -- the model store could not be read."""


def handle_list_models(
    find_all: FindAllModelManifestsFn,
    query: ListStoredModels,
) -> Result[ModelsListing, ListModelsError]:
    """Read every stored manifest and project the id + status listing.

    ``query`` carries no fields -- the listing ranges over the whole store -- but is kept
    for a uniform handler signature.
    """
    _ = query
    return find_all().fmap(to_models_listing)


# ======================= Projections (pure read transforms) =====================


@dataclass(frozen=True)
class TrainingSummary:
    """The training-run facts surfaced by describe-model (a flat read view)."""

    dataset_id: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    val_fraction: float
    max_epochs: int
    batch_size: int
    seed: int
    accelerator: str
    epochs_run: int
    train_loss: float | None
    val_loss: float | None
    best_val_loss: float | None
    trained_at: str


@dataclass(frozen=True)
class EvaluationSummary:
    """The evaluation-run facts surfaced by describe-model (a flat read view)."""

    test_loss: float
    n_samples: int
    evaluated_at: str


@dataclass(frozen=True)
class ModelDescription:
    """A stored model's lifecycle state + recipe, projected for display.

    ``training`` / ``evaluation`` are present only when the model has reached that
    state; ``archived_at`` is set only for a tombstoned model (whose live recipe is
    unwrapped from the manifest's nested ``previous`` state).
    """

    model_id: str
    status: str
    network_name: str
    input_dim: int
    hidden_dims: tuple[int, ...]
    target_dim: int
    activation: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    loss_name: str
    training: TrainingSummary | None
    evaluation: EvaluationSummary | None
    archived_at: str | None


def to_model_description(manifest: dict[str, Any]) -> ModelDescription:
    """Project a stored manifest to a display description (pure).

    An archived manifest nests the live recipe under ``previous``; this unwraps it
    and surfaces ``archived_at`` alongside the recipe it tombstoned. The manifest is
    trusted (written by this context's own save port) -- a read-model bypass, like
    :func:`gravityml.datasets.application.to_univariate_stats` over a stored frame.
    """
    archived_at = manifest.get("archived_at")
    recipe = manifest.get("previous", manifest)
    network = recipe["network"]
    optimizer = recipe["optimizer"]
    run = recipe.get("run")
    evaluation = recipe.get("evaluation")
    return ModelDescription(
        model_id=recipe["model_id"],
        status=manifest["status"],
        network_name=network["name"],
        input_dim=network["input_dim"],
        hidden_dims=tuple(network["hidden_dims"]),
        target_dim=network["target_dim"],
        activation=network["activation"],
        optimizer_name=optimizer["name"],
        learning_rate=optimizer["learning_rate"],
        weight_decay=optimizer["weight_decay"],
        loss_name=recipe["loss"]["name"],
        training=_training_summary(recipe, run) if run is not None else None,
        evaluation=_evaluation_summary(evaluation) if evaluation is not None else None,
        archived_at=archived_at,
    )


def _training_summary(recipe: dict[str, Any], run: dict[str, Any]) -> TrainingSummary:
    """Flatten a model's training facts into a display summary.

    The binding (trainer + data) is sourced from the recipe -- the single source set
    at define time -- and the metrics + timestamp from the run section.
    """
    trainer, data, metrics = recipe["trainer"], recipe["data"], run["metrics"]
    return TrainingSummary(
        dataset_id=data["dataset_id"],
        feature_columns=tuple(data["feature_columns"]),
        target_columns=tuple(data["target_columns"]),
        val_fraction=data["val_fraction"],
        max_epochs=trainer["max_epochs"],
        batch_size=trainer["batch_size"],
        seed=trainer["seed"],
        accelerator=trainer["accelerator"],
        epochs_run=metrics["epochs_run"],
        train_loss=metrics["train_loss"],
        val_loss=metrics["val_loss"],
        best_val_loss=metrics["best_val_loss"],
        trained_at=run["trained_at"],
    )


def _evaluation_summary(evaluation: dict[str, Any]) -> EvaluationSummary:
    """Flatten a manifest's evaluation-run section into a display summary."""
    metrics = evaluation["metrics"]
    return EvaluationSummary(
        test_loss=metrics["test_loss"],
        n_samples=metrics["n_samples"],
        evaluated_at=evaluation["evaluated_at"],
    )


@dataclass(frozen=True)
class ModelSummary:
    """One stored model's discovery view -- its id and lifecycle status."""

    model_id: str
    status: str


@dataclass(frozen=True)
class ModelsListing:
    """Every stored model as a discovery listing, in store order."""

    models: tuple[ModelSummary, ...]


def to_models_listing(manifests: list[dict[str, Any]]) -> ModelsListing:
    """Project stored manifests to the id + status discovery listing (pure).

    Each manifest carries a top-level ``model_id`` and ``status`` (a tombstone keeps both
    at the top level), so the listing reads them directly -- no aggregate hydration.
    """
    return ModelsListing(
        tuple(ModelSummary(m["model_id"], m["status"]) for m in manifests)
    )
