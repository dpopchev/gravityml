"""mlmodel bounded context -- imperative shell.

The Lightning adapter: it turns the domain's pure model RECIPE into a live PyTorch
object graph and runs it. :func:`make_train_model_fn` builds an ``nn.Sequential``
from a ``NetworkSpec``, an optimizer from an ``OptimizerSpec``, a loss from a
``LossSpec``, and a ``DataLoader`` from a frame fetched via the injected load-frame
port; it runs ``lightning.Trainer.fit``, writes the trained weights as pickle-free
safetensors (the net is rebuilt from the spec on load, so the model object is never
pickled), and returns a ``TrainingOutcome``.

This is the ONE ring that imports ``torch`` / ``lightning`` -- the OOP framework is
confined here, behind the application's ``TrainModelFn`` port (implemented
STRUCTURALLY via DIP; this ring never imports ``application``). All training I/O is
funnelled through ``@safe``; a framework failure becomes the domain's
``TrainingFailed`` with the underlying error encapsulated as an ``ErrorInfo`` cause.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import lightning as L
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gravityml.datasets.application import DatasetId, FindFrameFn
from gravityml.mlmodel.domain import (
    Accelerator,
    ActivationName,
    ArchivedMLModel,
    CheckpointRef,
    DataSpec,
    DefineModelError,
    DefinedMLModel,
    EvaluatedMLModel,
    EvaluationFailed,
    EvaluationMetrics,
    EvaluationOutcome,
    EvaluationRun,
    LiveMLModel,
    LossName,
    MLModel,
    ModelId,
    ModelNotFound,
    ModelNotSaved,
    ModelsNotListed,
    NetworkName,
    NetworkSpec,
    OptimizerName,
    OptimizerSpec,
    PredictionFailed,
    ScalerName,
    TrainedMLModel,
    TrainingFailed,
    TrainingMetrics,
    TrainingOutcome,
    TrainingRun,
    define_ml_model,
)
from gravityml.shared_kernel.error import ErrorInfo, fmap_error
from gravityml.shared_kernel.result import Result
from gravityml.shared_kernel.safe import safe

type LoadFrameFn = Callable[[str], Result[pd.DataFrame, ErrorInfo]]
"""Injected port: fetch a training frame by dataset id -- the datasets seam."""


def make_load_frame(find_frame: FindFrameFn) -> LoadFrameFn:
    """Adapt the datasets find-frame port to the local ``LoadFrameFn`` (the ACL).

    The consuming-side anti-corruption layer: it lifts the plain ``str`` dataset id
    into the datasets ``DatasetId`` and unwraps the foreign ``DatasetNotFound`` to its
    encapsulated ``ErrorInfo`` cause, so the mlmodel ring depends only on its own
    primitive seam -- never on a datasets domain type. The real ``find_frame`` is
    injected at the top-level composition root.
    """

    def load_frame(dataset_id: str) -> Result[pd.DataFrame, ErrorInfo]:
        return find_frame(DatasetId(dataset_id)).fmap_err(
            lambda not_found: not_found.cause
        )

    return load_frame


type TrainModelFn = Callable[[DefinedMLModel], Result[TrainingOutcome, TrainingFailed]]
"""Concrete shape of the train port -- structurally matches the application's
``TrainModelFn`` without importing it. The trainer + data binding is read off the
fully-defined model, so the port takes only the model."""

TRAIN_FAILED_CODE: Final = "mlmodel.training.failed"
"""``ErrorInfo`` code for a failed training run."""

_CAUGHT_TRAIN_ERRORS: Final = (RuntimeError, ValueError, OSError)
"""Exceptions treated as a training failure -- torch/lightning and filesystem errors."""

_ACTIVATIONS: Final[dict[ActivationName, Callable[[], nn.Module]]] = {
    ActivationName.IDENTITY: nn.Identity,
    ActivationName.RELU: nn.ReLU,
    ActivationName.TANH: nn.Tanh,
}

_LOSSES: Final[dict[LossName, Callable[[], nn.Module]]] = {
    LossName.MSE: nn.MSELoss,
    LossName.L1: nn.L1Loss,
}

_OPTIMIZERS: Final[dict[OptimizerName, Callable[..., torch.optim.Optimizer]]] = {
    OptimizerName.ADAM: torch.optim.Adam,
    OptimizerName.SGD: torch.optim.SGD,
}


class _Standardize(nn.Module):
    """A non-trainable StandardScaler layer: ``(x - mean) / std`` per feature.

    The per-feature ``mean`` / ``std`` are registered BUFFERS, not parameters -- the
    optimizer never touches them, but they ride in ``state_dict`` so they save to the
    ``.safetensors`` weights and reload at eval / predict. They are fit from the
    training inputs by :meth:`fit`; until then they are the identity (mean 0, std 1),
    so a freshly built net is well-defined before fitting. The two class annotations
    type the buffer attributes as tensors (``nn.Module.__getattr__`` otherwise widens
    them to ``Tensor | Module``).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))

    def fit(self, inputs: torch.Tensor) -> None:
        """Set the buffers to the per-feature mean and population std of ``inputs``.

        A zero std (a constant feature) is floored to 1.0, so a degenerate column
        passes through centred rather than dividing by zero.
        """
        mean = inputs.mean(dim=0)
        std = inputs.std(dim=0, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        with torch.no_grad():
            self.mean.copy_(mean)
            self.std.copy_(std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) / self.std


def _build_sequential_mlp(spec: NetworkSpec) -> nn.Sequential:
    """Build linear layers from the spec; hidden layers carry the activation.

    A ``STANDARD`` scaler prepends a :class:`_Standardize` layer (index 0), so every
    forward -- train, eval, predict -- standardizes inputs with the SAME fitted stats.
    ``IDENTITY`` prepends nothing, leaving the layer indices (and any pre-scaler saved
    weights) byte-identical. The final ``-> target_dim`` layer is linear (a regression
    head); empty ``hidden_dims`` yields a single linear map -- a line regression.
    """
    activation = _ACTIVATIONS[spec.activation]
    layers: list[nn.Module] = []
    if spec.scaler is ScalerName.STANDARD:
        layers.append(_Standardize(spec.input_dim))
    in_dim = spec.input_dim
    for hidden in spec.hidden_dims:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(activation())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, spec.target_dim))
    return nn.Sequential(*layers)


def _fit_scaler(net: nn.Sequential, inputs: torch.Tensor, scaler: ScalerName) -> None:
    """Fit a prepended :class:`_Standardize` layer's stats from the training inputs.

    A no-op unless the scaler is ``STANDARD`` -- in which case the layer at index 0 is
    that standardizer (the build invariant). The structural ``match`` both finds and
    types the layer (no ``isinstance``), then delegates the fit to it.
    """
    if scaler is not ScalerName.STANDARD:
        return
    match net[0]:
        case _Standardize() as standardize:
            standardize.fit(inputs)


_NETWORK_BUILDERS: Final[dict[NetworkName, Callable[[NetworkSpec], nn.Sequential]]] = {
    NetworkName.SEQUENTIAL_MLP: _build_sequential_mlp,
}


class _RegressionModule(L.LightningModule):
    """LightningModule wrapping the recipe's net, loss, and optimizer spec.

    Pure OOP glue -- it exists only to drive ``Trainer.fit``; the domain never sees
    it. ``train_loss`` / ``val_loss`` are logged per epoch for the outcome metrics.
    """

    def __init__(
        self, net: nn.Module, loss: nn.Module, optimizer: OptimizerSpec
    ) -> None:
        super().__init__()
        self._net = net
        self._loss = loss
        self._optimizer = optimizer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._net(inputs)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        loss = self._loss(self._net(inputs), targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        loss = self._loss(self._net(inputs), targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return _OPTIMIZERS[self._optimizer.name](
            self.parameters(),
            lr=self._optimizer.learning_rate,
            weight_decay=self._optimizer.weight_decay,
        )


def _to_tensors(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the frame's feature / target columns into float32 tensors.

    ``target_columns`` is multi-target: the target tensor has one column per name, so
    its width matches the net's ``target_dim`` (1 for a single-target 1-tuple).
    """
    inputs = torch.tensor(
        frame.loc[:, list(feature_columns)].to_numpy(), dtype=torch.float32
    )
    targets = torch.tensor(
        frame.loc[:, list(target_columns)].to_numpy(), dtype=torch.float32
    )
    return inputs, targets


def _loaders(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_fraction: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None]:
    """Split by ``val_fraction`` and wrap as train / (optional) val DataLoaders.

    The validation rows are carved AFTER a seeded permutation, so ``seed`` decides
    WHICH rows validate -- a representative split, not the head of a possibly
    row-ordered frame -- and a given seed reproduces it. The permutation uses its own
    generator (independent of the global RNG that drives per-epoch train shuffling).
    With ``val_fraction == 0`` there is no split, so the frame order is left untouched.
    """
    n_val = int(len(inputs) * val_fraction)
    if n_val == 0:
        train_loader = DataLoader(
            TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True
        )
        return train_loader, None
    perm = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(seed))
    inputs, targets = inputs[perm], targets[perm]
    train_loader = DataLoader(
        TensorDataset(inputs[n_val:], targets[n_val:]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(inputs[:n_val], targets[:n_val]), batch_size=batch_size
    )
    return train_loader, val_loader


def _metric(engine: L.Trainer, key: str) -> float:
    """Read one logged metric as a float (NaN if it was never logged)."""
    value = engine.callback_metrics.get(key)
    return float(value) if value is not None else float("nan")


def _best_val_loss(val_losses: list[float]) -> float:
    """The best (minimum) validation loss across epochs -- NaN if never validated.

    ``callback_metrics`` only retains the LAST epoch's ``val_loss``; the run's best is
    the minimum over the whole history, which an early epoch may own when a later one
    overfits. With no validation pass the history is empty and the best is NaN.
    """
    return min(val_losses) if val_losses else float("nan")


class _ValLossHistory(L.Callback):
    """Collects each epoch's validation loss so the run can report the best (min).

    Pure glue -- the domain never sees it. The sanity-check pass (an untrained forward
    Lightning runs before epoch 0) is skipped so it cannot masquerade as a real epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.val_losses: list[float] = []

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_loss")
        if value is not None:
            self.val_losses.append(float(value))


def _run_fit(
    defined: DefinedMLModel,
    frame: pd.DataFrame,
    models_dir: Path,
) -> TrainingOutcome:
    """Assemble the object graph, fit it, write the checkpoint, read the metrics.

    The trainer + data binding is read off the fully-defined model (the single
    source) rather than passed as separate arguments.
    """
    trainer, data = defined.trainer, defined.data
    L.seed_everything(trainer.seed, workers=True)
    inputs, targets = _to_tensors(frame, data.feature_columns, data.target_columns)
    train_loader, val_loader = _loaders(
        inputs, targets, data.val_fraction, trainer.batch_size, trainer.seed
    )
    net = _NETWORK_BUILDERS[defined.network.name](defined.network)
    _fit_scaler(net, inputs, defined.network.scaler)
    module = _RegressionModule(net, _LOSSES[defined.loss.name](), defined.optimizer)
    history = _ValLossHistory()
    engine = L.Trainer(
        max_epochs=trainer.max_epochs,
        accelerator=trainer.accelerator.value,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[history],
    )
    engine.fit(module, train_loader, val_loader)

    models_dir.mkdir(parents=True, exist_ok=True)
    weights = models_dir / f"{defined.model_id}.safetensors"
    save_file(net.state_dict(), str(weights))

    return TrainingOutcome(
        checkpoint=CheckpointRef(str(weights)),
        metrics=TrainingMetrics(
            epochs_run=int(engine.current_epoch),
            train_loss=_metric(engine, "train_loss"),
            val_loss=_metric(engine, "val_loss"),
            best_val_loss=_best_val_loss(history.val_losses),
        ),
        trained_at=datetime.now(),  # noqa: DTZ005 -- shell clock; the domain only records it
    )


def make_train_model_fn(load_frame: LoadFrameFn, models_dir: Path) -> TrainModelFn:
    """Build the Lightning training adapter the composition root injects.

    ``load_frame`` is the injected data seam (fetch the frame by dataset id);
    ``models_dir`` is where checkpoints are written. The returned callable is the
    concrete adapter bound where the application expects a ``TrainModelFn``.
    """

    def train_model(
        defined: DefinedMLModel,
    ) -> Result[TrainingOutcome, TrainingFailed]:
        guarded = safe(
            _CAUGHT_TRAIN_ERRORS,
            fmap_error(
                lambda cause: TrainingFailed(defined.model_id, cause),
                TRAIN_FAILED_CODE,
            ),
        )
        return (
            load_frame(defined.data.dataset_id)
            .fmap_err(lambda cause: TrainingFailed(defined.model_id, cause))
            .and_then(lambda frame: guarded(_run_fit)(defined, frame, models_dir))
        )

    return train_model


type EvaluateModelFn = Callable[
    [TrainedMLModel, DataSpec], Result[EvaluationOutcome, EvaluationFailed]
]
"""Concrete shape of the evaluate port -- structurally matches the application's
``EvaluateModelFn`` without importing it."""

EVAL_FAILED_CODE: Final = "mlmodel.evaluation.failed"
"""``ErrorInfo`` code for a failed evaluation pass."""

_CAUGHT_EVAL_ERRORS: Final = (RuntimeError, ValueError, OSError)
"""Exceptions treated as an evaluation failure -- torch and filesystem errors."""


def _run_eval(
    trained: TrainedMLModel, data: DataSpec, frame: pd.DataFrame, models_dir: Path
) -> EvaluationOutcome:
    """Rebuild the net, load the trained weights, run a forward pass, measure the loss.

    No Lightning here -- evaluation is a single ``no_grad`` forward over the whole
    referenced frame; the spec's loss becomes ``test_loss``.
    """
    defined = trained.defined
    net = _NETWORK_BUILDERS[defined.network.name](defined.network)
    net.load_state_dict(load_file(str(models_dir / f"{defined.model_id}.safetensors")))
    net.eval()
    inputs, targets = _to_tensors(frame, data.feature_columns, data.target_columns)
    loss = _LOSSES[defined.loss.name]()
    with torch.no_grad():
        test_loss = float(loss(net(inputs), targets))
    return EvaluationOutcome(
        metrics=EvaluationMetrics(test_loss=test_loss, n_samples=int(len(inputs))),
        evaluated_at=datetime.now(),  # noqa: DTZ005 -- shell clock; the domain only records it
    )


def make_evaluate_model_fn(
    load_frame: LoadFrameFn, models_dir: Path
) -> EvaluateModelFn:
    """Build the evaluation adapter the composition root injects.

    ``load_frame`` is the injected data seam (fetch the test frame by dataset id);
    ``models_dir`` is where the trained ``.safetensors`` weights live. A framework or
    filesystem failure becomes the domain's ``EvaluationFailed`` with the underlying
    error encapsulated as an ``ErrorInfo`` cause.
    """

    def evaluate_model(
        trained: TrainedMLModel, data: DataSpec
    ) -> Result[EvaluationOutcome, EvaluationFailed]:
        guarded = safe(
            _CAUGHT_EVAL_ERRORS,
            fmap_error(
                lambda cause: EvaluationFailed(trained.defined.model_id, cause),
                EVAL_FAILED_CODE,
            ),
        )
        return (
            load_frame(data.dataset_id)
            .fmap_err(lambda cause: EvaluationFailed(trained.defined.model_id, cause))
            .and_then(
                lambda frame: guarded(_run_eval)(trained, data, frame, models_dir)
            )
        )

    return evaluate_model


type PredictModelFn = Callable[
    [TrainedMLModel, pd.DataFrame], Result[pd.DataFrame, PredictionFailed]
]
"""Concrete shape of the predict port -- a forward pass over an input feature frame
returning a predictions frame keyed by the model's ``target_columns``; structurally
matches the application's ``PredictModelFn`` without importing it."""

PREDICT_FAILED_CODE: Final = "mlmodel.prediction.failed"
"""``ErrorInfo`` code for a failed prediction (inference) pass."""

_CAUGHT_PREDICT_ERRORS: Final = (RuntimeError, ValueError, OSError, KeyError)
"""Exceptions treated as a prediction failure -- torch, filesystem, and a missing
feature column (``KeyError``) in the input frame."""


def _run_predict(
    trained: TrainedMLModel, frame: pd.DataFrame, models_dir: Path
) -> pd.DataFrame:
    """Rebuild the net, load trained weights, forward over the frame's features.

    Mirror of :func:`_run_eval` without a loss: a single ``no_grad`` forward over the
    model's ``feature_columns`` yields one prediction row per input row, returned as a
    frame keyed by the model's ``target_columns``.
    """
    defined = trained.defined
    net = _NETWORK_BUILDERS[defined.network.name](defined.network)
    net.load_state_dict(load_file(str(models_dir / f"{defined.model_id}.safetensors")))
    net.eval()
    inputs = torch.tensor(
        frame.loc[:, list(defined.data.feature_columns)].to_numpy(),
        dtype=torch.float32,
    )
    with torch.no_grad():
        outputs = net(inputs).numpy()
    return pd.DataFrame(outputs, columns=list(defined.data.target_columns))


def make_predict_model_fn(models_dir: Path) -> PredictModelFn:
    """Build the inference adapter the composition root injects.

    Mirror of :func:`make_evaluate_model_fn` minus the loss and persistence: it loads
    the trained ``.safetensors`` weights and runs a forward pass. A framework,
    filesystem, or missing-column failure becomes the domain's ``PredictionFailed``
    with the underlying error encapsulated as an ``ErrorInfo`` cause.
    """

    def predict_model(
        trained: TrainedMLModel, frame: pd.DataFrame
    ) -> Result[pd.DataFrame, PredictionFailed]:
        guarded = safe(
            _CAUGHT_PREDICT_ERRORS,
            fmap_error(
                lambda cause: PredictionFailed(trained.defined.model_id, cause),
                PREDICT_FAILED_CODE,
            ),
        )
        return guarded(_run_predict)(trained, frame, models_dir)

    return predict_model


# --------------------- persistence: JSON manifest (pickle-free) ---------------------

type SaveModelFn = Callable[[MLModel], Result[MLModel, ModelNotSaved]]
"""Concrete shape of the save port -- structurally matches the application's
``SaveModelFn`` without importing it."""

type LoadModelFn = Callable[[ModelId], Result[DefinedMLModel, ModelNotFound]]
"""Concrete shape of the load port -- hydrates the defined recipe, re-certified."""

MANIFEST_WRITE_FAILED_CODE: Final = "mlmodel.storage.write_failed"
"""``ErrorInfo`` code for a failed manifest write."""

MANIFEST_READ_FAILED_CODE: Final = "mlmodel.storage.read_failed"
"""``ErrorInfo`` code for a failed manifest read or an invalid manifest."""


def _spec_manifest(defined: DefinedMLModel) -> dict[str, object]:
    """The training-ready recipe half of a manifest -- the fields ``define_ml_model``
    needs back: the recipe (network / optimizer / loss) AND the embedded training
    binding (trainer run knobs + data partition), the single source of the binding.
    """
    return {
        "model_id": defined.model_id,
        "network": {
            "name": defined.network.name.value,
            "input_dim": defined.network.input_dim,
            "hidden_dims": list(defined.network.hidden_dims),
            "target_dim": defined.network.target_dim,
            "activation": defined.network.activation.value,
            "scaler": defined.network.scaler.value,
        },
        "optimizer": {
            "name": defined.optimizer.name.value,
            "learning_rate": defined.optimizer.learning_rate,
            "weight_decay": defined.optimizer.weight_decay,
        },
        "loss": {"name": defined.loss.name.value},
        "trainer": {
            "max_epochs": defined.trainer.max_epochs,
            "batch_size": defined.trainer.batch_size,
            "seed": defined.trainer.seed,
            "accelerator": defined.trainer.accelerator.value,
        },
        "data": {
            "dataset_id": defined.data.dataset_id,
            "feature_columns": list(defined.data.feature_columns),
            "target_columns": list(defined.data.target_columns),
            "val_fraction": defined.data.val_fraction,
        },
    }


def _run_manifest(run: TrainingRun) -> dict[str, object]:
    """The training-run half of a manifest -- the outcome only.

    The trainer + data binding lives on the recipe (``_spec_manifest``) as the single
    source, so the run section records just the checkpoint, metrics, and timestamp.
    """
    return {
        "checkpoint": run.checkpoint.uri,
        "metrics": {
            "epochs_run": run.metrics.epochs_run,
            "train_loss": run.metrics.train_loss,
            "val_loss": run.metrics.val_loss,
            "best_val_loss": run.metrics.best_val_loss,
        },
        "trained_at": run.trained_at.isoformat(),
    }


def _evaluation_manifest(run: EvaluationRun) -> dict[str, object]:
    """The evaluation-run half of a manifest, written for an evaluated model.

    The data block is inlined (not a shared helper with ``_run_manifest``) to keep
    this change from editing the adjacent training mapper.
    """
    return {
        "data": {
            "dataset_id": run.data.dataset_id,
            "feature_columns": list(run.data.feature_columns),
            "target_columns": list(run.data.target_columns),
            "val_fraction": run.data.val_fraction,
        },
        "metrics": {
            "test_loss": run.metrics.test_loss,
            "n_samples": run.metrics.n_samples,
        },
        "evaluated_at": run.evaluated_at.isoformat(),
    }


def _to_manifest(model: MLModel) -> dict[str, object]:
    """Serialize a model state to a JSON-ready manifest (pure mapper)."""
    match model:
        case DefinedMLModel():
            return {"status": "defined", **_spec_manifest(model)}
        case TrainedMLModel():
            return {
                "status": "trained",
                **_spec_manifest(model.defined),
                "run": _run_manifest(model.run),
            }
        case EvaluatedMLModel():
            return {
                "status": "evaluated",
                **_spec_manifest(model.trained.defined),
                "run": _run_manifest(model.trained.run),
                "evaluation": _evaluation_manifest(model.run),
            }
        case ArchivedMLModel():
            previous = _to_manifest(model.previous)
            return {
                "status": "archived",
                "model_id": previous["model_id"],
                "archived_at": model.record.archived_at.isoformat(),
                "previous": previous,
            }


def _defined_from_manifest(
    manifest: dict[str, Any],
) -> Result[DefinedMLModel, DefineModelError]:
    """Re-certify the training-ready recipe half of a manifest into a ``DefinedMLModel``.

    The trust boundary: the stored recipe AND the embedded training binding pass back
    through ``define_ml_model``, so a corrupt manifest cannot smuggle an invalid model
    into the domain.
    """
    network = manifest["network"]
    optimizer = manifest["optimizer"]
    trainer = manifest["trainer"]
    data = manifest["data"]
    return define_ml_model(
        model_id=manifest["model_id"],
        network_name=NetworkName(network["name"]),
        input_dim=network["input_dim"],
        hidden_dims=tuple(network["hidden_dims"]),
        target_dim=network["target_dim"],
        activation=ActivationName(network["activation"]),
        scaler_name=ScalerName(network.get("scaler", ScalerName.IDENTITY.value)),
        optimizer_name=OptimizerName(optimizer["name"]),
        learning_rate=optimizer["learning_rate"],
        weight_decay=optimizer["weight_decay"],
        loss_name=LossName(manifest["loss"]["name"]),
        dataset_id=data["dataset_id"],
        feature_columns=tuple(data["feature_columns"]),
        target_columns=tuple(data["target_columns"]),
        val_fraction=data["val_fraction"],
        max_epochs=trainer["max_epochs"],
        batch_size=trainer["batch_size"],
        seed=trainer["seed"],
        accelerator=Accelerator(trainer["accelerator"]),
    )


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write the manifest as indented JSON, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read and parse a manifest JSON file."""
    return json.loads(path.read_text())


def make_save_model(models_dir: Path) -> SaveModelFn:
    """Build the save port persisting a model's manifest at ``<dir>/<id>.json``.

    The weights (a ``.safetensors`` file) are written by the training adapter; this
    port persists the recipe + lifecycle metadata that reference them.
    """

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        manifest = _to_manifest(model)
        model_id = ModelId(str(manifest["model_id"]))
        path = models_dir / f"{model_id}.json"
        guarded = safe(
            (OSError,),
            fmap_error(
                lambda cause: ModelNotSaved(model_id, cause),
                MANIFEST_WRITE_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_write_manifest)(path, manifest).fmap(lambda _: model)

    return save


def make_load_model(models_dir: Path) -> LoadModelFn:
    """Build the load port reading ``<dir>/<id>.json`` and re-certifying the recipe.

    Returns a ``DefinedMLModel`` -- the recipe -- for a command to act on; loading
    the trained weights for inference is a separate concern.
    """

    def load(model_id: ModelId) -> Result[DefinedMLModel, ModelNotFound]:
        path = models_dir / f"{model_id}.json"
        guarded = safe(
            (OSError, ValueError),
            fmap_error(
                lambda cause: ModelNotFound(model_id, cause),
                MANIFEST_READ_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_read_manifest)(path).and_then(
            lambda manifest: _defined_from_manifest(manifest).fmap_err(
                fmap_error(
                    lambda cause: ModelNotFound(model_id, cause),
                    MANIFEST_READ_FAILED_CODE,
                    where=f"{path}: invalid manifest",
                )
            )
        )

    return load


type LoadTrainedModelFn = Callable[[ModelId], Result[TrainedMLModel, ModelNotFound]]
"""Concrete shape of the trained-load port -- structurally matches the application's
``LoadTrainedModelFn`` without importing it."""


def _training_run_from_manifest(run: dict[str, Any]) -> TrainingRun:
    """Reconstruct a ``TrainingRun`` from a manifest's run section (may raise KeyError).

    The binding (trainer / data) is no longer in the run section -- it rides on the
    recipe via ``_defined_from_manifest`` -- so only the outcome is rebuilt here.
    """
    metrics = run["metrics"]
    return TrainingRun(
        checkpoint=CheckpointRef(run["checkpoint"]),
        metrics=TrainingMetrics(
            epochs_run=metrics["epochs_run"],
            train_loss=metrics["train_loss"],
            val_loss=metrics["val_loss"],
            best_val_loss=metrics["best_val_loss"],
        ),
        trained_at=datetime.fromisoformat(run["trained_at"]),
    )


def _trained_from_manifest(
    manifest: dict[str, Any],
) -> Result[TrainedMLModel, DefineModelError]:
    """Re-certify a trained manifest into a ``TrainedMLModel`` (recipe + training run).

    The recipe re-certifies through ``define_ml_model`` (the trust boundary); the run
    section is reconstructed. A non-trained manifest (no run section) raises KeyError,
    which the load port maps to ``ModelNotFound``.
    """
    return _defined_from_manifest(manifest).fmap(
        lambda defined: TrainedMLModel(
            defined=defined, run=_training_run_from_manifest(manifest["run"])
        )
    )


def _read_trained(path: Path) -> Result[TrainedMLModel, DefineModelError]:
    """Read and reconstruct a trained model from its manifest (raises on bad I/O/shape)."""
    return _trained_from_manifest(_read_manifest(path))


def make_load_trained_model(models_dir: Path) -> LoadTrainedModelFn:
    """Build the load port reading a "trained" manifest into a ``TrainedMLModel``.

    Mirror of :func:`make_load_model` for the trained state: re-certifies the recipe
    and reconstructs the training run. An unreadable, missing, or non-trained manifest
    (no run section) surfaces as ``ModelNotFound`` -- so evaluating a model that was
    only defined fails cleanly rather than crashing.
    """

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        path = models_dir / f"{model_id}.json"
        guarded = safe(
            (OSError, ValueError, KeyError),
            fmap_error(
                lambda cause: ModelNotFound(model_id, cause),
                MANIFEST_READ_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_read_trained)(path).and_then(
            lambda trained: trained.fmap_err(
                fmap_error(
                    lambda cause: ModelNotFound(model_id, cause),
                    MANIFEST_READ_FAILED_CODE,
                    where=f"{path}: invalid manifest",
                )
            )
        )

    return load


type LoadAnyModelFn = Callable[[ModelId], Result[LiveMLModel, ModelNotFound]]
"""Concrete shape of the load-any port -- hydrates whichever LIVE state is stored,
re-certified; structurally matches the application's ``LoadAnyModelFn``."""


def _as_live(model: LiveMLModel) -> LiveMLModel:
    """Identity widener lifting a concrete live state into the ``LiveMLModel`` union.

    Mirror of the application's ``_widen*`` error wideners: a ``Result``'s value
    parameter is invariant, so ``Result[DefinedMLModel, E]`` does not merge with
    ``Result[LiveMLModel, E]`` on its own -- each branch is widened through here.
    """
    return model


def _evaluation_run_from_manifest(evaluation: dict[str, Any]) -> EvaluationRun:
    """Reconstruct an ``EvaluationRun`` from a manifest's evaluation section.

    Mirror of :func:`_training_run_from_manifest` for the evaluation half (may raise
    KeyError on a malformed section).
    """
    data, metrics = evaluation["data"], evaluation["metrics"]
    return EvaluationRun(
        data=DataSpec(
            dataset_id=data["dataset_id"],
            feature_columns=tuple(data["feature_columns"]),
            target_columns=tuple(data["target_columns"]),
            val_fraction=data["val_fraction"],
        ),
        metrics=EvaluationMetrics(
            test_loss=metrics["test_loss"], n_samples=metrics["n_samples"]
        ),
        evaluated_at=datetime.fromisoformat(evaluation["evaluated_at"]),
    )


def _evaluated_from_manifest(
    manifest: dict[str, Any],
) -> Result[EvaluatedMLModel, DefineModelError]:
    """Re-certify an evaluated manifest into an ``EvaluatedMLModel``.

    Builds on :func:`_trained_from_manifest` (recipe + training run re-certified) and
    folds in the evaluation run from the manifest's "evaluation" section.
    """
    return _trained_from_manifest(manifest).fmap(
        lambda trained: EvaluatedMLModel(
            trained=trained, run=_evaluation_run_from_manifest(manifest["evaluation"])
        )
    )


def _live_from_manifest(
    manifest: dict[str, Any],
) -> Result[LiveMLModel, DefineModelError]:
    """Dispatch on the manifest's status to reconstruct whichever live state it holds.

    An archived (terminal) or unknown status raises ``KeyError`` -- caught by the load
    port and surfaced as ``ModelNotFound`` -- so a tombstone never loads back as live.
    """
    match manifest["status"]:
        case "defined":
            return _defined_from_manifest(manifest).fmap(_as_live)
        case "trained":
            return _trained_from_manifest(manifest).fmap(_as_live)
        case "evaluated":
            return _evaluated_from_manifest(manifest).fmap(_as_live)
        case other:
            raise KeyError(other)


def _read_live(path: Path) -> Result[LiveMLModel, DefineModelError]:
    """Read and reconstruct whichever live state a manifest holds (raises on bad I/O/shape)."""
    return _live_from_manifest(_read_manifest(path))


def make_load_any_model(models_dir: Path) -> LoadAnyModelFn:
    """Build a load port reading ANY live state from ``<dir>/<id>.json``.

    Mirror of :func:`make_load_model` / :func:`make_load_trained_model` widened across
    the whole live lifecycle: it dispatches on the manifest's status and re-certifies
    the recipe (the trust boundary). A missing, unreadable, or archived (terminal)
    manifest surfaces as ``ModelNotFound`` -- so archiving a model that does not exist,
    or one already tombstoned, fails cleanly rather than crashing.
    """

    def load(model_id: ModelId) -> Result[LiveMLModel, ModelNotFound]:
        path = models_dir / f"{model_id}.json"
        guarded = safe(
            (OSError, ValueError, KeyError),
            fmap_error(
                lambda cause: ModelNotFound(model_id, cause),
                MANIFEST_READ_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_read_live)(path).and_then(
            lambda live: live.fmap_err(
                fmap_error(
                    lambda cause: ModelNotFound(model_id, cause),
                    MANIFEST_READ_FAILED_CODE,
                    where=f"{path}: invalid manifest",
                )
            )
        )

    return load


type FindModelManifestFn = Callable[[ModelId], Result[dict[str, Any], ModelNotFound]]
"""Concrete shape of the find-manifest read port -- returns the raw manifest dict (no
certification), structurally matching the application's ``FindModelManifestFn``."""


def make_find_model_manifest(models_dir: Path) -> FindModelManifestFn:
    """Build the read port returning a stored model's raw manifest dict.

    The read-model bypass: unlike :func:`make_load_any_model`, this does NOT re-certify
    the recipe into an aggregate -- it returns the stored manifest as a primitive dict
    for the describe-model projection to display. A missing, unreadable, or malformed-JSON
    manifest surfaces as ``ModelNotFound``.
    """

    def find(model_id: ModelId) -> Result[dict[str, Any], ModelNotFound]:
        path = models_dir / f"{model_id}.json"
        guarded = safe(
            (OSError, ValueError),
            fmap_error(
                lambda cause: ModelNotFound(model_id, cause),
                MANIFEST_READ_FAILED_CODE,
                where=path,
            ),
        )
        return guarded(_read_manifest)(path)

    return find


type FindAllModelManifestsFn = Callable[
    [], Result[list[dict[str, Any]], ModelsNotListed]
]
"""Concrete shape of the list-all read port -- returns EVERY stored manifest dict (no
certification), structurally matching the application's ``FindAllModelManifestsFn``."""

MANIFEST_LIST_FAILED_CODE: Final = "mlmodel.storage.list_failed"
"""``ErrorInfo`` code for a failed model-store listing."""


def _read_all_manifests(models_dir: Path) -> list[dict[str, Any]]:
    """Read every ``<dir>/*.json`` manifest, sorted by id; a missing dir yields ``[]``."""
    if not models_dir.is_dir():
        return []
    return [_read_manifest(path) for path in sorted(models_dir.glob("*.json"))]


def make_list_model_manifests(models_dir: Path) -> FindAllModelManifestsFn:
    """Build the read port returning every stored model's raw manifest dict.

    The read-model bypass over the WHOLE store: globs ``*.json``, reads each manifest
    (no re-certification), and returns them sorted by id. A missing directory yields an
    empty list; an unreadable or malformed manifest surfaces as ``ModelsNotListed``.
    """

    def find_all() -> Result[list[dict[str, Any]], ModelsNotListed]:
        guarded = safe(
            (OSError, ValueError),
            fmap_error(ModelsNotListed, MANIFEST_LIST_FAILED_CODE, where=models_dir),
        )
        return guarded(_read_all_manifests)(models_dir)

    return find_all


def system_now() -> datetime:
    """The system clock as the injected ``NowFn`` -- the archive transition's time source.

    The domain forbids reading the clock; this shell function supplies the impure
    timestamp the composition root injects into ``handle_archive_model``.
    """
    return datetime.now()  # noqa: DTZ005 -- shell clock; the domain only records it
