"""Tests for the mlmodel application layer -- the DefineMLModel command.

``handle_define_ml_model`` certifies hyperparameters through the domain and
persists the resulting ``DefinedMLModel`` via an injected ``SaveModelFn``. The
save port is faked with an in-memory recorder -- no real I/O.
"""

from dataclasses import replace
from datetime import datetime

import pandas as pd

from gravityml.mlmodel.application import (
    ArchiveMLModel,
    DefineMLModel,
    EvaluateMLModel,
    PredictMLModel,
    TrainMLModel,
    handle_archive_model,
    handle_define_ml_model,
    handle_evaluate_model,
    handle_predict_model,
    handle_train_model,
)
from gravityml.mlmodel.domain import (
    Accelerator,
    ActivationName,
    CheckpointRef,
    DataSpec,
    DefinedMLModel,
    EvaluationFailed,
    EvaluationMetrics,
    EvaluationOutcome,
    LiveMLModel,
    LossName,
    MLModel,
    ModelId,
    ModelNotFound,
    ModelNotSaved,
    NetworkName,
    NonPositiveLearningRate,
    OptimizerName,
    PredictionFailed,
    ScalerName,
    TrainedMLModel,
    TrainingFailed,
    TrainingMetrics,
    TrainingOutcome,
    define_ml_model,
    train,
)
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result


def _define_command() -> DefineMLModel:
    """A line-regression define command (matches the domain fixture).

    Training-ready: the recipe plus the training binding (data + run knobs) the
    define command now carries, so train-model needs only the model id.
    """
    return DefineMLModel(
        model_id="line",
        network_name=NetworkName.SEQUENTIAL_MLP,
        input_dim=1,
        hidden_dims=(),
        target_dim=1,
        activation=ActivationName.IDENTITY,
        optimizer_name=OptimizerName.ADAM,
        learning_rate=0.01,
        weight_decay=0.0,
        loss_name=LossName.MSE,
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.2,
        max_epochs=50,
        batch_size=16,
        seed=0,
        accelerator=Accelerator.CPU,
    )


def test_handle_define_ml_model_persists_the_defined_model():
    captured: list[MLModel] = []

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        captured.append(model)
        return Ok(model)

    defined = handle_define_ml_model(save, _define_command()).unwrap()
    assert defined is captured[0]


def test_handle_define_ml_model_embeds_the_training_binding():
    """The define command's binding flows into the persisted training-ready model."""

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        return Ok(model)

    defined = handle_define_ml_model(save, _define_command()).unwrap()
    assert defined.trainer.max_epochs == 50
    assert defined.data.dataset_id == "line"


def test_handle_define_ml_model_carries_the_scaler_choice():
    """A STANDARD scaler on the command flows onto the persisted network spec."""

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        return Ok(model)

    cmd = replace(_define_command(), scaler=ScalerName.STANDARD)
    defined = handle_define_ml_model(save, cmd).unwrap()
    assert defined.network.scaler is ScalerName.STANDARD


def test_handle_define_ml_model_short_circuits_on_spec_error():
    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        raise AssertionError("save must not run when certification fails")

    bad = replace(_define_command(), learning_rate=0.0)
    assert handle_define_ml_model(save, bad).unwrap_err() == NonPositiveLearningRate(
        0.0
    )


def test_handle_define_ml_model_surfaces_a_save_failure():
    failure = ModelNotSaved(ModelId("line"), ErrorInfo(code="x", message="disk full"))

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        return Err(failure)

    assert handle_define_ml_model(save, _define_command()).unwrap_err() is failure


def _defined_line_model() -> DefinedMLModel:
    """A certified, training-ready line-regression DefinedMLModel to train."""
    return define_ml_model(
        model_id="line",
        network_name=NetworkName.SEQUENTIAL_MLP,
        input_dim=1,
        hidden_dims=(),
        target_dim=1,
        activation=ActivationName.IDENTITY,
        optimizer_name=OptimizerName.ADAM,
        learning_rate=0.01,
        weight_decay=0.0,
        loss_name=LossName.MSE,
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.2,
        max_epochs=50,
        batch_size=16,
        seed=0,
        accelerator=Accelerator.CPU,
    ).unwrap()


def _train_command() -> TrainMLModel:
    """A train command -- now just the id; the binding lives on the defined model."""
    return TrainMLModel(model_id="line")


def _training_outcome() -> TrainingOutcome:
    """A hand-written outcome standing in for one Lightning fit run."""
    return TrainingOutcome(
        checkpoint=CheckpointRef("state/models/line.safetensors"),
        metrics=TrainingMetrics(
            epochs_run=50, train_loss=0.01, val_loss=0.02, best_val_loss=0.015
        ),
        trained_at=datetime(2026, 6, 27, 12, 0, 0),
    )


def test_handle_train_model_trains_and_saves():
    defined = _defined_line_model()
    outcome = _training_outcome()
    captured: list[MLModel] = []

    def load(model_id: ModelId) -> Result[DefinedMLModel, ModelNotFound]:
        return Ok(defined)

    def train_model(
        model: DefinedMLModel,
    ) -> Result[TrainingOutcome, TrainingFailed]:
        return Ok(outcome)

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        captured.append(model)
        return Ok(model)

    trained = handle_train_model(load, train_model, save, _train_command()).unwrap()
    assert trained is captured[0]


def _trained_line_model() -> TrainedMLModel:
    """A TrainedMLModel folded from the line fixture and a hand-written outcome."""
    return train(_defined_line_model(), _training_outcome())


def _evaluate_command() -> EvaluateMLModel:
    """A line-regression evaluate command -- a test pass on a 1-feature dataset."""
    return EvaluateMLModel(
        model_id="line",
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
    )


def _evaluation_outcome() -> EvaluationOutcome:
    """A hand-written outcome standing in for one Lightning evaluation pass."""
    return EvaluationOutcome(
        metrics=EvaluationMetrics(test_loss=0.03, n_samples=20),
        evaluated_at=datetime(2026, 6, 27, 13, 0, 0),
    )


def test_handle_evaluate_model_evaluates_and_saves():
    trained = _trained_line_model()
    outcome = _evaluation_outcome()
    captured: list[MLModel] = []

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        return Ok(trained)

    def evaluate_model(
        model: TrainedMLModel, data: DataSpec
    ) -> Result[EvaluationOutcome, EvaluationFailed]:
        return Ok(outcome)

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        captured.append(model)
        return Ok(model)

    evaluated = handle_evaluate_model(
        load, evaluate_model, save, _evaluate_command()
    ).unwrap()
    assert evaluated is captured[0]


def test_handle_evaluate_model_short_circuits_on_missing_model():
    missing = ModelNotFound(ModelId("line"), ErrorInfo(code="x", message="absent"))

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        return Err(missing)

    def evaluate_model(
        model: TrainedMLModel, data: DataSpec
    ) -> Result[EvaluationOutcome, EvaluationFailed]:
        raise AssertionError("evaluate must not run when the model is missing")

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        raise AssertionError("save must not run when the model is missing")

    assert (
        handle_evaluate_model(
            load, evaluate_model, save, _evaluate_command()
        ).unwrap_err()
        is missing
    )


def test_handle_evaluate_model_surfaces_an_evaluation_failure():
    failure = EvaluationFailed(ModelId("line"), ErrorInfo(code="x", message="boom"))
    trained = _trained_line_model()

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        return Ok(trained)

    def evaluate_model(
        model: TrainedMLModel, data: DataSpec
    ) -> Result[EvaluationOutcome, EvaluationFailed]:
        return Err(failure)

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        raise AssertionError("save must not run when evaluation fails")

    assert (
        handle_evaluate_model(
            load, evaluate_model, save, _evaluate_command()
        ).unwrap_err()
        is failure
    )


def _archive_command() -> ArchiveMLModel:
    """An archive command naming the line model to tombstone."""
    return ArchiveMLModel(model_id="line")


_ARCHIVED_AT = datetime(2026, 6, 28, 9, 0, 0)


def test_archive_loads_archives_and_persists():
    defined = _defined_line_model()
    captured: list[MLModel] = []

    def load(model_id: ModelId) -> Result[LiveMLModel, ModelNotFound]:
        return Ok(defined)

    def now() -> datetime:
        return _ARCHIVED_AT

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        captured.append(model)
        return Ok(model)

    archived = handle_archive_model(load, now, save, _archive_command()).unwrap()
    assert archived is captured[0]
    assert archived.previous is defined
    assert archived.record.archived_at == _ARCHIVED_AT


def test_archive_missing_model_is_not_found():
    missing = ModelNotFound(ModelId("line"), ErrorInfo(code="x", message="absent"))

    def load(model_id: ModelId) -> Result[LiveMLModel, ModelNotFound]:
        return Err(missing)

    def now() -> datetime:
        raise AssertionError("clock must not be read when the model is missing")

    def save(model: MLModel) -> Result[MLModel, ModelNotSaved]:
        raise AssertionError("save must not run when the model is missing")

    assert (
        handle_archive_model(load, now, save, _archive_command()).unwrap_err()
        is missing
    )


# --- predict: load the trained model, run inference, return the predictions frame ---


def _predict_query() -> PredictMLModel:
    """A predict request -- a model id plus the input feature frame to infer over."""
    return PredictMLModel(model_id="line", frame=pd.DataFrame({"x": [0.0, 1.0]}))


def test_handle_predict_model_runs_inference():
    trained = _trained_line_model()
    predictions = pd.DataFrame({"y": [1.0, 3.0]})

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        return Ok(trained)

    def predict_model(
        model: TrainedMLModel, frame: pd.DataFrame
    ) -> Result[pd.DataFrame, PredictionFailed]:
        return Ok(predictions)

    out = handle_predict_model(load, predict_model, _predict_query()).unwrap()
    assert out is predictions


def test_handle_predict_model_short_circuits_on_missing_model():
    missing = ModelNotFound(ModelId("line"), ErrorInfo(code="x", message="absent"))

    def load(model_id: ModelId) -> Result[TrainedMLModel, ModelNotFound]:
        return Err(missing)

    def predict_model(
        model: TrainedMLModel, frame: pd.DataFrame
    ) -> Result[pd.DataFrame, PredictionFailed]:
        raise AssertionError("predict must not run when the model is missing")

    assert (
        handle_predict_model(load, predict_model, _predict_query()).unwrap_err()
        is missing
    )
