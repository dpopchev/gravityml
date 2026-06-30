"""Tests for the mlmodel domain -- the DefineMLModel transition.

The smart constructor ``define_ml_model`` validates the network and optimizer
hyperparameters and assembles them into a ``DefinedMLModel`` -- the untrained
state of the lifecycle (network + optimizer + loss, no weights yet). The fixture
is a line regression (y = w*x + b): a single input, no hidden layers, a single
target, mean-squared-error loss, Adam optimizer.
"""

from datetime import datetime

from gravityml.mlmodel.domain import (
    Accelerator,
    ActivationName,
    CheckpointRef,
    DataSpec,
    EvaluationFailed,
    EvaluationMetrics,
    EvaluationOutcome,
    LossName,
    ModelId,
    ModelNotFound,
    ModelNotSaved,
    NetworkName,
    NonPositiveDimensions,
    NonPositiveLearningRate,
    NonPositiveTrainingSize,
    OptimizerName,
    ScalerName,
    TrainerSpec,
    TrainingFailed,
    TrainingMetrics,
    TrainingOutcome,
    ValFractionOutOfRange,
    archive,
    define_ml_model,
    evaluate,
    make_data_spec,
    make_trainer_spec,
    train,
)
from gravityml.shared_kernel.error import ErrorInfo


def _line_regression_kwargs() -> dict:
    """Hyperparameters for a line regression -- the minimal trainable model.

    A training-ready definition: the recipe (network + optimizer + loss) plus the
    training binding (trainer run knobs + data) that ``define-model`` now embeds, so
    ``train-model`` needs only the model id.
    """
    return dict(
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


def test_define_ml_model_with_valid_line_regression_params_is_ok():
    defined = define_ml_model(**_line_regression_kwargs()).unwrap()
    assert defined.network.target_dim == 1


def test_define_ml_model_defaults_to_an_identity_scaler():
    """Omitting the scaler choice leaves features unscaled -- the backward-compatible default."""
    defined = define_ml_model(**_line_regression_kwargs()).unwrap()
    assert defined.network.scaler is ScalerName.IDENTITY


def test_define_ml_model_carries_a_standard_scaler_choice():
    """A STANDARD scaler choice rides on the certified network spec."""
    defined = define_ml_model(
        **{**_line_regression_kwargs(), "scaler_name": ScalerName.STANDARD}
    ).unwrap()
    assert defined.network.scaler is ScalerName.STANDARD


def test_define_ml_model_embeds_the_trainer_and_data_binding():
    """A training-ready defined model carries its trainer + data, not just the recipe."""
    defined = define_ml_model(**_line_regression_kwargs()).unwrap()
    assert defined.trainer == _trainer_spec()
    assert defined.data == _data_spec()


def test_define_ml_model_rejects_non_positive_max_epochs():
    outcome = define_ml_model(**{**_line_regression_kwargs(), "max_epochs": 0})
    assert outcome.unwrap_err() == NonPositiveTrainingSize((0,))


def test_define_ml_model_rejects_val_fraction_out_of_range():
    outcome = define_ml_model(**{**_line_regression_kwargs(), "val_fraction": 1.0})
    assert outcome.unwrap_err() == ValFractionOutOfRange(1.0)


def test_define_ml_model_rejects_non_positive_input_dim():
    outcome = define_ml_model(**{**_line_regression_kwargs(), "input_dim": 0})
    assert outcome.unwrap_err() == NonPositiveDimensions((0,))


def test_define_ml_model_rejects_non_positive_learning_rate():
    outcome = define_ml_model(**{**_line_regression_kwargs(), "learning_rate": 0.0})
    assert outcome.unwrap_err() == NonPositiveLearningRate(0.0)


def _defined_line_model():
    """A certified line-regression DefinedMLModel to train."""
    return define_ml_model(**_line_regression_kwargs()).unwrap()


def _trainer_spec() -> TrainerSpec:
    """A short CPU training run for the line fixture."""
    return TrainerSpec(
        max_epochs=50, batch_size=16, seed=0, accelerator=Accelerator.CPU
    )


def _data_spec() -> DataSpec:
    """The single-feature, single-target data binding for the line fixture."""
    return DataSpec(
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.2,
    )


def _training_outcome() -> TrainingOutcome:
    """A hand-written outcome standing in for one Lightning fit run."""
    return TrainingOutcome(
        checkpoint=CheckpointRef("state/models/line.safetensors"),
        metrics=TrainingMetrics(
            epochs_run=50, train_loss=0.01, val_loss=0.02, best_val_loss=0.015
        ),
        trained_at=datetime(2026, 6, 27, 12, 0, 0),
    )


def test_train_folds_outcome_into_a_trained_model():
    defined = _defined_line_model()
    trained = train(defined, _training_outcome())
    assert trained.defined is defined


def test_make_trainer_spec_rejects_non_positive_max_epochs():
    outcome = make_trainer_spec(
        max_epochs=0, batch_size=16, seed=0, accelerator=Accelerator.CPU
    )
    assert outcome.unwrap_err() == NonPositiveTrainingSize((0,))


def test_make_data_spec_rejects_val_fraction_out_of_range():
    outcome = make_data_spec(
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=1.0,
    )
    assert outcome.unwrap_err() == ValFractionOutOfRange(1.0)


def test_make_data_spec_accepts_multiple_targets():
    """The binding generalizes to multi-target -- e.g. a 2-tuple [y1, y2]."""
    data = make_data_spec(
        dataset_id="example",
        feature_columns=("x1", "x2", "x3"),
        target_columns=("y1", "y2"),
        val_fraction=0.2,
    ).unwrap()
    assert data.target_columns == ("y1", "y2")


def _trained_line_model():
    """A TrainedMLModel folded from the line fixture and a hand-written outcome."""
    return train(_defined_line_model(), _training_outcome())


def _evaluation_outcome() -> EvaluationOutcome:
    """A hand-written outcome standing in for one Lightning evaluation pass."""
    return EvaluationOutcome(
        metrics=EvaluationMetrics(test_loss=0.03, n_samples=20),
        evaluated_at=datetime(2026, 6, 27, 13, 0, 0),
    )


def test_evaluate_folds_outcome_into_an_evaluated_model():
    trained = _trained_line_model()
    evaluated = evaluate(trained, _data_spec(), _evaluation_outcome())
    assert evaluated.trained is trained


def test_evaluate_records_the_test_loss():
    evaluated = evaluate(_trained_line_model(), _data_spec(), _evaluation_outcome())
    assert evaluated.run.metrics.test_loss == 0.03


_ARCHIVED_AT = datetime(2026, 6, 28, 9, 0, 0)


def test_archive_folds_a_live_model():
    """A defined model archives into a tombstone embedding that very model."""
    defined = _defined_line_model()
    archived = archive(defined, _ARCHIVED_AT)
    assert archived.previous is defined


def test_archive_records_archived_at():
    """The injected archival time is recorded on the tombstone's record."""
    archived = archive(_defined_line_model(), _ARCHIVED_AT)
    assert archived.record.archived_at == _ARCHIVED_AT


def test_archive_tombstones_an_evaluated_model():
    """Archive is reachable from ANY live state -- here the evaluated state."""
    evaluated = evaluate(_trained_line_model(), _data_spec(), _evaluation_outcome())
    archived = archive(evaluated, _ARCHIVED_AT)
    assert archived.previous is evaluated


def test_non_positive_dimensions_str_names_the_values():
    bad = NonPositiveDimensions((-3, 0))
    assert str(bad) == "network dimensions must be strictly positive: -3, 0"


def test_non_positive_learning_rate_str_names_the_value():
    bad = NonPositiveLearningRate(0.0)
    assert str(bad) == "learning rate must be strictly positive: 0.0"


def test_non_positive_training_size_str_names_the_values():
    bad = NonPositiveTrainingSize((0,))
    assert str(bad) == "training sizes must be strictly positive: 0"


def test_val_fraction_out_of_range_str_names_the_value():
    bad = ValFractionOutOfRange(1.5)
    assert str(bad) == "validation fraction must be in [0, 1): 1.5"


def test_model_not_saved_str_is_the_cause_message():
    not_saved = ModelNotSaved(
        ModelId("m1"), ErrorInfo("store.write_failed", "disk full")
    )
    assert str(not_saved) == "disk full"


def test_model_not_found_str_is_the_cause_message():
    not_found = ModelNotFound(
        ModelId("m1"), ErrorInfo("store.read_failed", "no stored model")
    )
    assert str(not_found) == "no stored model"


def test_training_failed_str_is_the_cause_message():
    failed = TrainingFailed(
        ModelId("m1"), ErrorInfo("mlmodel.train_failed", "cuda oom")
    )
    assert str(failed) == "cuda oom"


def test_evaluation_failed_str_is_the_cause_message():
    failed = EvaluationFailed(
        ModelId("m1"), ErrorInfo("mlmodel.eval_failed", "bad metric")
    )
    assert str(failed) == "bad metric"
