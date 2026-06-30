"""Tests for the mlmodel infrastructure -- the Lightning training adapter.

``make_train_model_fn`` builds a network from a ``NetworkSpec``, fits it on data
fetched through an injected load-frame port, writes a checkpoint, and returns a
``TrainingOutcome``. The fixture is a noiseless line ``y = 2x + 1``; a few CPU
epochs of the line model write a checkpoint to disk.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from gravityml.mlmodel.domain import (
    Accelerator,
    ActivationName,
    CheckpointRef,
    DataSpec,
    DefinedMLModel,
    EvaluationMetrics,
    EvaluationOutcome,
    LossName,
    ModelId,
    NetworkName,
    OptimizerName,
    ScalerName,
    TrainingMetrics,
    TrainingOutcome,
    archive,
    define_ml_model,
    evaluate,
    train,
)
from gravityml.datasets.application import DatasetNotFound
from gravityml.mlmodel.infrastructure import (
    _best_val_loss,
    _loaders,
    make_evaluate_model_fn,
    make_find_model_manifest,
    make_load_any_model,
    make_load_frame,
    make_load_model,
    make_list_model_manifests,
    make_load_trained_model,
    make_predict_model_fn,
    make_save_model,
    make_train_model_fn,
)
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result


def _line_frame() -> pd.DataFrame:
    """A noiseless line: y = 2x + 1 over 64 points in [-1, 1]."""
    x = np.linspace(-1.0, 1.0, 64)
    return pd.DataFrame({"x": x, "y": 2.0 * x + 1.0})


def _defined_line_model() -> DefinedMLModel:
    """A training-ready line model: recipe + the embedded data/run binding."""
    return define_ml_model(
        model_id="line",
        network_name=NetworkName.SEQUENTIAL_MLP,
        input_dim=1,
        hidden_dims=(),
        target_dim=1,
        activation=ActivationName.IDENTITY,
        optimizer_name=OptimizerName.ADAM,
        learning_rate=0.1,
        weight_decay=0.0,
        loss_name=LossName.MSE,
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.25,
        max_epochs=50,
        batch_size=16,
        seed=0,
        accelerator=Accelerator.CPU,
    ).unwrap()


def _data_spec() -> DataSpec:
    """The evaluate test-set binding (distinct from the define-time training binding)."""
    return DataSpec(
        dataset_id="line",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.25,
    )


def test_train_adapter_writes_a_checkpoint(tmp_path):
    def load_frame(dataset_id: str) -> Result[pd.DataFrame, ErrorInfo]:
        return Ok(_line_frame())

    train_model = make_train_model_fn(load_frame, tmp_path)
    train_model(_defined_line_model()).unwrap()
    assert (tmp_path / "line.safetensors").exists()


def test_train_adapter_learns_the_line(tmp_path):
    def load_frame(dataset_id: str) -> Result[pd.DataFrame, ErrorInfo]:
        return Ok(_line_frame())

    train_model = make_train_model_fn(load_frame, tmp_path)
    outcome = train_model(_defined_line_model()).unwrap()
    assert outcome.metrics.train_loss < 0.01


def _two_target_frame() -> pd.DataFrame:
    """A noiseless 2-target map: y1 = 2x + 1, y2 = -x, over 64 points in [-1, 1]."""
    x = np.linspace(-1.0, 1.0, 64)
    return pd.DataFrame({"x": x, "y1": 2.0 * x + 1.0, "y2": -x})


def _two_target_model() -> DefinedMLModel:
    """A training-ready line model with target_dim=2 over two named target columns."""
    return define_ml_model(
        model_id="line2",
        network_name=NetworkName.SEQUENTIAL_MLP,
        input_dim=1,
        hidden_dims=(),
        target_dim=2,
        activation=ActivationName.IDENTITY,
        optimizer_name=OptimizerName.ADAM,
        learning_rate=0.1,
        weight_decay=0.0,
        loss_name=LossName.MSE,
        dataset_id="line2",
        feature_columns=("x",),
        target_columns=("y1", "y2"),
        val_fraction=0.0,
        max_epochs=50,
        batch_size=16,
        seed=0,
        accelerator=Accelerator.CPU,
    ).unwrap()


def test_train_adapter_fits_a_multi_target_model(tmp_path):
    """Multi-target: a target_dim=2 net fits a (N, 2) target tensor without shape error."""
    train_model = make_train_model_fn(lambda _: Ok(_two_target_frame()), tmp_path)
    outcome = train_model(_two_target_model()).unwrap()
    assert outcome.metrics.train_loss < 0.01


# --- feature scaling: a STANDARD scaler fits a feature on a huge scale ---


def _large_scale_frame() -> pd.DataFrame:
    """A noiseless line over a feature ~1e6 in magnitude: y = 2e-6 * x + 1.

    Raw, the feature dwarfs the O(1) target and an unscaled net cannot fit it; a
    StandardScaler maps it to ~unit scale, so a linear net learns the line.
    """
    x = np.linspace(1.0e6, 2.0e6, 64)
    return pd.DataFrame({"x": x, "y": 2.0e-6 * x + 1.0})


def _scaled_line_model(scaler_name: ScalerName) -> DefinedMLModel:
    """A linear line model over the large-scale fixture, with the given feature scaler."""
    return define_ml_model(
        model_id="scaled",
        network_name=NetworkName.SEQUENTIAL_MLP,
        input_dim=1,
        hidden_dims=(),
        target_dim=1,
        activation=ActivationName.IDENTITY,
        optimizer_name=OptimizerName.ADAM,
        learning_rate=0.1,
        weight_decay=0.0,
        loss_name=LossName.MSE,
        dataset_id="scaled",
        feature_columns=("x",),
        target_columns=("y",),
        val_fraction=0.0,
        max_epochs=200,
        batch_size=16,
        seed=0,
        accelerator=Accelerator.CPU,
        scaler_name=scaler_name,
    ).unwrap()


def test_standard_scaler_fits_a_large_scale_feature(tmp_path):
    """A STANDARD scaler standardizes a large-scale input, so the line model converges."""
    train_model = make_train_model_fn(lambda _: Ok(_large_scale_frame()), tmp_path)
    outcome = train_model(_scaled_line_model(ScalerName.STANDARD)).unwrap()
    assert outcome.metrics.train_loss < 0.01


def test_identity_scaler_cannot_fit_a_large_scale_feature(tmp_path):
    """The contrast: feeding the 1e6 feature unscaled leaves the loss far from fit."""
    train_model = make_train_model_fn(lambda _: Ok(_large_scale_frame()), tmp_path)
    outcome = train_model(_scaled_line_model(ScalerName.IDENTITY)).unwrap()
    assert not outcome.metrics.train_loss < 0.01  # diverged / nan, never a fit


def _trained_scaled_model(tmp_path) -> object:
    """Train the STANDARD model (writing scaler-bearing weights) and fold it trained."""
    train_model = make_train_model_fn(lambda _: Ok(_large_scale_frame()), tmp_path)
    train_model(_scaled_line_model(ScalerName.STANDARD)).unwrap()
    return train(
        _scaled_line_model(ScalerName.STANDARD),
        TrainingOutcome(
            checkpoint=CheckpointRef(str(tmp_path / "scaled.safetensors")),
            metrics=TrainingMetrics(
                epochs_run=200, train_loss=0.001, val_loss=0.0, best_val_loss=0.0
            ),
            trained_at=datetime(2026, 6, 29, 12, 0, 0),
        ),
    )


def test_predict_reapplies_the_saved_standard_scaler(tmp_path):
    """Predict rebuilds the net and loads the SAVED scaler stats, so a raw input is scaled."""
    trained = _trained_scaled_model(tmp_path)
    out = make_predict_model_fn(tmp_path)(
        trained, pd.DataFrame({"x": [2.0e6]})
    ).unwrap()
    assert abs(float(out["y"].iloc[0]) - (2.0e-6 * 2.0e6 + 1.0)) < 0.05


def test_standard_scaler_round_trips_through_the_manifest(tmp_path):
    """The scaler choice persists on the recipe and re-certifies on load."""
    make_save_model(tmp_path)(_scaled_line_model(ScalerName.STANDARD)).unwrap()
    loaded = make_load_model(tmp_path)(ModelId("scaled")).unwrap()
    assert loaded.network.scaler is ScalerName.STANDARD


def test_manifest_without_a_scaler_loads_as_identity(tmp_path):
    """Backward compatibility: a pre-scaler manifest re-certifies as the IDENTITY default."""
    make_save_model(tmp_path)(_defined_line_model()).unwrap()
    path = tmp_path / "line.json"
    manifest = json.loads(path.read_text())
    del manifest["network"]["scaler"]
    path.write_text(json.dumps(manifest))
    loaded = make_load_model(tmp_path)(ModelId("line")).unwrap()
    assert loaded.network.scaler is ScalerName.IDENTITY


# --- best_val_loss: the MINIMUM across epochs, not the final epoch's loss ---


def test_best_val_loss_is_the_minimum_across_epochs():
    assert _best_val_loss([0.5, 0.1, 0.3]) == 0.1  # the min, never the final 0.3


def test_best_val_loss_is_nan_without_validation():
    assert math.isnan(_best_val_loss([]))


# --- validation split: seeded membership (not the frame head) ---


def _val_rows(val_loader) -> set[int]:
    """The single-feature values that landed in the validation loader, as row ids."""
    return {int(v) for v in val_loader.dataset.tensors[0].flatten()}


def test_loaders_seeds_which_rows_validate():
    inputs = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    targets = inputs.clone()
    _, val_a = _loaders(inputs, targets, val_fraction=0.25, batch_size=8, seed=0)
    _, val_b = _loaders(inputs, targets, val_fraction=0.25, batch_size=8, seed=1)
    assert _val_rows(val_a) != {0, 1, 2, 3, 4}  # a seeded shuffle, not the frame head
    assert _val_rows(val_a) != _val_rows(val_b)  # seed changes WHICH rows validate


def test_loaders_same_seed_reproduces_the_val_split():
    inputs = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    targets = inputs.clone()
    _, val_a = _loaders(inputs, targets, val_fraction=0.25, batch_size=8, seed=3)
    _, val_b = _loaders(inputs, targets, val_fraction=0.25, batch_size=8, seed=3)
    assert _val_rows(val_a) == _val_rows(val_b)  # deterministic given the seed


def test_train_adapter_reports_best_not_above_the_final_val_loss(tmp_path):
    train_model = make_train_model_fn(lambda _: Ok(_line_frame()), tmp_path)
    outcome = train_model(_defined_line_model()).unwrap()
    metrics = outcome.metrics
    assert metrics.best_val_loss <= metrics.val_loss  # min(epochs) <= last epoch


def test_save_then_load_round_trips_a_defined_model(tmp_path):
    defined = _defined_line_model()
    make_save_model(tmp_path)(defined).unwrap()
    loaded = make_load_model(tmp_path)(ModelId("line")).unwrap()
    assert loaded == defined  # equality includes the embedded trainer + data binding


def test_defined_manifest_records_the_training_binding(tmp_path):
    """The single source: a defined model's manifest carries trainer + data sections."""
    make_save_model(tmp_path)(_defined_line_model()).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["data"]["dataset_id"] == "line"
    assert manifest["trainer"]["max_epochs"] == 50


def test_save_persists_a_trained_model_manifest(tmp_path):
    outcome = TrainingOutcome(
        checkpoint=CheckpointRef(str(tmp_path / "line.safetensors")),
        metrics=TrainingMetrics(
            epochs_run=50, train_loss=0.001, val_loss=0.002, best_val_loss=0.002
        ),
        trained_at=datetime(2026, 6, 27, 12, 0, 0),
    )
    trained = train(_defined_line_model(), outcome)
    make_save_model(tmp_path)(trained).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["status"] == "trained"


def test_trained_manifest_run_omits_the_duplicated_binding(tmp_path):
    """Single source: the run section drops trainer/data; they stay on the recipe."""
    make_save_model(tmp_path)(_trained_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert "trainer" not in manifest["run"]
    assert "data" not in manifest["run"]
    assert manifest["trainer"]["max_epochs"] == 50  # binding lives at the recipe level


def _trained_line_model(tmp_path):
    """A TrainedMLModel folded from the line fixture and a hand-written outcome."""
    return train(
        _defined_line_model(),
        TrainingOutcome(
            checkpoint=CheckpointRef(str(tmp_path / "line.safetensors")),
            metrics=TrainingMetrics(
                epochs_run=50, train_loss=0.001, val_loss=0.002, best_val_loss=0.002
            ),
            trained_at=datetime(2026, 6, 27, 12, 0, 0),
        ),
    )


def _evaluated_line_model(tmp_path):
    """An EvaluatedMLModel folded from the line fixture and hand-written outcomes."""
    return evaluate(
        _trained_line_model(tmp_path),
        _data_spec(),
        EvaluationOutcome(
            metrics=EvaluationMetrics(test_loss=0.05, n_samples=12),
            evaluated_at=datetime(2026, 6, 27, 13, 0, 0),
        ),
    )


def test_load_trained_round_trips_a_trained_model(tmp_path):
    trained = _trained_line_model(tmp_path)
    make_save_model(tmp_path)(trained).unwrap()
    assert make_load_trained_model(tmp_path)(ModelId("line")).unwrap() == trained


def test_load_trained_missing_model_is_not_found(tmp_path):
    assert make_load_trained_model(tmp_path)(ModelId("ghost")).is_err()


def test_load_trained_rejects_a_defined_only_manifest(tmp_path):
    make_save_model(tmp_path)(_defined_line_model()).unwrap()
    assert make_load_trained_model(tmp_path)(ModelId("line")).is_err()


def test_evaluate_adapter_measures_a_finite_test_loss(tmp_path):
    train_model = make_train_model_fn(lambda _: Ok(_line_frame()), tmp_path)
    train_model(_defined_line_model()).unwrap()
    trained = _trained_line_model(tmp_path)
    evaluate_model = make_evaluate_model_fn(lambda _: Ok(_line_frame()), tmp_path)
    outcome = evaluate_model(trained, _data_spec()).unwrap()
    assert outcome.metrics.n_samples == len(_line_frame())


def test_evaluate_adapter_missing_dataset_is_failed(tmp_path):
    train_model = make_train_model_fn(lambda _: Ok(_line_frame()), tmp_path)
    train_model(_defined_line_model()).unwrap()
    trained = _trained_line_model(tmp_path)
    absent = make_evaluate_model_fn(
        lambda _: Err(ErrorInfo("dataset.storage.read_failed", "absent")), tmp_path
    )
    assert absent(trained, _data_spec()).is_err()


# --- predict adapter: load trained weights, forward pass over a feature frame ---


def test_predict_adapter_predicts_the_line(tmp_path):
    train_model = make_train_model_fn(lambda _: Ok(_line_frame()), tmp_path)
    train_model(_defined_line_model()).unwrap()
    trained = _trained_line_model(tmp_path)
    out = make_predict_model_fn(tmp_path)(trained, pd.DataFrame({"x": [0.0]})).unwrap()
    assert list(out.columns) == ["y"] and abs(float(out["y"].iloc[0]) - 1.0) < 0.2


def test_predict_adapter_missing_weights_is_failed(tmp_path):
    trained = _trained_line_model(tmp_path)  # no .safetensors on disk
    assert make_predict_model_fn(tmp_path)(trained, pd.DataFrame({"x": [0.0]})).is_err()


def test_save_persists_an_evaluated_model_manifest(tmp_path):
    make_save_model(tmp_path)(_evaluated_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["status"] == "evaluated"


def test_evaluated_manifest_records_the_test_loss(tmp_path):
    make_save_model(tmp_path)(_evaluated_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["evaluation"]["metrics"]["test_loss"] == 0.05


_ARCHIVED_AT = datetime(2026, 6, 28, 9, 0, 0)


def _archived_line_model(tmp_path):
    """An ArchivedMLModel tombstoning the evaluated line model at a fixed time."""
    return archive(_evaluated_line_model(tmp_path), _ARCHIVED_AT)


def test_save_writes_an_archived_manifest(tmp_path):
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["status"] == "archived"


def test_archived_manifest_keeps_top_level_model_id(tmp_path):
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["model_id"] == "line"


def test_archived_manifest_nests_the_previous_state(tmp_path):
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["previous"]["status"] == "evaluated"


def test_archived_manifest_records_archived_at(tmp_path):
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    manifest = json.loads((tmp_path / "line.json").read_text())
    assert manifest["archived_at"] == _ARCHIVED_AT.isoformat()


# --- find-manifest read port: the describe-model read-model bypass ---


def test_find_manifest_round_trips_a_saved_model(tmp_path):
    make_save_model(tmp_path)(_defined_line_model()).unwrap()
    manifest = make_find_model_manifest(tmp_path)(ModelId("line")).unwrap()
    assert manifest["status"] == "defined"


def test_find_manifest_reads_an_archived_tombstone(tmp_path):
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    manifest = make_find_model_manifest(tmp_path)(ModelId("line")).unwrap()
    assert manifest["status"] == "archived"


def test_find_manifest_missing_model_is_not_found(tmp_path):
    assert make_find_model_manifest(tmp_path)(ModelId("ghost")).is_err()


def test_list_model_manifests_returns_every_saved_model(tmp_path):
    make_save_model(tmp_path)(_defined_line_model()).unwrap()  # id "line"
    make_save_model(tmp_path)(_two_target_model()).unwrap()  # id "line2"
    manifests = make_list_model_manifests(tmp_path)().unwrap()
    assert {m["model_id"] for m in manifests} == {"line", "line2"}


def test_list_model_manifests_empty_dir_is_empty(tmp_path):
    assert make_list_model_manifests(tmp_path / "empty")().unwrap() == []


def test_load_any_round_trips_each_live_state(tmp_path):
    """load-any reconstructs whichever live state was persisted -- defined/trained/eval."""
    load_any = make_load_any_model(tmp_path)
    for live in (
        _defined_line_model(),
        _trained_line_model(tmp_path),
        _evaluated_line_model(tmp_path),
    ):
        make_save_model(tmp_path)(live).unwrap()
        assert load_any(ModelId("line")).unwrap() == live


def test_load_any_rejects_an_archived_manifest(tmp_path):
    """A tombstone is terminal -- loading it as a live model is ModelNotFound."""
    make_save_model(tmp_path)(_archived_line_model(tmp_path)).unwrap()
    assert make_load_any_model(tmp_path)(ModelId("line")).is_err()


def test_load_any_missing_model_is_not_found(tmp_path):
    assert make_load_any_model(tmp_path)(ModelId("ghost")).is_err()


# --- datasets ACL: adapt the foreign find-frame port to the local LoadFrameFn ---


def test_load_frame_returns_a_found_frame():
    frame = _line_frame()
    load_frame = make_load_frame(lambda _: Ok(frame))
    assert load_frame("d1").unwrap() is frame


def test_load_frame_maps_not_found_to_its_cause():
    cause = ErrorInfo("dataset.storage.read_failed", "absent")
    load_frame = make_load_frame(
        lambda dataset_id: Err(DatasetNotFound(dataset_id, cause))
    )
    assert load_frame("d1").unwrap_err() is cause
