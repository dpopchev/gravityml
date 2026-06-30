"""Tests for the mlmodel read side -- the describe-model query + projection.

The projection is exercised against hand-written manifests for each lifecycle
shape (defined / trained / evaluated / archived), asserting one property per test.
The query handler is driven with a fake ``find`` port (one assert per test).
"""

from gravityml.mlmodel.application import (
    GetModelDescription,
    ListStoredModels,
    ModelDescription,
    handle_get_model_description,
    handle_list_models,
    to_model_description,
)
from gravityml.mlmodel.domain import ModelId, ModelNotFound, ModelsNotListed
from gravityml.shared_kernel.error import ErrorInfo
from gravityml.shared_kernel.result import Err, Ok, Result


def _trainer_section() -> dict:
    return {"max_epochs": 50, "batch_size": 64, "seed": 0, "accelerator": "cpu"}


def _data_section() -> dict:
    return {
        "dataset_id": "example",
        "feature_columns": ["x1", "x2", "x3"],
        "target_columns": ["y1", "y2"],
        "val_fraction": 0.2,
    }


def _defined_manifest() -> dict:
    """A training-ready defined manifest: recipe + the embedded trainer/data binding."""
    return {
        "status": "defined",
        "model_id": "demo",
        "network": {
            "name": "sequential-mlp",
            "input_dim": 3,
            "hidden_dims": [16, 8],
            "target_dim": 2,
            "activation": "relu",
        },
        "optimizer": {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
        "loss": {"name": "mse"},
        "trainer": _trainer_section(),
        "data": _data_section(),
    }


def _run_section() -> dict:
    """The run half -- the outcome only; the binding lives on the recipe (single source)."""
    return {
        "checkpoint": "demo.safetensors",
        "metrics": {
            "epochs_run": 50,
            "train_loss": 0.01,
            "val_loss": 0.02,
            "best_val_loss": 0.02,
        },
        "trained_at": "2026-06-27T12:00:00",
    }


def _trained_manifest() -> dict:
    return {**_defined_manifest(), "status": "trained", "run": _run_section()}


def _evaluated_manifest() -> dict:
    return {
        **_trained_manifest(),
        "status": "evaluated",
        "evaluation": {
            "data": _data_section(),
            "metrics": {"test_loss": 0.03, "n_samples": 1000},
            "evaluated_at": "2026-06-27T13:00:00",
        },
    }


def _archived_manifest() -> dict:
    return {
        "status": "archived",
        "model_id": "demo",
        "archived_at": "2026-06-27T14:00:00",
        "previous": _evaluated_manifest(),
    }


# --- projection: recipe + status ---


def test_projection_surfaces_status():
    assert to_model_description(_defined_manifest()).status == "defined"


def test_projection_surfaces_the_network_recipe():
    description = to_model_description(_defined_manifest())
    assert (description.network_name, description.target_dim) == ("sequential-mlp", 2)


def test_projection_surfaces_the_optimizer_and_loss():
    description = to_model_description(_defined_manifest())
    assert (description.learning_rate, description.loss_name) == (0.001, "mse")


def test_projection_omits_run_sections_for_a_defined_model():
    description = to_model_description(_defined_manifest())
    assert (description.training, description.evaluation, description.archived_at) == (
        None,
        None,
        None,
    )


# --- projection: training + evaluation runs ---


def test_projection_surfaces_the_training_run():
    training = to_model_description(_trained_manifest()).training
    assert training is not None and training.target_columns == ("y1", "y2")


def test_projection_surfaces_the_evaluation_run():
    evaluation = to_model_description(_evaluated_manifest()).evaluation
    assert evaluation is not None and evaluation.test_loss == 0.03


# --- projection: archived unwraps the tombstoned recipe ---


def test_projection_archived_reports_archived_status():
    assert to_model_description(_archived_manifest()).status == "archived"


def test_projection_archived_surfaces_the_archived_timestamp():
    assert (
        to_model_description(_archived_manifest()).archived_at == "2026-06-27T14:00:00"
    )


def test_projection_archived_unwraps_the_previous_recipe():
    description = to_model_description(_archived_manifest())
    assert (description.network_name, description.training is not None) == (
        "sequential-mlp",
        True,
    )


# --- query handler: railway over the find port ---


def test_query_maps_found_manifest_to_description():
    outcome = handle_get_model_description(
        lambda _: Ok(_trained_manifest()),
        GetModelDescription(model_id="demo"),
    )
    assert isinstance(outcome.unwrap(), ModelDescription)


def test_query_propagates_not_found():
    not_found = ModelNotFound(ModelId("ghost"), ErrorInfo("x", "absent"))
    outcome = handle_get_model_description(
        lambda _: Err(not_found),
        GetModelDescription(model_id="ghost"),
    )
    assert outcome.unwrap_err() is not_found


# --- list-models query: enumerate every stored model (id + status) ---


def test_list_models_projects_id_and_status():
    def find_all() -> Result[list[dict], ModelsNotListed]:
        return Ok([_defined_manifest(), _archived_manifest()])

    listing = handle_list_models(find_all, ListStoredModels()).unwrap()
    assert [(m.model_id, m.status) for m in listing.models] == [
        ("demo", "defined"),
        ("demo", "archived"),
    ]


def test_list_models_empty_store_is_an_empty_listing():
    listing = handle_list_models(lambda: Ok([]), ListStoredModels()).unwrap()
    assert listing.models == ()


def test_list_models_propagates_a_listing_failure():
    failure = ModelsNotListed(ErrorInfo("x", "boom"))
    outcome = handle_list_models(lambda: Err(failure), ListStoredModels())
    assert outcome.unwrap_err() is failure
