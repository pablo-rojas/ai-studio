from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_training_experiment_crud_via_api(test_client, workspace: Path) -> None:
    project_id = await _create_project(test_client, name="Training API Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="training_api_source",
    )

    list_response = await test_client.get(f"/api/training/{project_id}/experiments")
    assert list_response.status_code == 200
    assert list_response.json() == {"status": "ok", "data": {"experiments": []}}

    create_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Baseline"},
    )
    assert create_response.status_code == 200
    created_payload = create_response.json()["data"]
    experiment_id = created_payload["id"]
    assert experiment_id.startswith("exp-")
    assert created_payload["status"] == "created"
    assert created_payload["name"] == "Baseline"

    get_response = await test_client.get(f"/api/training/{project_id}/experiments/{experiment_id}")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == experiment_id

    patch_response = await test_client.patch(
        f"/api/training/{project_id}/experiments/{experiment_id}",
        json={"name": "Baseline Updated"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["data"]["name"] == "Baseline Updated"

    metrics_response = await test_client.get(
        f"/api/training/{project_id}/experiments/{experiment_id}/metrics"
    )
    assert metrics_response.status_code == 200
    assert metrics_response.json()["data"] == {"epochs": []}

    stream_response = await test_client.get(
        f"/api/training/{project_id}/experiments/{experiment_id}/stream"
    )
    assert stream_response.status_code == 200
    assert "text/event-stream" in stream_response.headers["content-type"]
    assert "event: status" in stream_response.text
    assert "event: complete" in stream_response.text

    delete_response = await test_client.delete(
        f"/api/training/{project_id}/experiments/{experiment_id}"
    )
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "ok",
        "data": {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "deleted": True,
        },
    }


@pytest.mark.asyncio
async def test_training_hx_experiment_detail_and_form_patch_flow(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Training HTMX Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="training_hx_source",
    )

    create_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "HX Baseline"},
    )
    assert create_response.status_code == 200
    created_payload = create_response.json()["data"]
    experiment_id = created_payload["id"]
    original_val_augmentations = created_payload["augmentations"]["val"]

    detail_response = await test_client.get(
        f"/api/training/{project_id}/experiments/{experiment_id}",
        headers={"HX-Request": "true"},
    )
    assert detail_response.status_code == 200
    assert 'id="training-experiment-workspace"' in detail_response.text
    assert "Loss Curves" not in detail_response.text
    assert "F1 Curves" not in detail_response.text
    assert "data-training-loss-train-latest" not in detail_response.text
    assert "data-training-metric-train-latest" not in detail_response.text
    assert "data-training-loss-axis-x" in detail_response.text
    assert "data-training-loss-axis-y" in detail_response.text
    assert "data-training-metric-axis-x" in detail_response.text
    assert "data-training-metric-axis-y" in detail_response.text
    assert "data-training-metric-train" in detail_response.text
    assert "data-training-metric-val" in detail_response.text
    assert 'data-training-config-section="setup"' in detail_response.text
    assert 'data-training-config-section="optimization"' in detail_response.text
    assert 'data-training-config-section="hardware"' in detail_response.text
    assert "training-config-section" in detail_response.text
    assert "data-training-section-header" in detail_response.text
    assert "data-training-section-chevron" in detail_response.text
    assert 'name="hardware.selected_devices[]"' in detail_response.text
    assert "data-training-effective-batch" in detail_response.text
    assert "Name and dataset split selection." not in detail_response.text
    assert "Model backbone and head behavior." not in detail_response.text
    assert "Optimizer and learning-rate schedule." not in detail_response.text
    assert "Loss function and loss-specific parameters." not in detail_response.text
    assert "Epoch, batch, and stopping controls." not in detail_response.text
    assert "Device selection and precision settings." not in detail_response.text
    assert "Enable or disable training transforms." not in detail_response.text
    assert "Augmentation Toggles" not in detail_response.text
    assert 'data-augmentation-row="RandomResizedCrop"' in detail_response.text

    patch_form_payload: list[tuple[str, str]] = [
        ("name", "HX Baseline Updated"),
        ("split_name", "80-10-10"),
        ("model.backbone", "resnet34"),
        ("model.pretrained", "true"),
        ("model.freeze_backbone", "false"),
        ("hyperparameters.optimizer", "adamw"),
        ("hyperparameters.learning_rate", "0.0005"),
        ("hyperparameters.weight_decay", "0.0001"),
        ("hyperparameters.momentum", "0.9"),
        ("hyperparameters.scheduler", "cosine"),
        ("hyperparameters.warmup_epochs", "5"),
        ("hyperparameters.step_size", "10"),
        ("hyperparameters.gamma", "0.1"),
        ("hyperparameters.poly_power", "0.9"),
        ("hyperparameters.batch_size", "16"),
        ("hyperparameters.batch_multiplier", "2"),
        ("hyperparameters.max_epochs", "12"),
        ("hyperparameters.early_stopping_patience", "3"),
        ("hyperparameters.loss", "cross_entropy"),
        ("hyperparameters.label_smoothing", "0.1"),
        ("hyperparameters.dropout", "0.25"),
        ("hardware.precision", "32"),
        ("hardware.selected_devices[]", "cpu"),
    ]
    train_augmentations = [
        {
            "name": "RandomResizedCrop",
            "params": {"size": [224, 224], "scale": [0.75, 1.0], "apply_p": 0.8},
        },
        {"name": "RandomHorizontalFlip", "params": {"p": 0.3}},
        {"name": "RandomRotation", "params": {"degrees": [-20, 20], "apply_p": 0.6}},
        {
            "name": "ColorJitter",
            "params": {
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.2,
                "hue": 0.04,
                "apply_p": 0.7,
            },
        },
        {"name": "ToImage", "params": {}},
        {
            "name": "Normalize",
            "params": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    ]
    for step in train_augmentations:
        patch_form_payload.append(("augmentations.train[]", json.dumps(step)))
    for step in original_val_augmentations:
        patch_form_payload.append(("augmentations.val[]", json.dumps(step)))

    encoded_payload = urlencode(patch_form_payload)
    patch_response = await test_client.patch(
        f"/api/training/{project_id}/experiments/{experiment_id}",
        content=encoded_payload,
        headers={
            "HX-Request": "true",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    assert patch_response.status_code == 200
    assert 'id="training-experiment-workspace"' in patch_response.text
    assert "HX Baseline Updated" in patch_response.text

    get_updated_response = await test_client.get(
        f"/api/training/{project_id}/experiments/{experiment_id}"
    )
    assert get_updated_response.status_code == 200
    updated_payload = get_updated_response.json()["data"]
    assert updated_payload["name"] == "HX Baseline Updated"
    assert updated_payload["model"]["backbone"] == "resnet34"
    assert updated_payload["hyperparameters"]["optimizer"] == "adamw"
    assert updated_payload["hyperparameters"]["batch_size"] == 16
    assert updated_payload["hyperparameters"]["dropout"] == pytest.approx(0.25)
    assert updated_payload["augmentations"]["train"] == train_augmentations
    assert updated_payload["augmentations"]["val"] == original_val_augmentations


@pytest.mark.asyncio
async def test_training_control_endpoints_delegate_to_service(
    test_client,
    workspace: Path,
    monkeypatch,
) -> None:
    project_id = await _create_project(test_client, name="Training Control API Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="training_control_api_source",
    )

    create_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Control Baseline"},
    )
    assert create_response.status_code == 200
    experiment = create_response.json()["data"]
    experiment_id = experiment["id"]

    app = test_client._transport.app
    service = app.state.training_service

    started = service.get_experiment(project_id, experiment_id).model_copy(
        update={"status": "pending"}
    )
    stopped = started.model_copy(update={"status": "cancelled"})
    restarted = stopped.model_copy(update={"status": "created"})

    monkeypatch.setattr(service, "start_training", lambda *args, **kwargs: started)
    monkeypatch.setattr(service, "stop_training", lambda *args, **kwargs: stopped)
    monkeypatch.setattr(service, "resume_training", lambda *args, **kwargs: started)
    monkeypatch.setattr(service, "restart_experiment", lambda *args, **kwargs: restarted)

    start_response = await test_client.post(
        f"/api/training/{project_id}/experiments/{experiment_id}/train"
    )
    assert start_response.status_code == 200
    assert start_response.json()["data"] == {"experiment_id": experiment_id, "status": "pending"}

    stop_response = await test_client.post(
        f"/api/training/{project_id}/experiments/{experiment_id}/stop"
    )
    assert stop_response.status_code == 200
    assert stop_response.json()["data"] == {"experiment_id": experiment_id, "status": "cancelled"}

    resume_response = await test_client.post(
        f"/api/training/{project_id}/experiments/{experiment_id}/resume"
    )
    assert resume_response.status_code == 200
    assert resume_response.json()["data"] == {"experiment_id": experiment_id, "status": "pending"}

    restart_response = await test_client.post(
        f"/api/training/{project_id}/experiments/{experiment_id}/restart"
    )
    assert restart_response.status_code == 200
    assert restart_response.json()["data"]["status"] == "created"


@pytest.mark.asyncio
async def test_training_missing_experiment_returns_standard_error(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Training Missing Experiment Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="training_missing_experiment_source",
    )

    response = await test_client.get(f"/api/training/{project_id}/experiments/exp-deadbeef")
    assert response.status_code == 404
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "EXPERIMENT_NOT_FOUND"


async def _create_project(test_client, *, name: str) -> str:
    response = await test_client.post(
        "/api/projects",
        json={"name": name, "task": "classification"},
    )
    assert response.status_code == 200
    return response.json()["data"]["id"]


async def _import_dataset_and_split(
    test_client,
    workspace: Path,
    project_id: str,
    *,
    source_name: str,
) -> None:
    source_root = workspace.parent / source_name
    cats_dir = source_root / "cats"
    dogs_dir = source_root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for index in range(20):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(220, 120, 120))
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 220))

    import_response = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={"source_path": str(source_root), "source_format": "image_folders"},
    )
    assert import_response.status_code == 200

    split_response = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert split_response.status_code == 200


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)
