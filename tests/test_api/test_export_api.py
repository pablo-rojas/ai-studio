from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import app.export.onnx_export as onnx_export_module


class _ColorHeuristicClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        red_mean = images[:, 0, :, :].mean(dim=(1, 2))
        return torch.stack([red_mean, 1.0 - red_mean], dim=1)


@pytest.mark.asyncio
async def test_export_api_end_to_end_flow(
    test_client,
    workspace: Path,
    monkeypatch,
) -> None:
    project_id = await _create_project(test_client, name="Export API Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="export_api_source",
    )
    experiment_id = await _create_experiment(test_client, project_id, name="Export Baseline")
    _mark_experiment_completed_and_seed_checkpoints(test_client, project_id, experiment_id)

    monkeypatch.setattr(
        onnx_export_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    formats_response = await test_client.get("/api/export/formats")
    assert formats_response.status_code == 200
    formats_payload = formats_response.json()["data"]["formats"]
    onnx_format = next(item for item in formats_payload if item["name"] == "onnx")
    assert onnx_format["display_name"] == "ONNX"
    assert onnx_format["available"] is True

    list_before_response = await test_client.get(f"/api/export/{project_id}")
    assert list_before_response.status_code == 200
    assert list_before_response.json() == {"status": "ok", "data": {"exports": []}}

    create_response = await test_client.post(
        f"/api/export/{project_id}",
        json={
            "experiment_id": experiment_id,
            "checkpoint": "best",
            "format": "onnx",
            "options": {"opset_version": 17, "input_shape": [1, 3, 32, 32], "simplify": False},
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()["data"]
    export_id = created["id"]
    assert export_id.startswith("export-")
    assert created["experiment_id"] == experiment_id
    assert created["checkpoint"] == "best"
    assert created["format"] == "onnx"
    assert created["status"] == "completed"
    assert created["output_file"] == "model.onnx"
    assert created["output_size_mb"] is not None
    assert created["validation"] is not None
    assert created["validation"]["passed"] is True

    list_after_response = await test_client.get(f"/api/export/{project_id}")
    assert list_after_response.status_code == 200
    exports = list_after_response.json()["data"]["exports"]
    assert len(exports) == 1
    assert exports[0]["id"] == export_id
    assert exports[0]["status"] == "completed"

    get_response = await test_client.get(f"/api/export/{project_id}/{export_id}")
    assert get_response.status_code == 200
    get_payload = get_response.json()["data"]
    assert get_payload["id"] == export_id
    assert get_payload["validation"]["passed"] is True

    download_response = await test_client.get(f"/api/export/{project_id}/{export_id}/download")
    assert download_response.status_code == 200
    assert "application/octet-stream" in download_response.headers["content-type"]
    assert "model.onnx" in download_response.headers["content-disposition"]
    assert len(download_response.content) > 0

    delete_response = await test_client.delete(f"/api/export/{project_id}/{export_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "ok",
        "data": {
            "project_id": project_id,
            "export_id": export_id,
            "deleted": True,
        },
    }

    list_final_response = await test_client.get(f"/api/export/{project_id}")
    assert list_final_response.status_code == 200
    assert list_final_response.json() == {"status": "ok", "data": {"exports": []}}


@pytest.mark.asyncio
async def test_export_api_conflict_and_not_found_errors(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Export API Error Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="export_api_error_source",
    )
    experiment_id = await _create_experiment(test_client, project_id, name="Error Baseline")

    conflict_response = await test_client.post(
        f"/api/export/{project_id}",
        json={"experiment_id": experiment_id, "checkpoint": "best", "format": "onnx"},
    )
    assert conflict_response.status_code == 409
    conflict_payload = conflict_response.json()
    assert conflict_payload["status"] == "error"
    assert conflict_payload["error"]["code"] == "CONFLICT"
    assert "must be completed" in conflict_payload["error"]["message"]

    missing_get_response = await test_client.get(f"/api/export/{project_id}/export-deadbeef")
    assert missing_get_response.status_code == 404
    missing_get_payload = missing_get_response.json()
    assert missing_get_payload["status"] == "error"
    assert missing_get_payload["error"]["code"] == "NOT_FOUND"

    missing_download_response = await test_client.get(
        f"/api/export/{project_id}/export-deadbeef/download"
    )
    assert missing_download_response.status_code == 404
    missing_download_payload = missing_download_response.json()
    assert missing_download_payload["status"] == "error"
    assert missing_download_payload["error"]["code"] == "NOT_FOUND"


async def _create_project(test_client, *, name: str) -> str:
    response = await test_client.post(
        "/api/projects",
        json={"name": name, "task": "classification"},
    )
    assert response.status_code == 200
    return response.json()["data"]["id"]


async def _create_experiment(test_client, project_id: str, *, name: str) -> str:
    response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": name},
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


def _mark_experiment_completed_and_seed_checkpoints(
    test_client,
    project_id: str,
    experiment_id: str,
) -> None:
    app = test_client._transport.app
    training_service = app.state.training_service
    experiment = training_service.get_experiment(project_id, experiment_id)
    completed = experiment.model_copy(
        update={
            "status": "completed",
            "started_at": _utc_now(),
            "completed_at": _utc_now(),
        }
    )
    training_service.store.write(
        training_service.paths.experiment_metadata_file(project_id, experiment_id),
        completed.model_dump(mode="json"),
    )

    checkpoints_dir = training_service.paths.experiment_checkpoints_dir(project_id, experiment_id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, checkpoints_dir / "best.ckpt")
    torch.save({"state_dict": {}}, checkpoints_dir / "last.ckpt")


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
