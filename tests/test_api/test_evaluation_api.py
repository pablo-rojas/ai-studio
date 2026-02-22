from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import app.evaluation.evaluator as evaluator_module


class _ColorHeuristicClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        red_mean = images[:, 0, :, :].mean(dim=(1, 2))
        return torch.stack([red_mean, 1.0 - red_mean], dim=1)


class _AlwaysFirstClassClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        batch_size = images.shape[0]
        logits = torch.zeros(
            (batch_size, 2),
            dtype=images.dtype,
            device=images.device,
        )
        logits[:, 0] = 1.0
        return logits


@pytest.mark.asyncio
async def test_evaluation_api_end_to_end_flow(
    test_client,
    workspace: Path,
    monkeypatch,
) -> None:
    project_id = await _create_project(test_client, name="Evaluation API Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="evaluation_api_source",
    )
    experiment_id = await _create_experiment(test_client, project_id, name="Eval Baseline")
    _mark_experiment_completed_and_seed_checkpoints(test_client, project_id, experiment_id)

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    start_response = await test_client.post(
        f"/api/evaluation/{project_id}/{experiment_id}",
        json={
            "checkpoint": "best",
            "split_subsets": ["test", "val"],
            "batch_size": 8,
            "device": "cpu",
        },
    )
    assert start_response.status_code == 200
    assert start_response.json() == {"status": "ok", "data": {"status": "completed"}}

    get_response = await test_client.get(f"/api/evaluation/{project_id}/{experiment_id}")
    assert get_response.status_code == 200
    get_payload = get_response.json()["data"]
    assert get_payload["evaluation"]["status"] == "completed"
    assert get_payload["aggregate"] is not None
    assert get_payload["aggregate"]["accuracy"] == pytest.approx(1.0)

    results_response = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/results",
        params={"page": 1, "page_size": 10, "filter_subset": "test"},
    )
    assert results_response.status_code == 200
    results_payload = results_response.json()["data"]
    assert results_payload["total_items"] == 4
    assert all(item["subset"] == "test" for item in results_payload["items"])

    checkpoints_response = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/checkpoints"
    )
    assert checkpoints_response.status_code == 200
    assert checkpoints_response.json() == {
        "status": "ok",
        "data": {"checkpoints": ["best", "last"]},
    }

    delete_response = await test_client.delete(f"/api/evaluation/{project_id}/{experiment_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "ok",
        "data": {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "deleted": True,
        },
    }
    evaluation_dir = (
        workspace / "projects" / project_id / "experiments" / experiment_id / "evaluation"
    )
    assert not evaluation_dir.exists()


@pytest.mark.asyncio
async def test_evaluation_results_filters_and_missing_experiment_error(
    test_client,
    workspace: Path,
    monkeypatch,
) -> None:
    project_id = await _create_project(test_client, name="Evaluation Filter Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="evaluation_filter_source",
    )
    experiment_id = await _create_experiment(test_client, project_id, name="Filter Baseline")
    _mark_experiment_completed_and_seed_checkpoints(test_client, project_id, experiment_id)

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _AlwaysFirstClassClassifier(),
    )

    start_response = await test_client.post(
        f"/api/evaluation/{project_id}/{experiment_id}",
        json={
            "checkpoint": "best",
            "split_subsets": ["test"],
            "batch_size": 8,
            "device": "cpu",
        },
    )
    assert start_response.status_code == 200

    correct_only = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/results",
        params={"filter_correct": True, "page_size": 20},
    )
    assert correct_only.status_code == 200
    correct_payload = correct_only.json()["data"]
    assert correct_payload["total_items"] == 2
    assert all(item["correct"] is True for item in correct_payload["items"])

    incorrect_only = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/results",
        params={"filter_correct": False, "page_size": 20},
    )
    assert incorrect_only.status_code == 200
    incorrect_payload = incorrect_only.json()["data"]
    assert incorrect_payload["total_items"] == 2
    assert all(item["correct"] is False for item in incorrect_payload["items"])

    class_filtered = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/results",
        params={"filter_class": "dogs", "page_size": 20},
    )
    assert class_filtered.status_code == 200
    class_payload = class_filtered.json()["data"]
    assert class_payload["total_items"] == 2
    assert all(
        item["ground_truth"]["class_name"].lower() == "dogs" for item in class_payload["items"]
    )

    missing_response = await test_client.get(f"/api/evaluation/{project_id}/exp-deadbeef")
    assert missing_response.status_code == 404
    missing_payload = missing_response.json()
    assert missing_payload["status"] == "error"
    assert missing_payload["error"]["code"] == "EXPERIMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_evaluation_hx_endpoints_render_workspace_and_results_fragments(
    test_client,
    workspace: Path,
    monkeypatch,
) -> None:
    project_id = await _create_project(test_client, name="Evaluation HX Project")
    await _import_dataset_and_split(
        test_client,
        workspace,
        project_id,
        source_name="evaluation_hx_source",
    )
    experiment_id = await _create_experiment(test_client, project_id, name="HX Baseline")
    _mark_experiment_completed_and_seed_checkpoints(test_client, project_id, experiment_id)

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _AlwaysFirstClassClassifier(),
    )
    started = await test_client.post(
        f"/api/evaluation/{project_id}/{experiment_id}",
        json={
            "checkpoint": "best",
            "split_subsets": ["test"],
            "batch_size": 8,
            "device": "cpu",
        },
    )
    assert started.status_code == 200

    detail_response = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}",
        headers={"HX-Request": "true"},
    )
    assert detail_response.status_code == 200
    assert 'id="evaluation-workspace"' in detail_response.text
    assert "Configuration" in detail_response.text
    assert "Per-Image Results" in detail_response.text
    assert "Reset Evaluation" in detail_response.text

    results_response = await test_client.get(
        f"/api/evaluation/{project_id}/{experiment_id}/results",
        params={
            "page": 1,
            "filter_correct": "",
            "filter_subset": "",
            "sort_by": "confidence",
            "sort_order": "desc",
        },
        headers={"HX-Request": "true"},
    )
    assert results_response.status_code == 200
    assert 'id="evaluation-results-grid"' in results_response.text
    assert "Class probabilities" in results_response.text

    reset_response = await test_client.delete(
        f"/api/evaluation/{project_id}/{experiment_id}",
        headers={"HX-Request": "true"},
    )
    assert reset_response.status_code == 200
    assert 'id="evaluation-workspace"' in reset_response.text
    assert "Evaluate" in reset_response.text
    evaluation_dir = (
        workspace / "projects" / project_id / "experiments" / experiment_id / "evaluation"
    )
    assert not evaluation_dir.exists()


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
    (checkpoints_dir / "notes.txt").write_text("ignore", encoding="utf-8")


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
