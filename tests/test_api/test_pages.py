from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.schemas.evaluation import (
    ClassificationAggregateMetrics,
    ClassificationLabelRef,
    ClassificationPerImageResult,
    ClassificationPrediction,
    EvaluationProgress,
    EvaluationRecord,
    EvaluationResultsFile,
)


@pytest.mark.asyncio
async def test_root_redirects_to_projects_page(test_client) -> None:
    response = await test_client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/projects"


@pytest.mark.asyncio
async def test_projects_page_renders_project_cards(test_client) -> None:
    await test_client.post(
        "/api/projects",
        json={"name": "GUI Project", "task": "classification"},
    )

    response = await test_client.get("/projects")

    assert response.status_code == 200
    assert "Projects" in response.text
    assert "GUI Project" in response.text
    assert "/projects/" in response.text
    assert "/dataset" in response.text


@pytest.mark.asyncio
async def test_dataset_page_shows_empty_state_for_selected_project(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert "No dataset imported." in response.text
    assert "Import Dataset" in response.text
    assert 'id="dataset-import-indicator"' in response.text
    assert "Importing dataset..." in response.text
    assert response.text.count('hx-indicator="#dataset-import-indicator"') == 2
    assert "importBusy" in response.text


@pytest.mark.asyncio
async def test_dataset_page_renders_initial_image_grid_after_import(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Browser Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "dataset_page_source"
    (source_root / "cat").mkdir(parents=True, exist_ok=True)
    (source_root / "dog").mkdir(parents=True, exist_ok=True)
    _create_image(source_root / "cat" / "cat_page_001.png", color=(220, 120, 120))
    _create_image(source_root / "dog" / "dog_page_001.png", color=(120, 120, 220))

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert "cat_page_001.png" in response.text
    assert "dog_page_001.png" in response.text
    assert "Showing 1" in response.text
    assert 'option value="500"' in response.text
    assert 'option value="24"' not in response.text
    assert "object-contain" in response.text
    assert 'id="dataset-page-input-top"' in response.text
    assert 'id="dataset-page-input-bottom"' in response.text
    assert 'name="split_name"' in response.text
    assert '<option value="" selected>None</option>' in response.text
    assert f"/api/datasets/{project_id}/thumbnails/cat_page_001.png" in response.text
    assert f"/api/datasets/{project_id}/images" in response.text
    assert "/api/datasets//thumbnails/" not in response.text


@pytest.mark.asyncio
async def test_dataset_page_defaults_to_first_split_for_split_labels(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Split Label Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "dataset_split_label_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert 'name="split_name"' in response.text
    assert '<option value="80-10-10" selected>' in response.text
    assert (
        "bg-emerald-50 text-emerald-700" in response.text
        or "bg-amber-50 text-amber-700" in response.text
        or "bg-sky-50 text-sky-700" in response.text
        or "bg-slate-100 text-slate-600" in response.text
    )


@pytest.mark.asyncio
async def test_hx_project_create_rename_delete_flow(test_client) -> None:
    create_response = await test_client.post(
        "/api/projects",
        data={"name": "HTMX Project", "task": "classification"},
        headers={"HX-Request": "true"},
    )
    assert create_response.status_code == 200
    assert "HTMX Project" in create_response.text
    listed = await test_client.get("/api/projects")
    project_id = listed.json()["data"]["projects"][0]["id"]
    assert create_response.headers.get("HX-Redirect") == f"/projects/{project_id}/dataset"

    rename_response = await test_client.patch(
        f"/api/projects/{project_id}",
        data={"name": "HTMX Renamed"},
        headers={"HX-Request": "true"},
    )
    assert rename_response.status_code == 200
    assert "HTMX Renamed" in rename_response.text

    delete_response = await test_client.delete(
        f"/api/projects/{project_id}",
        headers={"HX-Request": "true"},
    )
    assert delete_response.status_code == 200
    assert "No projects yet." in delete_response.text

    final_list = await test_client.get("/api/projects")
    assert final_list.status_code == 200
    assert final_list.json()["data"]["projects"] == []


@pytest.mark.asyncio
async def test_split_page_shows_dataset_required_state(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Split Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/split")

    assert response.status_code == 200
    assert "No dataset imported." in response.text
    assert "Go to Dataset Page" in response.text
    assert f"/projects/{project_id}/dataset" in response.text
    assert "New Split" in response.text


@pytest.mark.asyncio
async def test_split_page_renders_split_form_and_summary_after_creation(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Split Browser Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "split_page_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    response = await test_client.get(f"/projects/{project_id}/split")

    assert response.status_code == 200
    assert "Create Split" in response.text
    assert "80-10-10" in response.text
    assert "Immutable" in response.text
    assert 'id="split-ratio-train"' in response.text
    assert 'id="split-ratio-val"' in response.text
    assert 'id="split-ratio-test"' in response.text
    assert 'id="split-seed-input"' in response.text
    assert f"/api/splits/{project_id}/preview" in response.text
    assert 'x-on:split-created.window="handleCreateSuccess(' in response.text
    assert f"/projects/{project_id}/dataset?split_name=80-10-10" in response.text


@pytest.mark.asyncio
async def test_training_page_shows_dataset_required_state(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Training Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/training")

    assert response.status_code == 200
    assert "No dataset imported." in response.text
    assert "Go to Dataset Page" in response.text
    assert f"/projects/{project_id}/dataset" in response.text


@pytest.mark.asyncio
async def test_training_page_renders_experiment_editor_and_live_chart_workspace(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Training Browser Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "training_page_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    created_experiment = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Baseline", "split_name": "80-10-10"},
    )
    assert created_experiment.status_code == 200
    experiment_id = created_experiment.json()["data"]["id"]

    response = await test_client.get(
        f"/projects/{project_id}/training?experiment_id={experiment_id}"
    )

    assert response.status_code == 200
    assert "Training" in response.text
    assert "New Experiment" in response.text
    assert "Start Training" in response.text
    assert "Experiment Setup" in response.text
    assert "Architecture" in response.text
    assert "Optimization" in response.text
    assert "Objective (Loss)" in response.text
    assert "Training Settings" in response.text
    assert "Hardware" in response.text
    assert "Data Augmentation" in response.text
    assert "Augmentation Toggles" not in response.text
    assert 'data-augmentation-row="RandomResizedCrop"' in response.text
    assert 'data-augmentation-row="RandomHorizontalFlip"' in response.text
    assert 'data-augmentation-row="RandomRotation"' in response.text
    assert 'data-augmentation-row="ColorJitter"' in response.text
    assert "data-training-augmentation-hidden-fields" in response.text
    assert "data-training-augmentations" in response.text
    assert "training-config-section" in response.text
    assert "data-training-section-header" in response.text
    assert "data-training-section-chevron" in response.text
    assert 'data-training-config-section="hardware"' in response.text
    assert 'name="hardware.selected_devices[]"' in response.text
    assert "data-training-effective-batch" in response.text
    assert "Name and dataset split selection." not in response.text
    assert "Model backbone and head behavior." not in response.text
    assert "Optimizer and learning-rate schedule." not in response.text
    assert "Loss function and loss-specific parameters." not in response.text
    assert "Epoch, batch, and stopping controls." not in response.text
    assert "Device selection and precision settings." not in response.text
    assert "Enable or disable training transforms." not in response.text
    assert "Epoch" in response.text
    assert "Loss Curves" not in response.text
    assert "F1 Curves" not in response.text
    assert "data-training-loss-train-latest" not in response.text
    assert "data-training-loss-val-latest" not in response.text
    assert "data-training-metric-train-latest" not in response.text
    assert "data-training-metric-val-latest" not in response.text
    assert "data-training-loss-x-ticks" in response.text
    assert "data-training-loss-y-ticks" in response.text
    assert "data-training-loss-axis-x" in response.text
    assert "data-training-loss-axis-y" in response.text
    assert "data-training-metric-x-ticks" in response.text
    assert "data-training-metric-y-ticks" in response.text
    assert "data-training-metric-axis-x" in response.text
    assert "data-training-metric-axis-y" in response.text
    assert "data-training-metric-train" in response.text
    assert "data-training-metric-val" in response.text
    assert "data-training-epoch-label" in response.text
    assert "data-training-loss-caption" not in response.text
    assert "data-training-metric-caption" not in response.text
    assert 'id="training-experiment-list"' in response.text
    assert 'id="training-experiment-workspace"' in response.text
    assert f"/api/training/{project_id}/experiments" in response.text
    assert f"/api/training/{project_id}/experiments/{experiment_id}/stream" in response.text
    assert f"/projects/{project_id}/training" in response.text


@pytest.mark.asyncio
async def test_evaluation_page_shows_dataset_required_state(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Evaluation Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/evaluation")

    assert response.status_code == 200
    assert "Evaluation" in response.text
    assert "No dataset imported." in response.text
    assert "Go to Dataset Page" in response.text
    assert f"/projects/{project_id}/dataset" in response.text


@pytest.mark.asyncio
async def test_evaluation_page_renders_completed_experiment_workspace_and_results(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Evaluation Browser Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "evaluation_page_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    completed_experiment_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Eval Ready", "split_name": "80-10-10"},
    )
    assert completed_experiment_response.status_code == 200
    completed_experiment_id = completed_experiment_response.json()["data"]["id"]
    _mark_experiment_completed_and_seed_checkpoints(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
    )
    _seed_completed_evaluation_artifacts(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
    )

    created_experiment_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Still Training", "split_name": "80-10-10"},
    )
    assert created_experiment_response.status_code == 200

    response = await test_client.get(
        f"/projects/{project_id}/evaluation?experiment_id={completed_experiment_id}"
    )

    assert response.status_code == 200
    assert "Completed Experiments" in response.text
    assert "Eval Ready" in response.text
    assert "Still Training" not in response.text
    assert 'id="evaluation-workspace"' in response.text
    assert "Hardware and Configuration" in response.text
    assert "Metrics and Visualizations" in response.text
    assert "Per-Image Results" in response.text
    assert "data-eval-top-cards" in response.text
    assert 'data-eval-card="hardware"' in response.text
    assert 'data-eval-card="metrics"' in response.text
    assert "Confusion Matrix Heatmap" in response.text
    assert 'data-eval-kpi="accuracy"' in response.text
    assert 'data-eval-kpi="f1_macro"' in response.text
    assert 'data-eval-kpi="precision_macro"' in response.text
    assert 'data-eval-kpi="recall_macro"' in response.text
    assert "data-eval-confusion-matrix" in response.text
    assert "data-eval-per-class-table" in response.text
    assert 'data-eval-cm-col-label="cats"' in response.text
    assert 'data-eval-cm-col-label="dogs"' in response.text
    assert "P0" not in response.text
    assert "G0" not in response.text
    assert "Class probabilities" in response.text
    assert "Reset Evaluation" in response.text
    assert f"/api/evaluation/{project_id}/{completed_experiment_id}" in response.text
    assert f"/api/evaluation/{project_id}/{completed_experiment_id}/results" in response.text
    assert f"/api/datasets/{project_id}/thumbnails/cat_page_000.png" in response.text
    assert f"/projects/{project_id}/evaluation" in response.text


@pytest.mark.asyncio
async def test_evaluation_page_confusion_matrix_uses_class_labels(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Evaluation Matrix Labels Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "evaluation_matrix_labels_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    completed_experiment_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Eval Matrix Labels", "split_name": "80-10-10"},
    )
    assert completed_experiment_response.status_code == 200
    completed_experiment_id = completed_experiment_response.json()["data"]["id"]
    _mark_experiment_completed_and_seed_checkpoints(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
    )
    _seed_completed_evaluation_artifacts(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
    )

    response = await test_client.get(
        f"/projects/{project_id}/evaluation?experiment_id={completed_experiment_id}"
    )

    assert response.status_code == 200
    assert "data-eval-confusion-matrix" in response.text
    assert 'data-eval-cm-col-label="cats"' in response.text
    assert 'data-eval-cm-col-label="dogs"' in response.text
    assert 'data-eval-cm-row-label="cats"' in response.text
    assert 'data-eval-cm-row-label="dogs"' in response.text


@pytest.mark.asyncio
async def test_evaluation_page_confusion_matrix_handles_zero_support_rows(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Evaluation Zero Support Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "evaluation_zero_support_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    created_split = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert created_split.status_code == 200

    completed_experiment_response = await test_client.post(
        f"/api/training/{project_id}/experiments",
        json={"name": "Eval Zero Support", "split_name": "80-10-10"},
    )
    assert completed_experiment_response.status_code == 200
    completed_experiment_id = completed_experiment_response.json()["data"]["id"]
    _mark_experiment_completed_and_seed_checkpoints(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
    )

    _seed_completed_evaluation_artifacts(
        test_client,
        project_id=project_id,
        experiment_id=completed_experiment_id,
        aggregate_override=ClassificationAggregateMetrics(
            accuracy=0.5,
            precision_macro=0.5,
            recall_macro=0.25,
            f1_macro=0.33,
            confusion_matrix=[[0, 0], [1, 1]],
            per_class={
                "cats": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                "dogs": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "support": 2},
            },
        ),
    )

    response = await test_client.get(
        f"/projects/{project_id}/evaluation?experiment_id={completed_experiment_id}"
    )

    assert response.status_code == 200
    assert "Confusion Matrix Heatmap" in response.text
    assert "data-eval-confusion-matrix" in response.text
    assert 'data-eval-cm-row-label="cats"' in response.text


def _mark_experiment_completed_and_seed_checkpoints(
    test_client,
    *,
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
    (checkpoints_dir / "best.ckpt").write_bytes(b"best")
    (checkpoints_dir / "last.ckpt").write_bytes(b"last")


def _seed_completed_evaluation_artifacts(
    test_client,
    *,
    project_id: str,
    experiment_id: str,
    aggregate_override: ClassificationAggregateMetrics | None = None,
) -> None:
    app = test_client._transport.app
    evaluation_service = app.state.evaluation_service
    now = _utc_now()

    record = EvaluationRecord(
        checkpoint="best",
        split_subsets=["test"],
        batch_size=8,
        device="cpu",
        status="completed",
        progress=EvaluationProgress(processed=4, total=4),
        created_at=now,
        started_at=now,
        completed_at=now,
        error=None,
    )
    aggregate = aggregate_override or ClassificationAggregateMetrics(
        accuracy=0.75,
        precision_macro=0.75,
        recall_macro=0.75,
        f1_macro=0.75,
        confusion_matrix=[[2, 0], [1, 1]],
        per_class={
            "cats": {"precision": 0.67, "recall": 1.0, "f1": 0.8, "support": 2},
            "dogs": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "support": 2},
        },
    )
    results = EvaluationResultsFile(
        results=[
            ClassificationPerImageResult(
                filename="cat_page_000.png",
                subset="test",
                ground_truth=ClassificationLabelRef(class_id=0, class_name="cats"),
                prediction=ClassificationPrediction(
                    class_id=0,
                    class_name="cats",
                    confidence=0.93,
                ),
                correct=True,
                probabilities={"cats": 0.93, "dogs": 0.07},
            ),
            ClassificationPerImageResult(
                filename="dog_page_000.png",
                subset="test",
                ground_truth=ClassificationLabelRef(class_id=1, class_name="dogs"),
                prediction=ClassificationPrediction(
                    class_id=0,
                    class_name="cats",
                    confidence=0.62,
                ),
                correct=False,
                probabilities={"cats": 0.62, "dogs": 0.38},
            ),
        ]
    )

    evaluation_service.store.write(
        evaluation_service.paths.experiment_evaluation_metadata_file(project_id, experiment_id),
        record.model_dump(mode="json"),
    )
    evaluation_service.store.write(
        evaluation_service.paths.experiment_evaluation_aggregate_file(project_id, experiment_id),
        aggregate.model_dump(mode="json"),
    )
    evaluation_service.store.write(
        evaluation_service.paths.experiment_evaluation_results_file(project_id, experiment_id),
        results.model_dump(mode="json"),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_classification_source(source_root: Path, *, cats: int, dogs: int) -> None:
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    (source_root / "dogs").mkdir(parents=True, exist_ok=True)

    for index in range(cats):
        _create_image(source_root / "cats" / f"cat_page_{index:03d}.png", color=(220, 120, 120))
    for index in range(dogs):
        _create_image(source_root / "dogs" / f"dog_page_{index:03d}.png", color=(120, 120, 220))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(240, 180), color=color)
    image.save(path)
