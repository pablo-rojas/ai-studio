from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import app.evaluation.evaluator as evaluator_module
from app.core.dataset_service import DatasetService
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImportRequest
from app.schemas.evaluation import EvaluationConfig
from app.schemas.project import ProjectCreate
from app.schemas.split import SplitCreateRequest, SplitRatios
from app.schemas.training import AugmentationConfig, AugmentationStep
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


class _ColorHeuristicClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        red_mean = images[:, 0, :, :].mean(dim=(1, 2))
        return torch.stack([red_mean, 1.0 - red_mean], dim=1)


class _ToyDetector(nn.Module):
    def forward(self, images):  # type: ignore[override]
        predictions = []
        for _ in images:
            predictions.append(
                {
                    "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.97], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                }
            )
        return predictions


def test_evaluator_runs_classification_and_collects_subset_tagged_results(
    workspace: Path,
    monkeypatch,
) -> None:
    setup = _build_completed_experiment(workspace)
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]
    dataset = setup["dataset"]
    experiment = setup["experiment"]
    paths = setup["paths"]

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    progress_updates: list[tuple[int, int]] = []
    evaluator = evaluator_module.Evaluator(
        config=EvaluationConfig(
            checkpoint="best",
            split_subsets=["val", "test"],
            batch_size=8,
            device="cpu",
        ),
        project_id=project_id,
        experiment_id=experiment_id,
        dataset=dataset,
        experiment=experiment,
        images_dir=paths.dataset_images_dir(project_id),
        checkpoint_path=paths.experiment_checkpoints_dir(project_id, experiment_id) / "best.ckpt",
        progress_callback=lambda processed, total: progress_updates.append((processed, total)),
    )
    output = evaluator.run()

    assert output.results
    assert {result.subset for result in output.results} == {"val", "test"}
    assert output.aggregate.accuracy == pytest.approx(1.0)
    assert output.aggregate.confusion_matrix == [[8, 0], [0, 8]]
    assert set(output.aggregate.per_class) == set(dataset.classes)

    assert progress_updates
    assert progress_updates[0][0] == 0
    assert progress_updates[-1][0] == progress_updates[-1][1] == len(output.results)


def test_evaluator_runs_object_detection_and_collects_tp_fp_fn_results(
    workspace: Path,
    monkeypatch,
) -> None:
    setup = _build_completed_detection_experiment(workspace)
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]
    dataset = setup["dataset"]
    experiment = setup["experiment"]
    paths = setup["paths"]

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _ToyDetector(),
    )

    progress_updates: list[tuple[int, int]] = []
    evaluator = evaluator_module.Evaluator(
        config=EvaluationConfig(
            checkpoint="best",
            split_subsets=["val", "test"],
            batch_size=4,
            device="cpu",
        ),
        project_id=project_id,
        experiment_id=experiment_id,
        dataset=dataset,
        experiment=experiment,
        images_dir=paths.dataset_images_dir(project_id),
        checkpoint_path=paths.experiment_checkpoints_dir(project_id, experiment_id) / "best.ckpt",
        progress_callback=lambda processed, total: progress_updates.append((processed, total)),
    )
    output = evaluator.run()

    assert output.results
    assert {result.subset for result in output.results} == {"val", "test"}
    assert output.aggregate.mAP_50 == pytest.approx(1.0)
    assert output.aggregate.mAP_50_95 == pytest.approx(1.0)
    assert output.aggregate.total_tp == len(output.results)
    assert output.aggregate.total_fp == 0
    assert output.aggregate.total_fn == 0

    assert progress_updates
    assert progress_updates[0][0] == 0
    assert progress_updates[-1][0] == progress_updates[-1][1] == len(output.results)


def _build_completed_experiment(workspace: Path) -> dict[str, object]:
    source_root = workspace.parent / "evaluator_source"
    _build_classification_source(source_root)

    paths = WorkspacePaths(root=workspace)
    store = JsonStore()
    project_service = ProjectService(paths=paths, store=store)
    dataset_service = DatasetService(paths=paths, store=store, project_service=project_service)
    split_service = SplitService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    training_service = TrainingService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )

    project = project_service.create_project(
        ProjectCreate(name="Evaluator Project", task="classification")
    )
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )
    split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="80-10-10",
            ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
            seed=7,
        ),
    )

    experiment = training_service.create_experiment(project.id)
    completed = experiment.model_copy(
        update={
            "status": "completed",
            "augmentations": AugmentationConfig(
                train=[AugmentationStep(name="ToImage")],
                val=[AugmentationStep(name="ToImage")],
            ),
        }
    )
    store.write(
        paths.experiment_metadata_file(project.id, completed.id),
        completed.model_dump(mode="json"),
    )

    checkpoints_dir = paths.experiment_checkpoints_dir(project.id, completed.id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, checkpoints_dir / "best.ckpt")

    return {
        "project_id": project.id,
        "experiment_id": completed.id,
        "dataset": dataset_service.get_dataset(project.id),
        "experiment": training_service.get_experiment(project.id, completed.id),
        "paths": paths,
    }


def _build_classification_source(source_root: Path) -> None:
    cats_dir = source_root / "cats"
    dogs_dir = source_root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for index in range(40):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(220, 120, 120))
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 220))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(32, 32), color=color)
    image.save(path)


def _build_completed_detection_experiment(workspace: Path) -> dict[str, object]:
    source_root = workspace.parent / "evaluator_detection_source"
    _build_detection_source(source_root)

    paths = WorkspacePaths(root=workspace)
    store = JsonStore()
    project_service = ProjectService(paths=paths, store=store)
    dataset_service = DatasetService(paths=paths, store=store, project_service=project_service)
    split_service = SplitService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    training_service = TrainingService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )

    project = project_service.create_project(
        ProjectCreate(name="Evaluator Detection Project", task="object_detection")
    )
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="coco"),
    )
    split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="50-25-25",
            ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
            seed=3,
        ),
    )

    experiment = training_service.create_experiment(project.id)
    completed = experiment.model_copy(
        update={
            "status": "completed",
            "augmentations": AugmentationConfig(
                train=[AugmentationStep(name="ToImage")],
                val=[AugmentationStep(name="ToImage")],
            ),
        }
    )
    store.write(
        paths.experiment_metadata_file(project.id, completed.id),
        completed.model_dump(mode="json"),
    )

    checkpoints_dir = paths.experiment_checkpoints_dir(project.id, completed.id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, checkpoints_dir / "best.ckpt")

    return {
        "project_id": project.id,
        "experiment_id": completed.id,
        "dataset": dataset_service.get_dataset(project.id),
        "experiment": training_service.get_experiment(project.id, completed.id),
        "paths": paths,
    }


def _build_detection_source(source_root: Path) -> None:
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "img_001.png", color=(120, 120, 120))
    _create_image(images_dir / "img_002.png", color=(130, 130, 130))
    _create_image(images_dir / "img_003.png", color=(140, 140, 140))
    _create_image(images_dir / "img_004.png", color=(150, 150, 150))

    payload = {
        "images": [
            {"id": 1, "file_name": "img_001.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "img_002.png", "width": 32, "height": 32},
            {"id": 3, "file_name": "img_003.png", "width": 32, "height": 32},
            {"id": 4, "file_name": "img_004.png", "width": 32, "height": 32},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 16, 16]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [4, 4, 16, 16]},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [4, 4, 16, 16]},
            {"id": 4, "image_id": 4, "category_id": 1, "bbox": [4, 4, 16, 16]},
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    (source_root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
