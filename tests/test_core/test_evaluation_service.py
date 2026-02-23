from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import app.evaluation.evaluator as evaluator_module
from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.exceptions import ConflictError, NotFoundError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImportRequest
from app.schemas.evaluation import EvaluationConfig, EvaluationResultsQuery
from app.schemas.project import ProjectCreate
from app.schemas.split import SplitCreateRequest, SplitRatios
from app.schemas.training import AugmentationConfig, AugmentationStep
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


class _ColorHeuristicClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        red_mean = images[:, 0, :, :].mean(dim=(1, 2))
        return torch.stack([red_mean, 1.0 - red_mean], dim=1)


def test_evaluation_service_start_get_results_and_reset(
    workspace: Path,
    monkeypatch,
) -> None:
    setup = _build_services_and_experiment(workspace, status="completed")
    evaluation_service: EvaluationService = setup["evaluation_service"]
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]
    paths: WorkspacePaths = setup["paths"]

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    record = evaluation_service.start_evaluation(
        project_id,
        experiment_id,
        EvaluationConfig(
            checkpoint="best",
            split_subsets=["test", "val"],
            batch_size=8,
            device="cpu",
        ),
    )

    assert record.status == "completed"
    assert record.progress.total > 0
    assert record.progress.processed == record.progress.total

    persisted = evaluation_service.get_evaluation(project_id, experiment_id)
    assert persisted.status == "completed"

    aggregate = evaluation_service.get_aggregate_metrics(project_id, experiment_id)
    assert aggregate is not None
    assert aggregate.accuracy == pytest.approx(1.0)
    assert aggregate.confusion_matrix == [[8, 0], [0, 8]]

    page = evaluation_service.get_results(
        project_id,
        experiment_id,
        EvaluationResultsQuery(page=1, page_size=10, filter_subset="test"),
    )
    assert page.total_items == 8
    assert all(item.subset == "test" for item in page.items)
    first_filename = page.items[0].filename

    result = evaluation_service.get_result(project_id, experiment_id, first_filename)
    assert result.filename == first_filename

    with pytest.raises(NotFoundError, match="was not found"):
        evaluation_service.get_result(project_id, experiment_id, "missing.png")

    assert evaluation_service.list_checkpoints(project_id, experiment_id) == ["best", "last"]

    evaluation_service.reset_evaluation(project_id, experiment_id)
    assert not paths.experiment_evaluation_dir(project_id, experiment_id).exists()


def test_evaluation_service_requires_completed_experiment_and_reset_before_rerun(
    workspace: Path,
    monkeypatch,
) -> None:
    setup = _build_services_and_experiment(workspace, status="created")
    evaluation_service: EvaluationService = setup["evaluation_service"]
    training_service: TrainingService = setup["training_service"]
    store: JsonStore = setup["store"]
    paths: WorkspacePaths = setup["paths"]
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]

    monkeypatch.setattr(
        evaluator_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    config = EvaluationConfig(checkpoint="best", split_subsets=["test"], batch_size=8, device="cpu")
    with pytest.raises(ConflictError, match="must be completed"):
        evaluation_service.start_evaluation(project_id, experiment_id, config)

    experiment = training_service.get_experiment(project_id, experiment_id)
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
        paths.experiment_metadata_file(project_id, experiment_id),
        completed.model_dump(mode="json"),
    )

    first = evaluation_service.start_evaluation(project_id, experiment_id, config)
    assert first.status == "completed"

    with pytest.raises(ConflictError, match="Reset it first"):
        evaluation_service.start_evaluation(project_id, experiment_id, config)

    evaluation_service.reset_evaluation(project_id, experiment_id)
    second = evaluation_service.start_evaluation(project_id, experiment_id, config)
    assert second.status == "completed"


def _build_services_and_experiment(
    workspace: Path,
    *,
    status: str,
) -> dict[str, object]:
    source_root = workspace.parent / f"evaluation_service_source_{status}"
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
    evaluation_service = EvaluationService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
        training_service=training_service,
    )

    project = project_service.create_project(
        ProjectCreate(name="Evaluation Service Project", task="classification")
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
            seed=19,
        ),
    )

    experiment = training_service.create_experiment(project.id)
    updated = experiment.model_copy(
        update={
            "status": status,
            "augmentations": AugmentationConfig(
                train=[AugmentationStep(name="ToImage")],
                val=[AugmentationStep(name="ToImage")],
            ),
        }
    )
    store.write(
        paths.experiment_metadata_file(project.id, experiment.id),
        updated.model_dump(mode="json"),
    )

    checkpoints_dir = paths.experiment_checkpoints_dir(project.id, experiment.id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, checkpoints_dir / "best.ckpt")
    torch.save({"state_dict": {}}, checkpoints_dir / "last.ckpt")
    (checkpoints_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    return {
        "evaluation_service": evaluation_service,
        "training_service": training_service,
        "project_id": project.id,
        "experiment_id": experiment.id,
        "paths": paths,
        "store": store,
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
