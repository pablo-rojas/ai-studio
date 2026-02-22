from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import app.export.onnx_export as onnx_export_module
from app.core.dataset_service import DatasetService
from app.core.exceptions import CheckpointNotFoundError, ConflictError
from app.core.export_service import ExportService
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImportRequest
from app.schemas.export import ExportCreate, OnnxExportOptions
from app.schemas.project import ProjectCreate
from app.schemas.split import SplitCreateRequest, SplitRatios
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


class _ColorHeuristicClassifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        red_mean = images[:, 0, :, :].mean(dim=(1, 2))
        return torch.stack([red_mean, 1.0 - red_mean], dim=1)


def test_export_service_creates_onnx_and_persists_metadata(
    workspace: Path,
    monkeypatch,
) -> None:
    setup = _build_services_and_experiment(workspace, status="completed")
    export_service: ExportService = setup["export_service"]
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]
    paths: WorkspacePaths = setup["paths"]

    monkeypatch.setattr(
        onnx_export_module,
        "create_model",
        lambda *args, **kwargs: _ColorHeuristicClassifier(),
    )

    record = export_service.create_export(
        project_id,
        ExportCreate(
            experiment_id=experiment_id,
            checkpoint="best",
            format="onnx",
            options=OnnxExportOptions(input_shape=[1, 3, 32, 32], simplify=False),
        ),
    )

    assert record.status == "completed"
    assert record.output_file == "model.onnx"
    assert record.output_size_mb is not None
    assert record.validation is not None
    assert record.validation.passed

    export_file = paths.export_dir(project_id, record.id) / "model.onnx"
    assert export_file.exists()

    persisted = export_service.get_export(project_id, record.id)
    assert persisted.id == record.id
    assert persisted.validation is not None

    summaries = export_service.list_exports(project_id)
    assert [item.id for item in summaries] == [record.id]

    resolved_output = export_service.resolve_output_file(project_id, record.id)
    assert resolved_output == export_file

    export_service.delete_export(project_id, record.id)
    assert export_service.list_exports(project_id) == []


def test_export_service_requires_completed_experiment_and_existing_checkpoint(
    workspace: Path,
) -> None:
    setup = _build_services_and_experiment(workspace, status="created")
    export_service: ExportService = setup["export_service"]
    store: JsonStore = setup["store"]
    paths: WorkspacePaths = setup["paths"]
    project_id = setup["project_id"]
    experiment_id = setup["experiment_id"]

    with pytest.raises(ConflictError, match="must be completed"):
        export_service.create_export(project_id, ExportCreate(experiment_id=experiment_id))

    experiment = setup["training_service"].get_experiment(project_id, experiment_id)
    completed = experiment.model_copy(update={"status": "completed"})
    store.write(
        paths.experiment_metadata_file(project_id, experiment_id),
        completed.model_dump(mode="json"),
    )

    checkpoint_path = paths.experiment_checkpoints_dir(project_id, experiment_id) / "best.ckpt"
    checkpoint_path.unlink()

    with pytest.raises(CheckpointNotFoundError, match="Checkpoint 'best.ckpt'"):
        export_service.create_export(project_id, ExportCreate(experiment_id=experiment_id))


def _build_services_and_experiment(
    workspace: Path,
    *,
    status: str,
) -> dict[str, object]:
    source_root = workspace.parent / f"export_service_source_{status}"
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
    export_service = ExportService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
        training_service=training_service,
    )

    project = project_service.create_project(
        ProjectCreate(name="Export Service Project", task="classification")
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
            seed=11,
        ),
    )

    experiment = training_service.create_experiment(project.id)
    updated = experiment.model_copy(update={"status": status})
    store.write(
        paths.experiment_metadata_file(project.id, experiment.id),
        updated.model_dump(mode="json"),
    )

    checkpoints_dir = paths.experiment_checkpoints_dir(project.id, experiment.id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}}, checkpoints_dir / "best.ckpt")
    torch.save({"state_dict": {}}, checkpoints_dir / "last.ckpt")

    return {
        "export_service": export_service,
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

    for index in range(20):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(220, 120, 120))
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 220))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(32, 32), color=color)
    image.save(path)
