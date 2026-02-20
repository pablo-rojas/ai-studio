from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.dataset_service import DatasetService
from app.core.exceptions import ConflictError, NotFoundError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.schemas.dataset import DatasetImportRequest
from app.schemas.project import ProjectCreate
from app.schemas.split import SplitCreateRequest, SplitPreviewRequest, SplitRatios
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


def test_preview_split_returns_distribution_without_persisting(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="Split Preview Project", task="classification")
    )
    source_root = workspace.parent / "split_preview_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    dataset_service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )

    split_service = SplitService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    preview = split_service.preview_split(
        project.id,
        SplitPreviewRequest(
            ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
            seed=42,
        ),
    )

    assert preview.stats.train == 32
    assert preview.stats.val == 4
    assert preview.stats.test == 4
    assert preview.class_distribution["cats"].train == 16
    assert preview.class_distribution["cats"].val == 2
    assert preview.class_distribution["cats"].test == 2
    assert preview.class_distribution["dogs"].train == 16
    assert preview.class_distribution["dogs"].val == 2
    assert preview.class_distribution["dogs"].test == 2

    dataset_after_preview = dataset_service.get_dataset(project.id)
    assert dataset_after_preview.split_names == []
    assert all(image.split == [] for image in dataset_after_preview.images)


def test_create_split_persists_assignments_and_rejects_duplicate_name(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="Split Create Project", task="classification")
    )
    source_root = workspace.parent / "split_create_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    dataset_service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )
    split_service = SplitService(paths=WorkspacePaths(root=workspace), store=JsonStore())

    created = split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="80-10-10",
            ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
            seed=123,
        ),
    )

    assert created.name == "80-10-10"
    assert created.immutable is True
    assert created.index == 0
    assert created.stats.train == 32
    assert created.stats.val == 4
    assert created.stats.test == 4

    listed = split_service.list_splits(project.id)
    assert len(listed) == 1
    assert listed[0].name == "80-10-10"

    fetched = split_service.get_split(project.id, "80-10-10")
    assert fetched.name == "80-10-10"
    assert fetched.index == 0

    dataset_after_create = dataset_service.get_dataset(project.id)
    assert dataset_after_create.split_names == ["80-10-10"]
    assert all(len(image.split) == 1 for image in dataset_after_create.images)
    assert {image.split[0] for image in dataset_after_create.images} == {"train", "val", "test"}

    with pytest.raises(ConflictError):
        split_service.create_split(
            project.id,
            SplitCreateRequest(
                name="80-10-10",
                ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
                seed=123,
            ),
        )

    split_service.delete_split(project.id, "80-10-10")
    dataset_after_delete = dataset_service.get_dataset(project.id)
    assert dataset_after_delete.split_names == []
    assert all(image.split == [] for image in dataset_after_delete.images)

    with pytest.raises(NotFoundError):
        split_service.get_split(project.id, "80-10-10")


def test_anomaly_split_never_places_anomalous_images_in_train(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="Anomaly Split Project", task="anomaly_detection")
    )
    source_root = workspace.parent / "anomaly_split_source"
    _build_anomaly_source(source_root, normal_count=14, anomalous_count=6)

    dataset_service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )
    split_service = SplitService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="70-20-10",
            ratios=SplitRatios(train=0.7, val=0.2, test=0.1),
            seed=7,
        ),
    )

    dataset_after_create = dataset_service.get_dataset(project.id)
    anomalous_subsets = []
    for image in dataset_after_create.images:
        annotation = image.annotations[0]
        if annotation.type == "anomaly" and annotation.is_anomalous:
            anomalous_subsets.append(image.split[0])

    assert anomalous_subsets
    assert "train" not in anomalous_subsets
    assert set(anomalous_subsets) <= {"val", "test"}


def _build_classification_source(source_root: Path, *, cats: int, dogs: int) -> None:
    cats_dir = source_root / "cats"
    dogs_dir = source_root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for index in range(cats):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(210, 120, 120))
    for index in range(dogs):
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 210))


def _build_anomaly_source(
    source_root: Path,
    *,
    normal_count: int,
    anomalous_count: int,
) -> None:
    normal_dir = source_root / "normal"
    anomaly_dir = source_root / "anomaly"
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    for index in range(normal_count):
        _create_image(normal_dir / f"normal_{index:03d}.png", color=(100, 180, 120))
    for index in range(anomalous_count):
        _create_image(anomaly_dir / f"anomaly_{index:03d}.png", color=(210, 100, 100))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)
