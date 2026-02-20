from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.core.dataset_service import DatasetService
from app.core.project_service import ProjectService
from app.schemas.dataset import DatasetImportRequest, DatasetMetadata
from app.schemas.project import ProjectCreate
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


def test_import_image_folders_creates_dataset_metadata_and_thumbnails(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="Folder Import Project", task="classification")
    )
    source_root = workspace.parent / "folder_source"
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    (source_root / "dogs").mkdir(parents=True, exist_ok=True)

    _create_image(source_root / "cats" / "shared_name.png", size=(300, 200), color=(220, 90, 90))
    _create_image(source_root / "cats" / "cat_2.jpg", size=(250, 160), color=(220, 130, 130))
    _create_image(source_root / "dogs" / "shared_name.png", size=(180, 180), color=(90, 90, 220))

    service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    metadata = service.import_dataset(
        project.id,
        DatasetImportRequest(
            source_path=str(source_root),
            source_format="image_folders",
        ),
    )

    assert metadata.task == "classification"
    assert metadata.source_format == "image_folders"
    assert metadata.image_stats.num_images == 3
    assert metadata.split_names == []
    assert metadata.classes == ["cats", "dogs"]
    assert all(image.split == [] for image in metadata.images)

    copied_names = [image.filename for image in metadata.images]
    assert len(set(copied_names)) == 3
    assert "shared_name.png" in copied_names
    assert "shared_name_1.png" in copied_names

    paths = WorkspacePaths(root=workspace)
    dataset_json_path = paths.dataset_metadata_file(project.id)
    persisted_data = JsonStore().read(dataset_json_path)
    persisted = DatasetMetadata.model_validate(persisted_data)
    assert persisted.id == metadata.id

    for image in metadata.images:
        copied_image_path = paths.dataset_images_dir(project.id) / image.filename
        thumbnail_path = paths.dataset_thumbnails_dir(project.id) / image.filename
        assert copied_image_path.exists()
        assert thumbnail_path.exists()
        with Image.open(thumbnail_path) as thumbnail:
            assert thumbnail.width <= 128
            assert thumbnail.height <= 128


def test_import_csv_classification_dataset(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="CSV Import Project", task="classification")
    )
    source_root = workspace.parent / "csv_source"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "bird_1.png", size=(120, 90), color=(20, 120, 200))
    _create_image(images_dir / "cat_1.png", size=(180, 160), color=(120, 20, 200))

    labels_csv = source_root / "labels.csv"
    labels_csv.write_text("filename,label\nbird_1.png,bird\ncat_1.png,cat\n", encoding="utf-8")

    service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    metadata = service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root)),
    )

    assert metadata.source_format == "csv"
    assert metadata.classes == ["bird", "cat"]
    assert metadata.image_stats.num_images == 2
    assert metadata.images[0].annotations[0].type == "label"


def test_import_coco_classification_dataset(
    project_service: ProjectService,
    workspace: Path,
) -> None:
    project = project_service.create_project(
        ProjectCreate(name="COCO Import Project", task="classification")
    )
    source_root = workspace.parent / "coco_source"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "img_1.png", size=(400, 320), color=(100, 150, 200))
    _create_image(images_dir / "img_2.png", size=(220, 180), color=(140, 90, 200))

    coco_payload = {
        "images": [
            {"id": 1, "file_name": "img_1.png", "width": 400, "height": 320},
            {"id": 2, "file_name": "img_2.png", "width": 220, "height": 180},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 5},
            {"id": 11, "image_id": 2, "category_id": 8},
        ],
        "categories": [
            {"id": 5, "name": "cat"},
            {"id": 8, "name": "dog"},
        ],
    }
    (source_root / "annotations.json").write_text(
        json.dumps(coco_payload),
        encoding="utf-8",
    )

    service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    metadata = service.import_dataset(
        project.id,
        DatasetImportRequest(
            source_path=str(source_root),
            source_format="coco",
        ),
    )

    assert metadata.source_format == "coco"
    assert metadata.classes == ["cat", "dog"]
    assert metadata.image_stats.num_images == 2
    assert metadata.images[0].annotations[0].type == "label"


def _create_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size=size, color=color)
    image.save(path)
