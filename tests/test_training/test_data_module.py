from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.datasets.base import AIStudioDataModule
from app.models.catalog import get_default_augmentations
from app.schemas.dataset import DatasetMetadata, ImageStats


def test_data_module_builds_split_dataloaders(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "cat_1.png", color=(200, 100, 100))
    _create_image(images_dir / "cat_2.png", color=(190, 110, 110))
    _create_image(images_dir / "dog_1.png", color=(100, 100, 200))
    _create_image(images_dir / "dog_2.png", color=(110, 110, 190))

    dataset = DatasetMetadata.model_validate(
        {
            "version": "1.0",
            "id": "dataset-a1b2c3d4",
            "task": "classification",
            "source_format": "image_folders",
            "source_path": str(tmp_path),
            "imported_at": datetime.now(timezone.utc),
            "classes": ["cat", "dog"],
            "split_names": ["default"],
            "image_stats": ImageStats(
                num_images=4,
                min_width=64,
                max_width=64,
                min_height=64,
                max_height=64,
                formats=["png"],
            ),
            "images": [
                {
                    "filename": "cat_1.png",
                    "width": 64,
                    "height": 64,
                    "split": ["train"],
                    "annotations": [{"type": "label", "class_id": 0, "class_name": "cat"}],
                },
                {
                    "filename": "cat_2.png",
                    "width": 64,
                    "height": 64,
                    "split": ["val"],
                    "annotations": [{"type": "label", "class_id": 0, "class_name": "cat"}],
                },
                {
                    "filename": "dog_1.png",
                    "width": 64,
                    "height": 64,
                    "split": ["train"],
                    "annotations": [{"type": "label", "class_id": 1, "class_name": "dog"}],
                },
                {
                    "filename": "dog_2.png",
                    "width": 64,
                    "height": 64,
                    "split": ["test"],
                    "annotations": [{"type": "label", "class_id": 1, "class_name": "dog"}],
                },
            ],
        }
    )

    data_module = AIStudioDataModule(
        dataset=dataset,
        images_dir=images_dir,
        split_name="default",
        augmentations=get_default_augmentations("classification"),
        batch_size=2,
    )
    data_module.setup()

    assert data_module.num_classes == 2
    assert len(data_module.train_dataset or []) == 2
    assert len(data_module.val_dataset or []) == 1
    assert len(data_module.test_dataset or []) == 1

    images, targets = next(iter(data_module.train_dataloader()))
    assert isinstance(images, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert images.shape[0] == 2
    assert targets.dtype == torch.int64


def test_data_module_rejects_unknown_split_name(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "sample.png", color=(120, 120, 120))

    dataset = DatasetMetadata.model_validate(
        {
            "version": "1.0",
            "id": "dataset-aabbccdd",
            "task": "classification",
            "source_format": "image_folders",
            "source_path": str(tmp_path),
            "imported_at": datetime.now(timezone.utc),
            "classes": ["class_a"],
            "split_names": ["split_a"],
            "image_stats": ImageStats(
                num_images=1,
                min_width=64,
                max_width=64,
                min_height=64,
                max_height=64,
                formats=["png"],
            ),
            "images": [
                {
                    "filename": "sample.png",
                    "width": 64,
                    "height": 64,
                    "split": ["train"],
                    "annotations": [{"type": "label", "class_id": 0, "class_name": "class_a"}],
                }
            ],
        }
    )

    data_module = AIStudioDataModule(
        dataset=dataset,
        images_dir=images_dir,
        split_name="missing_split",
        augmentations=get_default_augmentations("classification"),
        batch_size=1,
    )
    with pytest.raises(ValueError, match="Unknown split"):
        data_module.setup()


def test_data_module_builds_detection_split_dataloaders(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _create_image(images_dir / "img_1.png", color=(200, 100, 100))
    _create_image(images_dir / "img_2.png", color=(190, 110, 110))
    _create_image(images_dir / "img_3.png", color=(100, 100, 200))

    dataset = DatasetMetadata.model_validate(
        {
            "version": "1.0",
            "id": "dataset-00112233",
            "task": "object_detection",
            "source_format": "coco",
            "source_path": str(tmp_path),
            "imported_at": datetime.now(timezone.utc),
            "classes": ["cat", "dog"],
            "split_names": ["default"],
            "image_stats": ImageStats(
                num_images=3,
                min_width=64,
                max_width=64,
                min_height=64,
                max_height=64,
                formats=["png"],
            ),
            "images": [
                {
                    "filename": "img_1.png",
                    "width": 64,
                    "height": 64,
                    "split": ["train"],
                    "annotations": [
                        {"type": "bbox", "class_id": 0, "class_name": "cat", "bbox": [5, 6, 20, 21]}
                    ],
                },
                {
                    "filename": "img_2.png",
                    "width": 64,
                    "height": 64,
                    "split": ["train"],
                    "annotations": [],
                },
                {
                    "filename": "img_3.png",
                    "width": 64,
                    "height": 64,
                    "split": ["val"],
                    "annotations": [
                        {
                            "type": "bbox",
                            "class_id": 1,
                            "class_name": "dog",
                            "bbox": [10, 8, 15, 18],
                        }
                    ],
                },
            ],
        }
    )

    data_module = AIStudioDataModule(
        dataset=dataset,
        images_dir=images_dir,
        split_name="default",
        augmentations=get_default_augmentations("object_detection"),
        batch_size=2,
    )
    data_module.setup()

    assert data_module.num_classes == 2
    assert len(data_module.train_dataset or []) == 2
    assert len(data_module.val_dataset or []) == 1
    assert len(data_module.test_dataset or []) == 0

    images, targets = next(iter(data_module.train_dataloader()))
    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert len(images) == 2
    assert len(targets) == 2
    assert all(isinstance(image, torch.Tensor) for image in images)
    assert all(isinstance(target["boxes"], torch.Tensor) for target in targets)
    assert all(isinstance(target["labels"], torch.Tensor) for target in targets)


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(64, 64), color=color)
    image.save(path)
