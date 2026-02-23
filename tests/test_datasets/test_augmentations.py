from __future__ import annotations

import pytest
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from app.datasets.augmentations import build_augmentation_pipeline
from app.models.catalog import get_default_augmentations


def test_build_augmentation_pipeline_from_classification_defaults() -> None:
    defaults = get_default_augmentations("classification")
    train_pipeline = build_augmentation_pipeline(defaults.train)
    val_pipeline = build_augmentation_pipeline(defaults.val)
    image = Image.new("RGB", (320, 320), color=(120, 130, 140))

    train_output = train_pipeline(image)
    val_output = val_pipeline(image)

    assert isinstance(train_output, torch.Tensor)
    assert isinstance(val_output, torch.Tensor)
    assert train_output.dtype == torch.float32
    assert val_output.dtype == torch.float32
    assert tuple(train_output.shape[-2:]) == (224, 224)
    assert tuple(val_output.shape[-2:]) == (224, 224)


def test_build_augmentation_pipeline_from_detection_defaults() -> None:
    defaults = get_default_augmentations("object_detection")
    train_pipeline = build_augmentation_pipeline(defaults.train)
    val_pipeline = build_augmentation_pipeline(defaults.val)
    image = Image.new("RGB", (320, 320), color=(120, 130, 140))
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            torch.tensor([[10.0, 20.0, 90.0, 120.0]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(320, 320),
        ),
        "labels": torch.tensor([0], dtype=torch.int64),
    }

    train_image, train_target = train_pipeline(image, target)
    val_image, val_target = val_pipeline(image, target)

    assert isinstance(train_image, torch.Tensor)
    assert isinstance(val_image, torch.Tensor)
    assert train_image.dtype == torch.float32
    assert val_image.dtype == torch.float32
    assert train_target["labels"].dtype == torch.int64
    assert val_target["labels"].dtype == torch.int64


def test_build_augmentation_pipeline_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation transform"):
        build_augmentation_pipeline([{"name": "UnknownTransform", "params": {}}])


def test_build_augmentation_pipeline_wraps_transform_with_apply_probability() -> None:
    pipeline = build_augmentation_pipeline(
        [
            {
                "name": "RandomRotation",
                "params": {"degrees": [-15, 15], "apply_p": 0.25},
            }
        ]
    )

    wrapped_transform = pipeline.transforms[0]
    assert isinstance(wrapped_transform, v2.RandomApply)
    assert wrapped_transform.p == pytest.approx(0.25)
    assert isinstance(wrapped_transform.transforms[0], v2.RandomRotation)


def test_build_augmentation_pipeline_skips_wrapper_for_apply_probability_one() -> None:
    pipeline = build_augmentation_pipeline(
        [
            {
                "name": "RandomRotation",
                "params": {"degrees": [-15, 15], "apply_p": 1.0},
            }
        ]
    )

    assert isinstance(pipeline.transforms[0], v2.RandomRotation)


def test_build_augmentation_pipeline_rejects_invalid_apply_probability() -> None:
    with pytest.raises(ValueError, match="apply_p"):
        build_augmentation_pipeline(
            [
                {
                    "name": "RandomRotation",
                    "params": {"degrees": [-15, 15], "apply_p": 1.2},
                }
            ]
        )
