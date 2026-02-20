from __future__ import annotations

import pytest
import torch
from PIL import Image

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


def test_build_augmentation_pipeline_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation transform"):
        build_augmentation_pipeline([{"name": "UnknownTransform", "params": {}}])
