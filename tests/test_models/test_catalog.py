from __future__ import annotations

import pytest
import torch

from app.models.catalog import (
    CLASSIFICATION_BACKBONES,
    OBJECT_DETECTION_ARCHITECTURES,
    build_default_training_config,
    create_model,
    get_task_config,
    is_task_selectable,
    list_architectures,
    list_selectable_tasks,
)
from app.schemas.training import ModelConfig, TrainingConfig


def test_classification_task_is_selectable_and_registered() -> None:
    selectable = list_selectable_tasks()

    assert "classification" in selectable
    assert "object_detection" in selectable
    assert is_task_selectable("classification")
    assert is_task_selectable("object_detection")

    config = get_task_config("classification")
    assert config.annotation_types == ("label",)
    assert config.default_loss == "cross_entropy"
    assert config.primary_metric == "accuracy"


def test_object_detection_task_is_registered() -> None:
    config = get_task_config("object_detection")

    assert config.annotation_types == ("bbox",)
    assert config.default_loss == "default"
    assert config.primary_metric == "mAP_50"


def test_classification_backbone_catalog_contains_expected_architectures() -> None:
    assert list_architectures("classification") == list(CLASSIFICATION_BACKBONES)


def test_object_detection_catalog_contains_expected_architectures() -> None:
    assert list_architectures("object_detection") == list(OBJECT_DETECTION_ARCHITECTURES)


def test_default_classification_training_config_is_valid() -> None:
    training_config = build_default_training_config("classification", backbone="resnet18")

    assert isinstance(training_config, TrainingConfig)
    assert training_config.model.backbone == "resnet18"
    assert training_config.hyperparameters.optimizer == "adam"
    assert training_config.hyperparameters.scheduler == "cosine"
    assert training_config.hyperparameters.batch_size == 32
    assert training_config.hyperparameters.max_epochs == 50


def test_default_object_detection_training_config_is_valid() -> None:
    training_config = build_default_training_config(
        "object_detection", backbone="fasterrcnn_resnet50"
    )

    assert isinstance(training_config, TrainingConfig)
    assert training_config.model.backbone == "fasterrcnn_resnet50"
    assert training_config.model.head == "object_detection"
    assert training_config.hyperparameters.optimizer == "sgd"
    assert training_config.hyperparameters.scheduler == "step"
    assert training_config.hyperparameters.batch_size == 8
    assert training_config.hyperparameters.loss == "default"


@pytest.mark.parametrize("backbone", CLASSIFICATION_BACKBONES)
def test_create_model_supports_each_classification_backbone(backbone: str) -> None:
    config = ModelConfig(
        backbone=backbone,
        head="classification",
        pretrained=False,
        freeze_backbone=True,
        dropout=0.2,
    )
    model = create_model("classification", backbone, config, num_classes=3)

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 128, 128))

    assert logits.shape == (2, 3)
    assert not any(parameter.requires_grad for parameter in model[0].parameters())
    assert all(parameter.requires_grad for parameter in model[1].parameters())


@pytest.mark.parametrize("architecture", OBJECT_DETECTION_ARCHITECTURES)
def test_create_model_supports_each_detection_architecture(architecture: str) -> None:
    config = ModelConfig(
        backbone=architecture,
        head="object_detection",
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
    )

    model = create_model("object_detection", architecture, config, num_classes=3)
    assert model is not None
