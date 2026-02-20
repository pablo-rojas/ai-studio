from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b3,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
)

from app.models.heads.classification import ClassificationHead
from app.schemas.training import (
    AugmentationConfig,
    AugmentationStep,
    HyperparameterConfig,
    ModelConfig,
    TrainingConfig,
)

TaskType = Literal[
    "classification",
    "anomaly_detection",
    "object_detection",
    "oriented_object_detection",
    "segmentation",
    "instance_segmentation",
    "regression",
]

ACTIVE_PHASES: tuple[int, ...] = tuple(range(1, 9))

TASK_PHASE_BY_TYPE: dict[TaskType, int] = {
    "classification": 8,
    "anomaly_detection": 23,
    "object_detection": 19,
    "oriented_object_detection": 25,
    "segmentation": 20,
    "instance_segmentation": 21,
    "regression": 24,
}

CLASSIFICATION_BACKBONES: tuple[str, ...] = (
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
)


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Task metadata used by the training stack."""

    phase: int
    annotation_types: tuple[str, ...]
    architectures: tuple[str, ...]
    default_loss: str
    available_losses: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    default_augmentations: AugmentationConfig
    default_hyperparameters: HyperparameterConfig


_CLASSIFICATION_DEFAULT_AUGMENTATIONS = AugmentationConfig(
    train=[
        AugmentationStep(
            name="RandomResizedCrop",
            params={"size": [224, 224], "scale": [0.8, 1.0]},
        ),
        AugmentationStep(name="RandomHorizontalFlip", params={"p": 0.5}),
        AugmentationStep(name="RandomRotation", params={"degrees": 15}),
        AugmentationStep(
            name="ColorJitter",
            params={
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05,
            },
        ),
        AugmentationStep(name="ToImage"),
        AugmentationStep(
            name="Normalize",
            params={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
    ],
    val=[
        AugmentationStep(name="Resize", params={"size": [256, 256]}),
        AugmentationStep(name="CenterCrop", params={"size": [224, 224]}),
        AugmentationStep(name="ToImage"),
        AugmentationStep(
            name="Normalize",
            params={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
    ],
)

_CLASSIFICATION_DEFAULT_HYPERPARAMETERS = HyperparameterConfig(
    optimizer="adam",
    learning_rate=0.001,
    weight_decay=0.0001,
    scheduler="cosine",
    warmup_epochs=5,
    batch_size=32,
    batch_multiplier=1,
    max_epochs=50,
    early_stopping_patience=10,
    loss="cross_entropy",
    dropout=0.2,
)

TASK_REGISTRY: dict[TaskType, TaskConfig] = {
    "classification": TaskConfig(
        phase=8,
        annotation_types=("label",),
        architectures=CLASSIFICATION_BACKBONES,
        default_loss="cross_entropy",
        available_losses=(
            "cross_entropy",
            "focal",
            "label_smoothing_cross_entropy",
        ),
        primary_metric="accuracy",
        secondary_metrics=("precision", "recall", "f1", "confusion_matrix"),
        default_augmentations=_CLASSIFICATION_DEFAULT_AUGMENTATIONS,
        default_hyperparameters=_CLASSIFICATION_DEFAULT_HYPERPARAMETERS,
    )
}

ModelFactory = Callable[[ModelConfig, int], nn.Module]


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Registered architecture metadata and model factory."""

    task: TaskType
    name: str
    display_name: str
    parameter_count_millions: float
    summary: str
    factory: ModelFactory


ARCHITECTURE_CATALOG: dict[tuple[TaskType, str], ArchitectureSpec] = {}


def list_selectable_tasks(active_phases: tuple[int, ...] | None = None) -> list[TaskType]:
    """Return tasks whose phase is active in the current build."""
    enabled_phases = set(active_phases or ACTIVE_PHASES)
    return [
        task
        for task, phase in TASK_PHASE_BY_TYPE.items()
        if phase in enabled_phases and task in TASK_REGISTRY
    ]


def is_task_selectable(task: TaskType, active_phases: tuple[int, ...] | None = None) -> bool:
    """Return whether a task is active in the current build."""
    return task in set(list_selectable_tasks(active_phases))


def get_task_config(task: TaskType) -> TaskConfig:
    """Return task config from the registry."""
    if task not in TASK_REGISTRY:
        raise ValueError(f"Task '{task}' is not implemented yet.")
    return TASK_REGISTRY[task]


def get_default_hyperparameters(task: TaskType) -> HyperparameterConfig:
    """Return a detached copy of task default hyperparameters."""
    return deepcopy(get_task_config(task).default_hyperparameters)


def get_default_augmentations(task: TaskType) -> AugmentationConfig:
    """Return a detached copy of task default augmentations."""
    return deepcopy(get_task_config(task).default_augmentations)


def build_default_training_config(
    task: TaskType,
    *,
    backbone: str | None = None,
) -> TrainingConfig:
    """Build and validate a default training config for a task."""
    task_config = get_task_config(task)
    selected_backbone = backbone or task_config.architectures[0]
    if selected_backbone not in task_config.architectures:
        raise ValueError(f"Backbone '{selected_backbone}' is not available for task '{task}'.")

    hyperparameters = get_default_hyperparameters(task)
    model = ModelConfig(
        backbone=selected_backbone,
        head="classification",
        pretrained=True,
        freeze_backbone=False,
        dropout=hyperparameters.dropout,
    )
    augmentations = get_default_augmentations(task)
    return TrainingConfig(
        model=model,
        hyperparameters=hyperparameters,
        augmentations=augmentations,
    )


def register_architecture(
    *,
    task: TaskType,
    name: str,
    display_name: str,
    parameter_count_millions: float,
    summary: str,
) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator to register model factories by (task, architecture key)."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        key = (task, name)
        ARCHITECTURE_CATALOG[key] = ArchitectureSpec(
            task=task,
            name=name,
            display_name=display_name,
            parameter_count_millions=parameter_count_millions,
            summary=summary,
            factory=factory,
        )
        return factory

    return decorator


def list_architectures(task: TaskType) -> list[str]:
    """List architecture keys available for a task."""
    return [spec.name for spec in ARCHITECTURE_CATALOG.values() if spec.task == task]


def list_architecture_specs(task: TaskType) -> list[ArchitectureSpec]:
    """List architecture metadata for a task."""
    return [spec for spec in ARCHITECTURE_CATALOG.values() if spec.task == task]


def create_model(
    task: TaskType,
    architecture: str,
    config: ModelConfig,
    num_classes: int,
) -> nn.Module:
    """Instantiate a model for the requested task and architecture."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    if config.head != "classification":
        raise ValueError(f"Unsupported model head '{config.head}'.")
    if config.backbone != architecture:
        raise ValueError("ModelConfig.backbone must match the selected architecture.")

    key = (task, architecture)
    if key not in ARCHITECTURE_CATALOG:
        raise ValueError(f"Unknown architecture '{architecture}' for task '{task}'.")
    return ARCHITECTURE_CATALOG[key].factory(config, num_classes)


def _freeze_module(module: nn.Module) -> None:
    """Disable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def _attach_classification_head(
    backbone: nn.Module,
    *,
    in_features: int,
    config: ModelConfig,
    num_classes: int,
) -> nn.Module:
    """Compose a feature extractor and classification head."""
    if config.freeze_backbone:
        _freeze_module(backbone)
    head = ClassificationHead(
        in_features=in_features,
        num_classes=num_classes,
        dropout=config.dropout,
    )
    return nn.Sequential(backbone, head)


@register_architecture(
    task="classification",
    name="resnet18",
    display_name="ResNet-18",
    parameter_count_millions=11.7,
    summary="Lightweight, fast training.",
)
def _create_resnet18(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT if config.pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="resnet34",
    display_name="ResNet-34",
    parameter_count_millions=21.8,
    summary="Balanced speed and capacity.",
)
def _create_resnet34(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = resnet34(weights=ResNet34_Weights.DEFAULT if config.pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="resnet50",
    display_name="ResNet-50",
    parameter_count_millions=25.6,
    summary="Standard baseline with strong accuracy.",
)
def _create_resnet50(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT if config.pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="efficientnet_b0",
    display_name="EfficientNet-B0",
    parameter_count_millions=5.3,
    summary="Parameter-efficient baseline.",
)
def _create_efficientnet_b0(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
    )
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="efficientnet_b3",
    display_name="EfficientNet-B3",
    parameter_count_millions=12.2,
    summary="Higher-accuracy efficient model.",
)
def _create_efficientnet_b3(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = efficientnet_b3(
        weights=EfficientNet_B3_Weights.DEFAULT if config.pretrained else None
    )
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="mobilenet_v3_small",
    display_name="MobileNetV3-Small",
    parameter_count_millions=2.5,
    summary="Edge-optimized compact model.",
)
def _create_mobilenet_v3_small(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
    )
    first_layer = backbone.classifier[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3-Small classifier structure.")
    in_features = first_layer.in_features
    backbone.classifier = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


@register_architecture(
    task="classification",
    name="mobilenet_v3_large",
    display_name="MobileNetV3-Large",
    parameter_count_millions=5.5,
    summary="Edge-friendly model with stronger capacity.",
)
def _create_mobilenet_v3_large(config: ModelConfig, num_classes: int) -> nn.Module:
    backbone = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.DEFAULT if config.pretrained else None
    )
    first_layer = backbone.classifier[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3-Large classifier structure.")
    in_features = first_layer.in_features
    backbone.classifier = nn.Identity()
    return _attach_classification_head(
        backbone,
        in_features=in_features,
        config=config,
        num_classes=num_classes,
    )


__all__ = [
    "ACTIVE_PHASES",
    "ARCHITECTURE_CATALOG",
    "CLASSIFICATION_BACKBONES",
    "TASK_PHASE_BY_TYPE",
    "TASK_REGISTRY",
    "ArchitectureSpec",
    "TaskConfig",
    "build_default_training_config",
    "create_model",
    "get_default_augmentations",
    "get_default_hyperparameters",
    "get_task_config",
    "is_task_selectable",
    "list_architecture_specs",
    "list_architectures",
    "list_selectable_tasks",
    "register_architecture",
]
