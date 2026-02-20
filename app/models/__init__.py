"""Model registry and task-specific model builders."""

from app.models.catalog import (
    ACTIVE_PHASES,
    CLASSIFICATION_BACKBONES,
    TASK_REGISTRY,
    build_default_training_config,
    create_model,
    get_default_augmentations,
    get_default_hyperparameters,
    get_task_config,
    is_task_selectable,
    list_architectures,
    list_selectable_tasks,
)

__all__ = [
    "ACTIVE_PHASES",
    "CLASSIFICATION_BACKBONES",
    "TASK_REGISTRY",
    "build_default_training_config",
    "create_model",
    "get_default_augmentations",
    "get_default_hyperparameters",
    "get_task_config",
    "is_task_selectable",
    "list_architectures",
    "list_selectable_tasks",
]
