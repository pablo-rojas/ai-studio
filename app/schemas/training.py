from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["cosine", "step", "multistep", "poly", "none"]
ClassificationLossType = Literal["cross_entropy", "focal", "label_smoothing_cross_entropy"]


class ModelConfig(BaseModel):
    """Model-selection fields used to build a training model."""

    backbone: str = Field(min_length=1, max_length=80)
    head: Literal["classification"] = "classification"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)

    @field_validator("backbone")
    @classmethod
    def normalize_backbone(cls, value: str) -> str:
        """Normalize architecture keys to lowercase."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Backbone cannot be empty.")
        return normalized


class HyperparameterConfig(BaseModel):
    """Validated hyperparameter set for a training run."""

    optimizer: OptimizerType = "adam"
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    momentum: float = Field(default=0.9, ge=0, le=1)
    scheduler: SchedulerType = "cosine"
    warmup_epochs: int = Field(default=5, ge=0, le=1000)
    step_size: int = Field(default=10, ge=1, le=1000)
    gamma: float = Field(default=0.1, gt=0, lt=1)
    poly_power: float = Field(default=0.9, gt=0)
    batch_size: int = Field(default=32, ge=1, le=256)
    batch_multiplier: int = Field(default=1, ge=1, le=64)
    max_epochs: int = Field(default=50, ge=1, le=1000)
    early_stopping_patience: int = Field(default=10, ge=0, le=100)
    loss: ClassificationLossType = "cross_entropy"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)


class AugmentationStep(BaseModel):
    """Single transform step used in augmentation pipelines."""

    name: str = Field(min_length=1, max_length=80)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize transform names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Augmentation transform name cannot be empty.")
        return normalized


class AugmentationConfig(BaseModel):
    """Training and validation augmentation pipelines."""

    train: list[AugmentationStep] = Field(default_factory=list)
    val: list[AugmentationStep] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Combined model + hyperparameter + augmentation training config."""

    model: ModelConfig
    hyperparameters: HyperparameterConfig
    augmentations: AugmentationConfig
