from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["cosine", "step", "multistep", "poly", "none"]
ClassificationLossType = Literal["cross_entropy", "focal", "label_smoothing_cross_entropy"]
ExperimentStatus = Literal["created", "pending", "training", "completed", "failed", "cancelled"]
PrecisionType = Literal["32", "16-mixed", "bf16-mixed"]

_EXPERIMENT_ID_PATTERN = re.compile(r"^exp-[0-9a-f]{8}$")
_EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$")
_DEVICE_PATTERN = re.compile(r"^(cpu|gpu|cuda|gpu:[0-9]+|cuda:[0-9]+)$")


def _normalize_name(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("Experiment name cannot be empty.")
    if not _EXPERIMENT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Experiment name must use only letters, numbers, spaces, underscores, and hyphens."
        )
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


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
    milestones: list[int] = Field(default_factory=list)
    gamma: float = Field(default=0.1, gt=0, lt=1)
    poly_power: float = Field(default=0.9, gt=0)
    batch_size: int = Field(default=32, ge=1, le=256)
    batch_multiplier: int = Field(default=1, ge=1, le=64)
    max_epochs: int = Field(default=50, ge=1, le=1000)
    early_stopping_patience: int = Field(default=10, ge=0, le=100)
    loss: ClassificationLossType = "cross_entropy"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)

    @field_validator("milestones")
    @classmethod
    def validate_milestones(cls, value: list[int]) -> list[int]:
        """Normalize scheduler milestones as sorted positive unique integers."""
        unique = sorted({milestone for milestone in value if milestone > 0})
        if len(unique) != len(value):
            raise ValueError("milestones must contain unique positive integers.")
        return unique


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


class HardwareConfig(BaseModel):
    """Hardware-related training settings persisted per experiment."""

    selected_devices: list[str] = Field(default_factory=lambda: ["cpu"])
    precision: PrecisionType = "32"

    @field_validator("selected_devices")
    @classmethod
    def validate_selected_devices(cls, value: list[str]) -> list[str]:
        """Normalize and validate selected device labels."""
        normalized: list[str] = []
        seen: set[str] = set()
        for device in value:
            clean = device.strip().lower()
            if not clean:
                continue
            if not _DEVICE_PATTERN.fullmatch(clean):
                raise ValueError(f"Unsupported device label '{device}'.")
            if clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        if not normalized:
            return ["cpu"]
        return normalized


class ExperimentError(BaseModel):
    """Structured error payload stored for failed experiments."""

    type: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=4000)
    traceback: str = Field(min_length=1)


class ExperimentSummary(BaseModel):
    """Experiment list item persisted in `experiments_index.json`."""

    id: str = Field(pattern=_EXPERIMENT_ID_PATTERN.pattern)
    name: str = Field(min_length=1, max_length=80)
    created_at: datetime
    status: ExperimentStatus = "created"
    best_metric_value: float | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize experiment names."""
        return _normalize_name(value)


class ExperimentsIndex(BaseModel):
    """Top-level index file payload for project experiments."""

    version: str = Field(default="1.0")
    experiments: list[ExperimentSummary] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    """Canonical experiment record stored in `experiment.json`."""

    id: str = Field(pattern=_EXPERIMENT_ID_PATTERN.pattern)
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=280)
    created_at: datetime
    status: ExperimentStatus = "created"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    split_name: str = Field(min_length=1, max_length=120)
    model: ModelConfig
    hyperparameters: HyperparameterConfig
    augmentations: AugmentationConfig
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    best_epoch: int | None = Field(default=None, ge=1)
    best_checkpoint_path: str | None = None
    best_metric: dict[str, float] | None = None
    final_metrics: dict[str, float] | None = None
    error: ExperimentError | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize experiment names."""
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        """Normalize optional experiment descriptions."""
        return _normalize_optional_text(value)

    @field_validator("split_name")
    @classmethod
    def normalize_split_name(cls, value: str) -> str:
        """Normalize split names."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("split_name cannot be empty.")
        return normalized


class ExperimentCreate(BaseModel):
    """Create payload for a new experiment."""

    name: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=280)
    split_name: str | None = Field(default=None, max_length=120)
    model: ModelConfig | None = None
    hyperparameters: HyperparameterConfig | None = None
    augmentations: AugmentationConfig | None = None
    hardware: HardwareConfig | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        """Normalize optional experiment names."""
        if value is None:
            return None
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        """Normalize optional descriptions."""
        return _normalize_optional_text(value)

    @field_validator("split_name")
    @classmethod
    def normalize_split_name(cls, value: str | None) -> str | None:
        """Normalize optional split names."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None


class ExperimentUpdate(BaseModel):
    """Patch payload for editable experiment configuration fields."""

    name: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=280)
    split_name: str | None = Field(default=None, max_length=120)
    model: ModelConfig | None = None
    hyperparameters: HyperparameterConfig | None = None
    augmentations: AugmentationConfig | None = None
    hardware: HardwareConfig | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        """Normalize optional experiment names."""
        if value is None:
            return None
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        """Normalize optional descriptions."""
        return _normalize_optional_text(value)

    @field_validator("split_name")
    @classmethod
    def normalize_split_name(cls, value: str | None) -> str | None:
        """Normalize optional split names."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None


class ExperimentMetrics(BaseModel):
    """Per-epoch metrics persisted in `metrics.json`."""

    epochs: list[dict[str, float | int]] = Field(default_factory=list)
