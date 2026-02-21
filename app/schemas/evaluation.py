from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

EvaluationSubset = Literal["train", "val", "test"]
EvaluationStatus = Literal["pending", "running", "completed", "failed"]
EvaluationResultSortBy = Literal["filename", "confidence", "error"]
SortOrder = Literal["asc", "desc"]

_CHECKPOINT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_DEVICE_PATTERN = re.compile(r"^(cpu|gpu|cuda|gpu:[0-9]+|cuda:[0-9]+)$")


def _normalize_checkpoint(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".ckpt"):
        normalized = normalized[:-5]
    if not normalized:
        raise ValueError("checkpoint cannot be empty.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("checkpoint must not contain path separators.")
    if not _CHECKPOINT_PATTERN.fullmatch(normalized):
        raise ValueError(
            "checkpoint must use only letters, numbers, periods, underscores, and hyphens."
        )
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


class EvaluationProgress(BaseModel):
    """Progress counters persisted while evaluation is running."""

    processed: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_progress_bounds(self) -> EvaluationProgress:
        """Ensure processed does not exceed total."""
        if self.processed > self.total:
            raise ValueError("progress.processed cannot exceed progress.total.")
        return self


class EvaluationConfig(BaseModel):
    """User-provided configuration for an evaluation run."""

    checkpoint: str = Field(default="best", min_length=1, max_length=120)
    split_subsets: list[EvaluationSubset] = Field(default_factory=lambda: ["test"])
    batch_size: int = Field(default=32, ge=1, le=1024)
    device: str = Field(default="cpu", min_length=1, max_length=32)

    @field_validator("checkpoint")
    @classmethod
    def normalize_checkpoint(cls, value: str) -> str:
        """Normalize checkpoint selectors to checkpoint stems."""
        return _normalize_checkpoint(value)

    @field_validator("split_subsets")
    @classmethod
    def normalize_split_subsets(cls, value: list[EvaluationSubset]) -> list[EvaluationSubset]:
        """Deduplicate subsets while preserving order."""
        cleaned: list[EvaluationSubset] = []
        seen: set[EvaluationSubset] = set()
        for subset in value:
            if subset in seen:
                continue
            seen.add(subset)
            cleaned.append(subset)
        if not cleaned:
            raise ValueError("split_subsets must include at least one subset.")
        return cleaned

    @field_validator("device")
    @classmethod
    def normalize_device(cls, value: str) -> str:
        """Normalize and validate device selectors."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("device cannot be empty.")
        if not _DEVICE_PATTERN.fullmatch(normalized):
            raise ValueError(f"Unsupported device label '{value}'.")
        return normalized


class EvaluationRecord(EvaluationConfig):
    """Canonical evaluation metadata stored in `evaluation.json`."""

    status: EvaluationStatus = "pending"
    progress: EvaluationProgress = Field(default_factory=EvaluationProgress)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        """Normalize optional error text."""
        return _normalize_optional_text(value)


class ClassificationLabelRef(BaseModel):
    """Class label reference used in per-image results."""

    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=120)

    @field_validator("class_name")
    @classmethod
    def normalize_class_name(cls, value: str) -> str:
        """Normalize class names."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("class_name cannot be empty.")
        return normalized


class ClassificationPrediction(ClassificationLabelRef):
    """Predicted class payload for one evaluated image."""

    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationPerImageResult(BaseModel):
    """Per-image evaluation payload for classification tasks."""

    filename: str = Field(min_length=1)
    subset: EvaluationSubset
    ground_truth: ClassificationLabelRef
    prediction: ClassificationPrediction
    correct: bool
    probabilities: dict[str, float] = Field(default_factory=dict)

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str) -> str:
        """Ensure filenames are basename-only values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("filename cannot be empty.")
        if "/" in normalized or "\\" in normalized:
            raise ValueError("filename must not contain path separators.")
        return normalized

    @field_validator("probabilities")
    @classmethod
    def normalize_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        """Validate class probability map values."""
        cleaned: dict[str, float] = {}
        for key, raw_value in value.items():
            class_name = " ".join(key.strip().split())
            if not class_name:
                continue
            probability = float(raw_value)
            if probability < 0.0 or probability > 1.0:
                raise ValueError("probabilities must contain values between 0 and 1.")
            cleaned[class_name] = probability
        return cleaned


class ClassificationPerClassAggregate(BaseModel):
    """Per-class aggregate metrics for classification evaluation."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)


class ClassificationAggregateMetrics(BaseModel):
    """Aggregate metrics persisted in `aggregate.json` for classification."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision_macro: float = Field(ge=0.0, le=1.0)
    recall_macro: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    per_class: dict[str, ClassificationPerClassAggregate] = Field(default_factory=dict)

    @field_validator("confusion_matrix")
    @classmethod
    def validate_confusion_matrix(cls, value: list[list[int]]) -> list[list[int]]:
        """Validate confusion matrix shape and non-negative counts."""
        size = len(value)
        for row in value:
            if len(row) != size:
                raise ValueError("confusion_matrix must be square.")
            for cell in row:
                if cell < 0:
                    raise ValueError("confusion_matrix cannot contain negative counts.")
        return value


class EvaluationResultsFile(BaseModel):
    """Top-level results payload persisted in `results.json`."""

    results: list[ClassificationPerImageResult] = Field(default_factory=list)


class EvaluationResultsQuery(BaseModel):
    """Query parameters for paginated evaluation results."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    sort_by: EvaluationResultSortBy = "filename"
    sort_order: SortOrder = "asc"
    filter_correct: bool | None = None
    filter_class: str | None = None
    filter_subset: EvaluationSubset | None = None

    @field_validator("filter_class")
    @classmethod
    def normalize_filter_class(cls, value: str | None) -> str | None:
        """Normalize optional class filters."""
        return _normalize_optional_text(value)


class EvaluationResultsPage(BaseModel):
    """Paginated evaluation results response payload."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    items: list[ClassificationPerImageResult] = Field(default_factory=list)
