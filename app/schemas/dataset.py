from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TaskType = Literal[
    "classification",
    "anomaly_detection",
    "object_detection",
    "oriented_object_detection",
    "segmentation",
    "instance_segmentation",
    "regression",
]

DatasetSourceFormat = Literal["image_folders", "coco", "csv"]
SplitValue = Literal["train", "val", "test", "none"]
DatasetImageSortBy = Literal["filename", "class", "size"]
SortOrder = Literal["asc", "desc"]

_DATASET_ID_PATTERN = re.compile(r"^dataset-[0-9a-f]{8}$")
_SOURCE_PATH_PATTERN = re.compile(r"^.+$")


def _normalize_class_name(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("Class name cannot be empty.")
    if len(normalized) > 120:
        raise ValueError("Class name must be 120 characters or fewer.")
    return normalized


def _normalize_filename(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Filename cannot be empty.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("Filename must not contain path separators.")
    return normalized


def _normalize_optional_query_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


class LabelAnnotation(BaseModel):
    """Classification annotation for a single image."""

    type: Literal["label"] = "label"
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=80)

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        """Validate and normalize class names."""
        return _normalize_class_name(value)


class AnomalyAnnotation(BaseModel):
    """Anomaly annotation for a single image."""

    type: Literal["anomaly"] = "anomaly"
    is_anomalous: bool


Annotation = LabelAnnotation | AnomalyAnnotation


class ImageStats(BaseModel):
    """Aggregate statistics for imported dataset images."""

    num_images: int = Field(ge=0)
    min_width: int = Field(ge=0)
    max_width: int = Field(ge=0)
    min_height: int = Field(ge=0)
    max_height: int = Field(ge=0)
    formats: list[str] = Field(default_factory=list)

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        """Ensure file extensions are non-empty lowercase strings."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip().lower().lstrip(".")
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized


class DatasetImage(BaseModel):
    """Per-image metadata persisted in `dataset.json`."""

    filename: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    split: list[SplitValue] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Validate copied image filename."""
        return _normalize_filename(value)


class DatasetMetadata(BaseModel):
    """Unified dataset metadata representation for `dataset.json`."""

    version: str = Field(default="1.0")
    id: str = Field(pattern=_DATASET_ID_PATTERN.pattern)
    task: TaskType
    source_format: DatasetSourceFormat
    source_path: str = Field(pattern=_SOURCE_PATH_PATTERN.pattern)
    imported_at: datetime
    classes: list[str] = Field(default_factory=list)
    split_names: list[str] = Field(default_factory=list)
    image_stats: ImageStats
    images: list[DatasetImage] = Field(default_factory=list)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        """Validate class list and remove duplicates while preserving order."""
        normalized: list[str] = []
        seen: set[str] = set()
        for class_name in value:
            clean_name = _normalize_class_name(class_name)
            if clean_name in seen:
                continue
            seen.add(clean_name)
            normalized.append(clean_name)
        return normalized

    @field_validator("split_names")
    @classmethod
    def validate_split_names(cls, value: list[str]) -> list[str]:
        """Validate split names for uniqueness and emptiness."""
        cleaned: list[str] = []
        seen: set[str] = set()
        for split_name in value:
            normalized = " ".join(split_name.strip().split())
            if not normalized:
                raise ValueError("Split names cannot be empty.")
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """Normalize and validate non-empty source paths."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Source path cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_consistency(self) -> DatasetMetadata:
        """Validate consistency across image entries and aggregate stats."""
        expected_split_size = len(self.split_names)
        for image in self.images:
            if len(image.split) != expected_split_size:
                raise ValueError("Each image split list must have the same length as split_names.")

        if self.image_stats.num_images != len(self.images):
            raise ValueError("image_stats.num_images must match images length.")

        if self.images:
            widths = [image.width for image in self.images]
            heights = [image.height for image in self.images]
            if self.image_stats.min_width != min(widths):
                raise ValueError("image_stats.min_width does not match image data.")
            if self.image_stats.max_width != max(widths):
                raise ValueError("image_stats.max_width does not match image data.")
            if self.image_stats.min_height != min(heights):
                raise ValueError("image_stats.min_height does not match image data.")
            if self.image_stats.max_height != max(heights):
                raise ValueError("image_stats.max_height does not match image data.")
        return self


class DatasetImportRequest(BaseModel):
    """Input schema for local dataset imports."""

    source_path: str = Field(min_length=1)
    source_format: DatasetSourceFormat | None = None

    @field_validator("source_format", mode="before")
    @classmethod
    def normalize_source_format(cls, value: object) -> object:
        """Treat blank source format values as auto-detect."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """Normalize non-empty source paths."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Source path cannot be empty.")
        return normalized


class DatasetImageListQuery(BaseModel):
    """Query parameters for browsing dataset images."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    sort_by: DatasetImageSortBy = "filename"
    sort_order: SortOrder = "asc"
    filter_class: str | None = None
    search: str | None = None

    @field_validator("filter_class", "search")
    @classmethod
    def normalize_query_text(cls, value: str | None) -> str | None:
        """Normalize optional text query parameters."""
        return _normalize_optional_query_text(value)


class DatasetImageListItem(BaseModel):
    """Single image entry in paginated dataset browsing responses."""

    filename: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    class_name: str | None = None
    split: list[SplitValue] = Field(default_factory=list)
    annotation_count: int = Field(default=0, ge=0)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Validate copied image filename."""
        return _normalize_filename(value)

    @field_validator("class_name")
    @classmethod
    def normalize_class_name(cls, value: str | None) -> str | None:
        """Normalize optional class name values."""
        if value is None:
            return None
        return _normalize_class_name(value)


class DatasetImageListResponse(BaseModel):
    """Paginated image browse response payload."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    items: list[DatasetImageListItem] = Field(default_factory=list)
