from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ExportFormat = Literal["onnx"]
ExportStatus = Literal["pending", "running", "completed", "failed"]

_EXPORT_ID_PATTERN = re.compile(r"^export-[0-9a-f]{8}$")
_EXPERIMENT_ID_PATTERN = re.compile(r"^exp-[0-9a-f]{8}$")
_CHECKPOINT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


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


class OnnxExportOptions(BaseModel):
    """ONNX export options persisted in `export.json`."""

    opset_version: int = Field(default=17, ge=13, le=21)
    input_shape: list[int] = Field(
        default_factory=lambda: [1, 3, 224, 224],
        min_length=4,
        max_length=4,
    )
    dynamic_axes: dict[str, dict[str, str]] | None = Field(
        default_factory=lambda: {
            "input": {"0": "batch_size"},
            "output": {"0": "batch_size"},
        }
    )
    simplify: bool = True

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, value: list[int]) -> list[int]:
        """Validate NCHW input shape dimensions."""
        cleaned = [int(dimension) for dimension in value]
        if any(dimension <= 0 for dimension in cleaned):
            raise ValueError("input_shape dimensions must be positive integers.")
        return cleaned

    @field_validator("dynamic_axes")
    @classmethod
    def validate_dynamic_axes(
        cls,
        value: dict[str, dict[str, str]] | None,
    ) -> dict[str, dict[str, str]] | None:
        """Normalize dynamic-axis entries to JSON-safe string indices."""
        if value is None:
            return None

        cleaned: dict[str, dict[str, str]] = {}
        for tensor_name, axes in value.items():
            normalized_name = tensor_name.strip()
            if not normalized_name:
                continue
            cleaned_axes: dict[str, str] = {}
            for axis_key, axis_label in axes.items():
                axis_index = int(str(axis_key))
                if axis_index < 0:
                    raise ValueError("dynamic axis indices must be >= 0.")
                normalized_label = " ".join(str(axis_label).strip().split())
                if not normalized_label:
                    raise ValueError("dynamic axis labels cannot be empty.")
                cleaned_axes[str(axis_index)] = normalized_label
            if cleaned_axes:
                cleaned[normalized_name] = cleaned_axes
        return cleaned or None

    def torch_dynamic_axes(self) -> dict[str, dict[int, str]] | None:
        """Return dynamic-axis mapping compatible with `torch.onnx.export`."""
        if self.dynamic_axes is None:
            return None

        converted: dict[str, dict[int, str]] = {}
        for tensor_name, axes in self.dynamic_axes.items():
            converted[tensor_name] = {int(axis): label for axis, label in axes.items()}
        return converted


class ExportValidationResult(BaseModel):
    """Numerical validation summary between PyTorch and exported ONNX output."""

    passed: bool
    max_diff: float = Field(ge=0.0)
    mean_diff: float = Field(default=0.0, ge=0.0)


class ExportCreate(BaseModel):
    """Create payload for one export run."""

    experiment_id: str = Field(pattern=_EXPERIMENT_ID_PATTERN.pattern)
    checkpoint: str = Field(default="best", min_length=1, max_length=120)
    format: ExportFormat = "onnx"
    options: OnnxExportOptions = Field(default_factory=OnnxExportOptions)

    @field_validator("checkpoint")
    @classmethod
    def normalize_checkpoint(cls, value: str) -> str:
        """Normalize checkpoint selectors to checkpoint stems."""
        return _normalize_checkpoint(value)


class ExportRecord(ExportCreate):
    """Canonical export metadata persisted in `export.json`."""

    id: str = Field(pattern=_EXPORT_ID_PATTERN.pattern)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: ExportStatus = "pending"
    output_file: str | None = Field(default=None, max_length=260)
    output_size_mb: float | None = Field(default=None, ge=0.0)
    validation: ExportValidationResult | None = None
    error: str | None = Field(default=None, max_length=4000)

    @field_validator("output_file")
    @classmethod
    def normalize_output_file(cls, value: str | None) -> str | None:
        """Ensure output filenames are basename-only values."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "/" in normalized or "\\" in normalized:
            raise ValueError("output_file must not contain path separators.")
        return normalized

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        """Normalize optional error text."""
        return _normalize_optional_text(value)


class ExportSummary(BaseModel):
    """Export list item persisted in `exports_index.json`."""

    id: str = Field(pattern=_EXPORT_ID_PATTERN.pattern)
    experiment_id: str = Field(pattern=_EXPERIMENT_ID_PATTERN.pattern)
    checkpoint: str = Field(min_length=1, max_length=120)
    format: ExportFormat
    status: ExportStatus
    created_at: datetime
    output_size_mb: float | None = Field(default=None, ge=0.0)

    @field_validator("checkpoint")
    @classmethod
    def normalize_checkpoint(cls, value: str) -> str:
        """Normalize checkpoint selectors to checkpoint stems."""
        return _normalize_checkpoint(value)


class ExportsIndex(BaseModel):
    """Top-level index payload for project exports."""

    version: str = Field(default="1.0")
    exports: list[ExportSummary] = Field(default_factory=list)


class ExportFormatInfo(BaseModel):
    """Metadata returned when listing supported export formats."""

    name: str = Field(min_length=1, max_length=40)
    display_name: str = Field(min_length=1, max_length=80)
    available: bool
