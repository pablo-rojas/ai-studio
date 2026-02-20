from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_SPLIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$")


def _normalize_split_name(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("Split name cannot be empty.")
    if not _SPLIT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Split name must use only letters, numbers, spaces, underscores, and hyphens."
        )
    return normalized


class SplitRatios(BaseModel):
    """Normalized train/val/test ratios."""

    train: float = Field(ge=0.0, le=1.0)
    val: float = Field(ge=0.0, le=1.0)
    test: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> SplitRatios:
        """Ensure ratios sum to 1.0."""
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0.")
        return self


class SplitPreviewRequest(BaseModel):
    """Input schema for split preview."""

    ratios: SplitRatios
    seed: int | None = Field(default=None, ge=0)


class SplitCreateRequest(SplitPreviewRequest):
    """Input schema for persisted split creation."""

    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Normalize and validate split names."""
        return _normalize_split_name(value)


class SplitCounts(BaseModel):
    """Subset counts for a single split."""

    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)
    none: int = Field(default=0, ge=0)


class SplitPreviewResponse(BaseModel):
    """Preview data returned before persisting a split."""

    ratios: SplitRatios
    seed: int
    stats: SplitCounts
    class_distribution: dict[str, SplitCounts] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SplitSummary(BaseModel):
    """Summary of a persisted split."""

    name: str
    index: int = Field(ge=0)
    immutable: bool = True
    stats: SplitCounts
    class_distribution: dict[str, SplitCounts] = Field(default_factory=dict)
