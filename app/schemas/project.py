from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal[
    "classification",
    "anomaly_detection",
    "object_detection",
    "oriented_object_detection",
    "segmentation",
    "instance_segmentation",
    "regression",
]

_PROJECT_ID_PATTERN = re.compile(r"^proj-[0-9a-f]{8}$")
_PROJECT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$")
_OPTIONAL_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9 _-]*$")


def _normalize_name(value: str) -> str:
    compact = " ".join(value.strip().split())
    if not compact:
        raise ValueError("Project name cannot be empty.")
    if not _PROJECT_NAME_PATTERN.fullmatch(compact):
        raise ValueError(
            "Project name must use only letters, numbers, spaces, underscores, and hyphens."
        )
    return compact


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    if not normalized:
        return None
    if not _OPTIONAL_TEXT_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Description must use only letters, numbers, spaces, underscores, and hyphens."
        )
    return normalized


class ProjectCreate(BaseModel):
    """Input schema for project creation."""

    name: str = Field(min_length=1, max_length=80)
    task: TaskType
    description: str | None = Field(default=None, max_length=280)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate project name formatting."""
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        """Validate optional project description formatting."""
        return _normalize_optional_text(value)


class ProjectRename(BaseModel):
    """Input schema for renaming projects."""

    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate project name formatting."""
        return _normalize_name(value)


class ProjectResponse(BaseModel):
    """Serialized project metadata for API responses and persistence."""

    id: str = Field(pattern=_PROJECT_ID_PATTERN.pattern)
    name: str = Field(min_length=1, max_length=80)
    task: TaskType
    description: str | None = Field(default=None, max_length=280)
    created_at: datetime
    updated_at: datetime

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate project name formatting."""
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        """Validate optional project description formatting."""
        return _normalize_optional_text(value)


class WorkspaceMetadata(BaseModel):
    """Serialized `workspace.json` metadata."""

    version: str = Field(default="1.0")
    created_at: datetime
    projects: list[str] = Field(default_factory=list)

    @field_validator("projects")
    @classmethod
    def validate_projects(cls, value: list[str]) -> list[str]:
        """Validate project ID list integrity."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for project_id in value:
            if not _PROJECT_ID_PATTERN.fullmatch(project_id):
                raise ValueError("Workspace contains an invalid project ID.")
            if project_id in seen:
                continue
            seen.add(project_id)
            cleaned.append(project_id)
        return cleaned
