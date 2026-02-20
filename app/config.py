from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ENV_WORKSPACE_ROOT: Final[str] = "AI_STUDIO_WORKSPACE_ROOT"
ENV_DEFAULT_DEVICE: Final[str] = "AI_STUDIO_DEFAULT_DEVICE"
ENV_MAX_UPLOAD_SIZE: Final[str] = "AI_STUDIO_MAX_UPLOAD_SIZE"

DEFAULT_MAX_UPLOAD_SIZE: Final[int] = 200 * 1024 * 1024
DEFAULT_WORKSPACE_DIRNAME: Final[str] = "workspace"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for the application."""

    workspace_root: Path
    default_device: str
    max_upload_size: int


def _detect_default_device() -> str:
    """Choose a default training device based on local hardware."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_positive_int(raw_value: str | None, *, fallback: int) -> int:
    """Parse a positive integer environment value with fallback."""
    if not raw_value:
        return fallback

    try:
        parsed = int(raw_value)
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def get_settings() -> Settings:
    """Build settings from environment variables and local defaults."""
    project_root = Path(__file__).resolve().parent.parent
    workspace_override = os.getenv(ENV_WORKSPACE_ROOT)
    workspace_root = (
        Path(workspace_override).expanduser().resolve()
        if workspace_override
        else (project_root / DEFAULT_WORKSPACE_DIRNAME).resolve()
    )

    default_device = (
        os.getenv(ENV_DEFAULT_DEVICE, _detect_default_device()).strip().lower() or "cpu"
    )
    max_upload_size = _parse_positive_int(
        os.getenv(ENV_MAX_UPLOAD_SIZE), fallback=DEFAULT_MAX_UPLOAD_SIZE
    )

    return Settings(
        workspace_root=workspace_root,
        default_device=default_device,
        max_upload_size=max_upload_size,
    )
