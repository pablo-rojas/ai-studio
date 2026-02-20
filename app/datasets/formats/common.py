from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class ParsedImage:
    """Normalized parser output for a single source image."""

    source_path: Path
    source_filename: str
    width: int
    height: int
    annotations: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ParsedDataset:
    """Normalized parser output before workspace persistence."""

    source_format: str
    source_path: Path
    task: str
    classes: list[str]
    images: list[ParsedImage]


def is_supported_image_file(path: Path) -> bool:
    """Return whether a path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def list_visible_directories(root: Path) -> list[Path]:
    """List direct child directories excluding hidden names."""
    return sorted(
        [entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")],
        key=lambda path: path.name.lower(),
    )


def list_visible_files(root: Path) -> list[Path]:
    """List direct child files excluding hidden names."""
    return sorted(
        [entry for entry in root.iterdir() if entry.is_file() and not entry.name.startswith(".")],
        key=lambda path: path.name.lower(),
    )


def read_image_size(path: Path) -> tuple[int, int]:
    """Read and return image width and height.

    Args:
        path: Image file path.

    Returns:
        A tuple of (width, height).
    """
    with Image.open(path) as image:
        width, height = image.size
    return width, height
