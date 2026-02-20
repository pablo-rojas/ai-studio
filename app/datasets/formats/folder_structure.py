from __future__ import annotations

from pathlib import Path

from app.datasets.formats.common import (
    ParsedDataset,
    ParsedImage,
    is_supported_image_file,
    list_visible_directories,
    list_visible_files,
    read_image_size,
)

_NORMAL_ALIASES = {"good", "ok", "normal", "pass"}
_ANOMALOUS_ALIASES = {
    "anomaly",
    "anomalous",
    "bad",
    "nok",
    "defect",
    "defective",
    "ng",
    "fail",
}


def parse_image_folders(source_path: Path, *, task: str) -> ParsedDataset:
    """Parse folder-structured datasets.

    Args:
        source_path: Dataset source root.
        task: Project task type.

    Returns:
        ParsedDataset ready for import persistence.

    Raises:
        ValueError: If structure or content is invalid.
    """
    if not source_path.exists() or not source_path.is_dir():
        raise ValueError("Image folder source path must be an existing directory.")

    class_dirs = list_visible_directories(source_path)
    if not class_dirs:
        raise ValueError("Image folder source must contain class subdirectories.")

    if task == "classification":
        return _parse_classification_folders(source_path, class_dirs)
    if task == "anomaly_detection":
        return _parse_anomaly_folders(source_path, class_dirs)
    raise ValueError(f"Image folder import is not supported for task '{task}'.")


def _parse_classification_folders(source_path: Path, class_dirs: list[Path]) -> ParsedDataset:
    classes = [directory.name for directory in class_dirs]
    class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}

    images: list[ParsedImage] = []
    for class_dir in class_dirs:
        _validate_no_nested_directories(class_dir)
        class_name = class_dir.name
        class_id = class_to_id[class_name]
        for file_path in list_visible_files(class_dir):
            if not is_supported_image_file(file_path):
                continue
            width, height = read_image_size(file_path)
            images.append(
                ParsedImage(
                    source_path=file_path,
                    source_filename=file_path.name,
                    width=width,
                    height=height,
                    annotations=[
                        {
                            "type": "label",
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    ],
                )
            )

    if not images:
        raise ValueError("No supported images were found in class subdirectories.")

    return ParsedDataset(
        source_format="image_folders",
        source_path=source_path,
        task="classification",
        classes=classes,
        images=images,
    )


def _parse_anomaly_folders(source_path: Path, class_dirs: list[Path]) -> ParsedDataset:
    if len(class_dirs) != 2:
        raise ValueError(
            "Anomaly detection folder import requires exactly two class folders:"
            " one normal and one anomalous."
        )

    normal_dir: Path | None = None
    anomalous_dir: Path | None = None
    for directory in class_dirs:
        semantic_name = directory.name.strip().lower()
        if semantic_name in _NORMAL_ALIASES:
            normal_dir = directory
            continue
        if semantic_name in _ANOMALOUS_ALIASES:
            anomalous_dir = directory
            continue
        raise ValueError(
            "Anomaly folder names must map to supported aliases for normal or anomalous classes."
        )

    if normal_dir is None or anomalous_dir is None:
        raise ValueError(
            "Anomaly detection import requires one normal folder and one anomalous folder."
        )

    images: list[ParsedImage] = []
    for folder, is_anomalous in ((normal_dir, False), (anomalous_dir, True)):
        _validate_no_nested_directories(folder)
        for file_path in list_visible_files(folder):
            if not is_supported_image_file(file_path):
                continue
            width, height = read_image_size(file_path)
            images.append(
                ParsedImage(
                    source_path=file_path,
                    source_filename=file_path.name,
                    width=width,
                    height=height,
                    annotations=[{"type": "anomaly", "is_anomalous": is_anomalous}],
                )
            )

    if not images:
        raise ValueError("No supported images were found in anomaly folders.")

    return ParsedDataset(
        source_format="image_folders",
        source_path=source_path,
        task="anomaly_detection",
        classes=["normal", "anomalous"],
        images=images,
    )


def _validate_no_nested_directories(class_dir: Path) -> None:
    nested_dirs = [
        entry for entry in class_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    ]
    if nested_dirs:
        raise ValueError(
            f"Nested subdirectories are not supported in folder import: {class_dir.name}."
        )
