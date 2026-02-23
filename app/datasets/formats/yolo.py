from __future__ import annotations

from pathlib import Path

from app.datasets.formats.common import (
    ParsedDataset,
    ParsedImage,
    is_supported_image_file,
    read_image_size,
)


def parse_yolo_dataset(source_path: Path, *, task: str) -> ParsedDataset:
    """Parse YOLO object detection datasets."""
    if task != "object_detection":
        raise ValueError(f"YOLO import is not supported for task '{task}'.")
    if not source_path.exists() or not source_path.is_dir():
        raise ValueError("YOLO source path must be an existing directory.")

    classes_file = source_path / "classes.txt"
    if not classes_file.exists() or not classes_file.is_file():
        raise ValueError("YOLO source must include a 'classes.txt' file.")
    classes = _read_classes(classes_file)
    if not classes:
        raise ValueError("YOLO classes.txt must include at least one class.")

    images_dir = source_path / "images"
    labels_dir = source_path / "labels"
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError("YOLO source must include an 'images/' directory.")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise ValueError("YOLO source must include a 'labels/' directory.")

    image_files = sorted(
        [entry for entry in images_dir.iterdir() if is_supported_image_file(entry)],
        key=lambda path: path.name.lower(),
    )
    if not image_files:
        raise ValueError("YOLO source does not include any supported images in 'images/'.")

    parsed_images: list[ParsedImage] = []
    for image_path in image_files:
        width, height = read_image_size(image_path)
        labels_path = labels_dir / f"{image_path.stem}.txt"
        annotations = _parse_label_file(
            labels_path=labels_path,
            class_names=classes,
            image_width=width,
            image_height=height,
        )
        parsed_images.append(
            ParsedImage(
                source_path=image_path,
                source_filename=image_path.name,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    return ParsedDataset(
        source_format="yolo",
        source_path=source_path,
        task="object_detection",
        classes=classes,
        images=parsed_images,
    )


def _read_classes(path: Path) -> list[str]:
    classes: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        class_name = " ".join(raw_line.strip().split())
        if not class_name or class_name in seen:
            continue
        seen.add(class_name)
        classes.append(class_name)
    return classes


def _parse_label_file(
    *,
    labels_path: Path,
    class_names: list[str],
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    if not labels_path.exists():
        return []
    if not labels_path.is_file():
        raise ValueError(f"YOLO label path is not a file: {labels_path}")

    annotations: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(
                f"YOLO label '{labels_path.name}' line {line_number} must contain 5 fields."
            )

        class_id = _parse_class_id(parts[0], labels_path=labels_path, line_number=line_number)
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(
                f"YOLO label '{labels_path.name}' line {line_number} uses unknown "
                f"class_id={class_id}."
            )

        center_x = _parse_ratio(parts[1], labels_path=labels_path, line_number=line_number)
        center_y = _parse_ratio(parts[2], labels_path=labels_path, line_number=line_number)
        width_ratio = _parse_ratio(parts[3], labels_path=labels_path, line_number=line_number)
        height_ratio = _parse_ratio(parts[4], labels_path=labels_path, line_number=line_number)

        bbox = _denormalize_bbox(
            center_x=center_x,
            center_y=center_y,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            image_width=image_width,
            image_height=image_height,
        )

        annotations.append(
            {
                "type": "bbox",
                "class_id": class_id,
                "class_name": class_names[class_id],
                "bbox": bbox,
            }
        )
    return annotations


def _parse_class_id(raw: str, *, labels_path: Path, line_number: int) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"YOLO label '{labels_path.name}' line {line_number} has invalid class_id '{raw}'."
        ) from exc


def _parse_ratio(raw: str, *, labels_path: Path, line_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"YOLO label '{labels_path.name}' line {line_number} has invalid float value '{raw}'."
        ) from exc
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"YOLO label '{labels_path.name}' line {line_number} values must be within [0, 1]."
        )
    return value


def _denormalize_bbox(
    *,
    center_x: float,
    center_y: float,
    width_ratio: float,
    height_ratio: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    width = max(width_ratio * image_width, 1e-6)
    height = max(height_ratio * image_height, 1e-6)
    x = center_x * image_width - (width / 2.0)
    y = center_y * image_height - (height / 2.0)

    x = max(0.0, min(x, float(image_width)))
    y = max(0.0, min(y, float(image_height)))
    width = min(width, max(float(image_width) - x, 1e-6))
    height = min(height, max(float(image_height) - y, 1e-6))
    return [x, y, width, height]
