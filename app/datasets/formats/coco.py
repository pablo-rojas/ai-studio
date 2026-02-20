from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.datasets.formats.common import ParsedDataset, ParsedImage, read_image_size


def parse_coco_dataset(
    source_path: Path,
    *,
    task: str,
    annotation_filename: str | None = None,
) -> ParsedDataset:
    """Parse COCO JSON datasets.

    Args:
        source_path: Dataset source root.
        task: Project task type.
        annotation_filename: Optional annotations filename override.

    Returns:
        ParsedDataset ready for import persistence.

    Raises:
        ValueError: If structure, schema, or content is invalid.
    """
    if task != "classification":
        raise ValueError("Phase 2 COCO import currently supports classification only.")

    if not source_path.exists() or not source_path.is_dir():
        raise ValueError("COCO source path must be an existing directory.")

    annotations_path = _resolve_annotations_file(
        source_path, annotation_filename=annotation_filename
    )
    payload = _read_coco_json(annotations_path)
    return _parse_classification_payload(payload=payload, source_path=source_path)


def _parse_classification_payload(*, payload: dict[str, Any], source_path: Path) -> ParsedDataset:
    images_payload = payload.get("images")
    annotations_payload = payload.get("annotations")
    categories_payload = payload.get("categories")
    if not isinstance(images_payload, list):
        raise ValueError("COCO JSON must include an 'images' list.")
    if not isinstance(annotations_payload, list):
        raise ValueError("COCO JSON must include an 'annotations' list.")
    if not isinstance(categories_payload, list):
        raise ValueError("COCO JSON must include a 'categories' list.")
    if not images_payload:
        raise ValueError("COCO JSON does not include any images.")
    if not categories_payload:
        raise ValueError("COCO JSON does not include any categories.")

    category_names_by_id: dict[int, str] = {}
    for category in categories_payload:
        if not isinstance(category, dict):
            raise ValueError("COCO categories entries must be objects.")
        raw_id = category.get("id")
        raw_name = category.get("name")
        if not isinstance(raw_id, int) or not isinstance(raw_name, str):
            raise ValueError("COCO categories must include integer id and string name.")
        name = raw_name.strip()
        if not name:
            raise ValueError("COCO categories must include a non-empty name.")
        category_names_by_id[raw_id] = name

    sorted_category_ids = sorted(category_names_by_id)
    classes = [category_names_by_id[category_id] for category_id in sorted_category_ids]
    category_id_to_class_id = {
        category_id: class_id for class_id, category_id in enumerate(sorted_category_ids)
    }

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations_payload:
        if not isinstance(annotation, dict):
            raise ValueError("COCO annotations entries must be objects.")
        if annotation.get("iscrowd") == 1:
            continue
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if not isinstance(image_id, int) or not isinstance(category_id, int):
            raise ValueError("COCO annotation entries must include image_id and category_id.")
        if category_id not in category_names_by_id:
            raise ValueError("COCO annotation category_id does not exist in categories.")
        annotations_by_image_id[image_id].append(annotation)

    parsed_images: list[ParsedImage] = []
    for image in sorted(images_payload, key=lambda item: int(item.get("id", 0))):
        if not isinstance(image, dict):
            raise ValueError("COCO images entries must be objects.")
        image_id = image.get("id")
        file_name = image.get("file_name")
        if not isinstance(image_id, int) or not isinstance(file_name, str):
            raise ValueError("COCO image entries must include id and file_name.")
        image_annotations = annotations_by_image_id.get(image_id, [])
        if len(image_annotations) != 1:
            raise ValueError(
                "Classification COCO import requires exactly one annotation per image."
            )

        annotation = image_annotations[0]
        category_id = int(annotation["category_id"])
        class_id = category_id_to_class_id[category_id]
        class_name = category_names_by_id[category_id]

        image_path = _resolve_coco_image_path(source_path, file_name=file_name)
        width, height = read_image_size(image_path)
        parsed_images.append(
            ParsedImage(
                source_path=image_path,
                source_filename=Path(file_name).name,
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

    return ParsedDataset(
        source_format="coco",
        source_path=source_path,
        task="classification",
        classes=classes,
        images=parsed_images,
    )


def _resolve_annotations_file(source_path: Path, *, annotation_filename: str | None) -> Path:
    if annotation_filename:
        candidate = Path(annotation_filename)
        annotation_path = candidate if candidate.is_absolute() else source_path / candidate
        if annotation_path.exists() and annotation_path.is_file():
            return annotation_path
        raise ValueError(f"COCO annotation file not found: {annotation_path}")

    default_annotation = source_path / "annotations.json"
    if default_annotation.exists() and default_annotation.is_file():
        return default_annotation

    json_files = sorted(source_path.glob("*.json"))
    if not json_files:
        raise ValueError("Could not locate a COCO annotation JSON file in the source folder.")
    if len(json_files) > 1:
        raise ValueError("Multiple JSON files found; specify annotation_filename explicitly.")
    return json_files[0]


def _read_coco_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("COCO annotation file must contain a JSON object.")
    return payload


def _resolve_coco_image_path(source_path: Path, *, file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        if candidate.exists() and candidate.is_file():
            return candidate
        raise ValueError(f"COCO image not found: {candidate}")

    search_paths = [source_path / candidate, source_path / "images" / candidate]
    for path in search_paths:
        if path.exists() and path.is_file():
            return path
    raise ValueError(f"COCO image not found for file_name '{file_name}'.")
