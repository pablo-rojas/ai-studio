from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.datasets.formats.common import ParsedDataset, ParsedImage, read_image_size


@dataclass(frozen=True, slots=True)
class _RawCsvRow:
    filename: str
    image_path: Path
    width: int
    height: int
    label: str | None = None
    is_anomalous: bool | None = None


def parse_csv_dataset(
    source_path: Path,
    *,
    task: str,
    csv_filename: str | None = None,
) -> ParsedDataset:
    """Parse CSV-based datasets.

    Args:
        source_path: Dataset source root.
        task: Project task type.
        csv_filename: Optional CSV filename override.

    Returns:
        ParsedDataset ready for import persistence.

    Raises:
        ValueError: If structure, schema, or content is invalid.
    """
    if task not in {"classification", "anomaly_detection"}:
        raise ValueError(f"CSV import is not supported for task '{task}'.")

    if not source_path.exists() or not source_path.is_dir():
        raise ValueError("CSV source path must be an existing directory.")

    csv_path = _resolve_csv_file(source_path, csv_filename=csv_filename)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file must include a header row.")

        field_map = {field.strip().lower(): field for field in reader.fieldnames}
        filename_field = field_map.get("filename")
        if filename_field is None:
            raise ValueError("CSV file must include a 'filename' column.")

        rows = list(reader)
        if not rows:
            raise ValueError("CSV file does not contain any data rows.")

        if task == "classification":
            label_field = field_map.get("label")
            if label_field is None:
                raise ValueError("Classification CSV must include a 'label' column.")
            parsed = _parse_classification_rows(
                rows=rows,
                source_root=source_path,
                filename_field=filename_field,
                label_field=label_field,
            )
        else:
            anomaly_field = field_map.get("is_anomalous")
            if anomaly_field is None:
                raise ValueError("Anomaly CSV must include an 'is_anomalous' column.")
            parsed = _parse_anomaly_rows(
                rows=rows,
                source_root=source_path,
                filename_field=filename_field,
                anomaly_field=anomaly_field,
            )

    return ParsedDataset(
        source_format="csv",
        source_path=source_path,
        task=task,
        classes=parsed["classes"],
        images=parsed["images"],
    )


def _parse_classification_rows(
    *,
    rows: list[dict[str, str]],
    source_root: Path,
    filename_field: str,
    label_field: str,
) -> dict[str, list[str] | list[ParsedImage]]:
    raw_rows: list[_RawCsvRow] = []
    class_names: list[str] = []
    seen_classes: set[str] = set()

    for row_index, row in enumerate(rows, start=2):
        raw_filename = (row.get(filename_field) or "").strip()
        raw_label = (row.get(label_field) or "").strip()
        if not raw_filename:
            raise ValueError(f"CSV row {row_index} has an empty filename.")
        if not raw_label:
            raise ValueError(f"CSV row {row_index} has an empty label.")

        image_path = _resolve_image_path(source_root, raw_filename)
        width, height = read_image_size(image_path)
        raw_rows.append(
            _RawCsvRow(
                filename=Path(raw_filename).name,
                image_path=image_path,
                width=width,
                height=height,
                label=raw_label,
            )
        )

        if raw_label not in seen_classes:
            seen_classes.add(raw_label)
            class_names.append(raw_label)

    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}
    images = [
        ParsedImage(
            source_path=row.image_path,
            source_filename=row.filename,
            width=row.width,
            height=row.height,
            annotations=[
                {
                    "type": "label",
                    "class_id": class_to_id[row.label or ""],
                    "class_name": row.label,
                }
            ],
        )
        for row in raw_rows
    ]
    return {"classes": class_names, "images": images}


def _parse_anomaly_rows(
    *,
    rows: list[dict[str, str]],
    source_root: Path,
    filename_field: str,
    anomaly_field: str,
) -> dict[str, list[str] | list[ParsedImage]]:
    raw_rows: list[_RawCsvRow] = []
    for row_index, row in enumerate(rows, start=2):
        raw_filename = (row.get(filename_field) or "").strip()
        raw_is_anomalous = (row.get(anomaly_field) or "").strip()
        if not raw_filename:
            raise ValueError(f"CSV row {row_index} has an empty filename.")
        if not raw_is_anomalous:
            raise ValueError(f"CSV row {row_index} has an empty is_anomalous value.")

        image_path = _resolve_image_path(source_root, raw_filename)
        width, height = read_image_size(image_path)
        raw_rows.append(
            _RawCsvRow(
                filename=Path(raw_filename).name,
                image_path=image_path,
                width=width,
                height=height,
                is_anomalous=_parse_bool(raw_is_anomalous, row_index=row_index),
            )
        )

    images = [
        ParsedImage(
            source_path=row.image_path,
            source_filename=row.filename,
            width=row.width,
            height=row.height,
            annotations=[{"type": "anomaly", "is_anomalous": bool(row.is_anomalous)}],
        )
        for row in raw_rows
    ]
    return {"classes": ["normal", "anomalous"], "images": images}


def _parse_bool(raw_value: str, *, row_index: int) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"CSV row {row_index} has invalid boolean value '{raw_value}'.")


def _resolve_csv_file(source_path: Path, *, csv_filename: str | None) -> Path:
    if csv_filename:
        candidate = Path(csv_filename)
        csv_path = candidate if candidate.is_absolute() else source_path / candidate
        if csv_path.exists() and csv_path.is_file():
            return csv_path
        raise ValueError(f"CSV file not found: {csv_path}")

    labels_csv = source_path / "labels.csv"
    if labels_csv.exists() and labels_csv.is_file():
        return labels_csv

    candidates = sorted(source_path.glob("*.csv"))
    if not candidates:
        raise ValueError("Could not locate a CSV file in the source folder.")
    if len(candidates) > 1:
        raise ValueError("Multiple CSV files found; specify csv_filename explicitly.")
    return candidates[0]


def _resolve_image_path(source_root: Path, raw_filename: str) -> Path:
    candidate = Path(raw_filename)
    if candidate.is_absolute():
        if candidate.exists() and candidate.is_file():
            return candidate
        raise ValueError(f"Image not found: {candidate}")

    search_paths = [source_root / candidate, source_root / "images" / candidate]
    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    raise ValueError(f"Image not found for CSV entry: {raw_filename}")
