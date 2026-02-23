from __future__ import annotations

import logging
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import NotFoundError, ValidationError
from app.core.project_service import ProjectService
from app.datasets.formats import (
    ParsedDataset,
    parse_coco_dataset,
    parse_csv_dataset,
    parse_image_folders,
    parse_yolo_dataset,
)
from app.schemas.dataset import (
    DatasetImage,
    DatasetImageListItem,
    DatasetImageListQuery,
    DatasetImageListResponse,
    DatasetImportRequest,
    DatasetMetadata,
    DatasetSourceFormat,
    ImageStats,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)

_DATASET_ID_PREFIX = "dataset"
_DEFAULT_VERSION = "1.0"
_THUMBNAIL_MAX_SIZE = (128, 128)


def _utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class DatasetService:
    """Service for dataset import, persistence, and metadata access."""

    def __init__(
        self,
        *,
        paths: WorkspacePaths | None = None,
        store: JsonStore | None = None,
        project_service: ProjectService | None = None,
    ) -> None:
        self.paths = paths or WorkspacePaths.from_settings()
        self.store = store or JsonStore()
        self.project_service = project_service or ProjectService(
            paths=self.paths,
            store=self.store,
        )

    def import_dataset(
        self,
        project_id: str,
        payload: DatasetImportRequest,
    ) -> DatasetMetadata:
        """Import a dataset for an existing project.

        Args:
            project_id: Target project identifier.
            payload: Import request including source path and format.

        Returns:
            The persisted dataset metadata.
        """
        project = self.project_service.get_project(project_id)
        source_path = Path(payload.source_path).expanduser().resolve()
        if not source_path.exists():
            raise ValidationError(f"Dataset source path does not exist: {source_path}")
        if not source_path.is_dir():
            raise ValidationError(f"Dataset source path must be a directory: {source_path}")

        source_format = payload.source_format or self._detect_source_format(source_path)
        parsed_dataset = self._parse_dataset(
            source_path=source_path,
            source_format=source_format,
            task=project.task,
        )

        if parsed_dataset.task != project.task:
            raise ValidationError(
                "Imported dataset task does not match project task "
                f"('{parsed_dataset.task}' != '{project.task}')."
            )

        self._reset_dataset_layout(project_id)
        metadata = self._persist_dataset(project_id=project_id, parsed_dataset=parsed_dataset)
        return metadata

    def get_dataset(self, project_id: str) -> DatasetMetadata:
        """Load persisted dataset metadata for a project.

        Args:
            project_id: Target project identifier.

        Returns:
            Parsed dataset metadata.
        """
        self.project_service.get_project(project_id)
        try:
            payload = self.store.read(self.paths.dataset_metadata_file(project_id))
        except FileNotFoundError as exc:
            raise NotFoundError(f"Dataset not found for project {project_id}.") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        try:
            return DatasetMetadata.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(f"Dataset metadata is invalid for {project_id}.") from exc

    def clear_dataset(self, project_id: str) -> None:
        """Delete and recreate the dataset folder for a project."""
        self.project_service.get_project(project_id)
        self._reset_dataset_layout(project_id)

    def list_images(
        self,
        project_id: str,
        query: DatasetImageListQuery,
    ) -> DatasetImageListResponse:
        """List dataset images using pagination, filters, and sorting."""
        dataset = self.get_dataset(project_id)
        filter_class = query.filter_class.lower() if query.filter_class else None
        search_query = query.search.lower() if query.search else None
        selected_split_name = self._resolve_selected_split_name(
            requested_split_name=query.split_name,
            available_split_names=dataset.split_names,
        )
        selected_split_index = self._resolve_split_index(
            split_name=selected_split_name,
            split_names=dataset.split_names,
        )

        items: list[DatasetImageListItem] = []
        for image in dataset.images:
            class_name = self._extract_primary_class_name(
                task=dataset.task,
                annotations=image.annotations,
            )
            if filter_class is not None:
                if class_name is None or class_name.lower() != filter_class:
                    continue
            if search_query is not None and search_query not in image.filename.lower():
                continue

            selected_split_value = None
            if selected_split_index is not None and selected_split_index < len(image.split):
                selected_split_value = image.split[selected_split_index]

            items.append(
                DatasetImageListItem(
                    filename=image.filename,
                    width=image.width,
                    height=image.height,
                    class_name=class_name,
                    split=image.split,
                    selected_split_value=selected_split_value,
                    annotation_count=len(image.annotations),
                )
            )

        reverse = query.sort_order == "desc"
        if query.sort_by == "class":
            items.sort(
                key=lambda item: ((item.class_name or "").lower(), item.filename.lower()),
                reverse=reverse,
            )
        elif query.sort_by == "size":
            items.sort(
                key=lambda item: (item.width * item.height, item.filename.lower()),
                reverse=reverse,
            )
        else:
            items.sort(key=lambda item: item.filename.lower(), reverse=reverse)

        total_items = len(items)
        total_pages = math.ceil(total_items / query.page_size) if total_items else 0
        start = (query.page - 1) * query.page_size
        end = start + query.page_size

        return DatasetImageListResponse(
            page=query.page,
            page_size=query.page_size,
            total_items=total_items,
            total_pages=total_pages,
            selected_split_name=selected_split_name,
            items=items[start:end],
        )

    def get_image_info(self, project_id: str, filename: str) -> DatasetImage:
        """Return metadata for a single dataset image."""
        safe_filename = self._validate_image_filename(filename)
        dataset = self.get_dataset(project_id)
        for image in dataset.images:
            if image.filename == safe_filename:
                return image
        raise NotFoundError(f"Image '{safe_filename}' not found for project {project_id}.")

    def get_image_path(self, project_id: str, filename: str) -> Path:
        """Resolve and validate the path to a dataset image file."""
        safe_filename = self._validate_image_filename(filename)
        self.get_image_info(project_id, safe_filename)
        image_path = self.paths.dataset_images_dir(project_id) / safe_filename
        if not image_path.exists():
            raise NotFoundError(f"Image file '{safe_filename}' is missing on disk.")
        return image_path

    def get_thumbnail_path(self, project_id: str, filename: str) -> Path:
        """Resolve thumbnail path and generate it if missing."""
        safe_filename = self._validate_image_filename(filename)
        image_path = self.get_image_path(project_id, safe_filename)
        thumbnail_path = self.paths.dataset_thumbnails_dir(project_id) / safe_filename
        if not thumbnail_path.exists():
            self._generate_thumbnail(image_path, thumbnail_path)
        return thumbnail_path

    def _parse_dataset(
        self,
        *,
        source_path: Path,
        source_format: DatasetSourceFormat,
        task: str,
    ) -> ParsedDataset:
        try:
            if source_format == "image_folders":
                return parse_image_folders(source_path, task=task)
            if source_format == "csv":
                return parse_csv_dataset(source_path, task=task)
            if source_format == "coco":
                return parse_coco_dataset(source_path, task=task)
            if source_format == "yolo":
                return parse_yolo_dataset(source_path, task=task)
        except OSError as exc:
            raise ValidationError(f"Unable to parse dataset source: {exc}") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        raise ValidationError(f"Unsupported dataset source format: {source_format}")

    def _persist_dataset(
        self,
        *,
        project_id: str,
        parsed_dataset: ParsedDataset,
    ) -> DatasetMetadata:
        copied_images: list[DatasetImage] = []
        used_filenames: set[str] = set()

        image_widths: list[int] = []
        image_heights: list[int] = []
        image_formats: set[str] = set()

        dataset_images_dir = self.paths.dataset_images_dir(project_id)
        dataset_thumbs_dir = self.paths.dataset_thumbnails_dir(project_id)
        dataset_images_dir.mkdir(parents=True, exist_ok=True)
        dataset_thumbs_dir.mkdir(parents=True, exist_ok=True)

        for parsed_image in parsed_dataset.images:
            target_filename = self._next_available_filename(
                parsed_image.source_filename,
                used_filenames=used_filenames,
            )
            image_target_path = dataset_images_dir / target_filename
            thumbnail_target_path = dataset_thumbs_dir / target_filename

            shutil.copy2(parsed_image.source_path, image_target_path)
            self._generate_thumbnail(image_target_path, thumbnail_target_path)

            copied_images.append(
                DatasetImage(
                    filename=target_filename,
                    width=parsed_image.width,
                    height=parsed_image.height,
                    split=[],
                    annotations=parsed_image.annotations,
                )
            )
            image_widths.append(parsed_image.width)
            image_heights.append(parsed_image.height)
            image_formats.add(image_target_path.suffix.lower().lstrip("."))

        if not copied_images:
            raise ValidationError("Dataset import produced no images.")

        image_stats = ImageStats(
            num_images=len(copied_images),
            min_width=min(image_widths),
            max_width=max(image_widths),
            min_height=min(image_heights),
            max_height=max(image_heights),
            formats=sorted(image_formats),
        )

        metadata = DatasetMetadata(
            version=_DEFAULT_VERSION,
            id=self._generate_dataset_id(),
            task=parsed_dataset.task,
            source_format=cast(DatasetSourceFormat, parsed_dataset.source_format),
            source_path=str(parsed_dataset.source_path),
            imported_at=_utc_now(),
            classes=parsed_dataset.classes,
            split_names=[],
            image_stats=image_stats,
            images=copied_images,
        )
        self.store.write(
            self.paths.dataset_metadata_file(project_id),
            metadata.model_dump(mode="json"),
        )
        logger.info(
            "Imported dataset for project_id=%s with %d images",
            project_id,
            image_stats.num_images,
        )
        return metadata

    def _detect_source_format(self, source_path: Path) -> DatasetSourceFormat:
        csv_files = sorted(source_path.glob("*.csv"))
        json_files = sorted(source_path.glob("*.json"))
        classes_file = source_path / "classes.txt"
        labels_dir = source_path / "labels"
        has_visible_directories = any(
            entry.is_dir() and not entry.name.startswith(".") for entry in source_path.iterdir()
        )

        if json_files:
            return "coco"
        if (
            classes_file.exists()
            and classes_file.is_file()
            and labels_dir.exists()
            and labels_dir.is_dir()
        ):
            return "yolo"
        if csv_files:
            return "csv"
        if has_visible_directories:
            return "image_folders"
        raise ValidationError(
            "Could not detect source format automatically. Please provide source_format explicitly."
        )

    def _reset_dataset_layout(self, project_id: str) -> None:
        dataset_dir = self.paths.dataset_dir(project_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        self.paths.dataset_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.paths.dataset_images_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.paths.dataset_masks_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.paths.dataset_thumbnails_dir(project_id).mkdir(parents=True, exist_ok=True)

    def _generate_dataset_id(self) -> str:
        return f"{_DATASET_ID_PREFIX}-{uuid.uuid4().hex[:8]}"

    def _next_available_filename(
        self,
        source_filename: str,
        *,
        used_filenames: set[str],
    ) -> str:
        original = Path(source_filename).name.strip()
        if not original:
            original = "image"

        stem = Path(original).stem or "image"
        suffix = Path(original).suffix.lower()
        if not suffix:
            suffix = ".png"

        candidate = f"{stem}{suffix}"
        counter = 1
        while candidate in used_filenames:
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_filenames.add(candidate)
        return candidate

    def _generate_thumbnail(self, source_path: Path, target_path: Path) -> None:
        """Generate and save a thumbnail with the original filename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            thumbnail = image.copy()
            thumbnail.thumbnail(_THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
            save_options: dict[str, Any] = {}
            if target_path.suffix.lower() in {".jpg", ".jpeg"} and thumbnail.mode in {
                "RGBA",
                "LA",
                "P",
            }:
                thumbnail = thumbnail.convert("RGB")
                save_options["quality"] = 90
            thumbnail.save(target_path, **save_options)

    def _validate_image_filename(self, filename: str) -> str:
        normalized = filename.strip()
        if not normalized:
            raise ValidationError("Filename cannot be empty.")
        if Path(normalized).name != normalized:
            raise ValidationError("Filename must not contain path separators.")
        return normalized

    def _extract_primary_class_name(
        self,
        *,
        task: str,
        annotations: list[Any],
    ) -> str | None:
        if task == "anomaly_detection":
            for annotation in annotations:
                if getattr(annotation, "type", None) == "anomaly":
                    return "anomalous" if annotation.is_anomalous else "normal"
            return None

        if task == "object_detection":
            for annotation in annotations:
                if getattr(annotation, "type", None) == "bbox":
                    return annotation.class_name
            return None

        for annotation in annotations:
            if getattr(annotation, "type", None) == "label":
                return annotation.class_name
        return None

    def _resolve_selected_split_name(
        self,
        *,
        requested_split_name: str | None,
        available_split_names: list[str],
    ) -> str:
        if requested_split_name is None:
            return available_split_names[0] if available_split_names else ""
        if requested_split_name == "":
            return ""
        if requested_split_name in available_split_names:
            return requested_split_name
        return available_split_names[0] if available_split_names else ""

    def _resolve_split_index(self, *, split_name: str, split_names: list[str]) -> int | None:
        if not split_name:
            return None
        try:
            return split_names.index(split_name)
        except ValueError:
            return None
