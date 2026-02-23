from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.dataset_service import DatasetService
from app.core.exceptions import (
    ConflictError,
    DatasetNotImportedError,
    NotFoundError,
    SplitNotFoundError,
    ValidationError,
)
from app.core.project_service import ProjectService
from app.datasets.splits import SplitComputation, compute_split_assignments
from app.schemas.dataset import DatasetMetadata
from app.schemas.split import (
    SplitCounts,
    SplitCreateRequest,
    SplitPreviewRequest,
    SplitPreviewResponse,
    SplitRatios,
    SplitSummary,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)


class SplitService:
    """Service for split preview, persistence, and retrieval."""

    def __init__(
        self,
        *,
        paths: WorkspacePaths | None = None,
        store: JsonStore | None = None,
        project_service: ProjectService | None = None,
        dataset_service: DatasetService | None = None,
    ) -> None:
        self.paths = paths or WorkspacePaths.from_settings()
        self.store = store or JsonStore()
        self.project_service = project_service or ProjectService(
            paths=self.paths,
            store=self.store,
        )
        self.dataset_service = dataset_service or DatasetService(
            paths=self.paths,
            store=self.store,
            project_service=self.project_service,
        )

    def preview_split(
        self,
        project_id: str,
        payload: SplitPreviewRequest,
    ) -> SplitPreviewResponse:
        """Compute split statistics without persisting assignments."""
        dataset = self._get_required_dataset(project_id)
        computation = self._compute_split(
            dataset=dataset,
            seed=payload.seed,
            ratios=payload.ratios,
        )
        return SplitPreviewResponse(
            ratios=payload.ratios,
            seed=self._resolve_seed(payload.seed),
            stats=self._to_split_counts(computation.subset_counts),
            class_distribution=self._to_class_distribution(computation.class_distribution),
            warnings=computation.warnings,
        )

    def create_split(
        self,
        project_id: str,
        payload: SplitCreateRequest,
    ) -> SplitSummary:
        """Persist a new immutable split by appending it to dataset metadata."""
        self.project_service.get_project(project_id)
        self._get_required_dataset(project_id)

        split_name = payload.name
        split_index: int | None = None
        split_result: SplitComputation | None = None

        def update_dataset(raw_payload: Any) -> dict[str, Any]:
            nonlocal split_index
            nonlocal split_result

            dataset = self._validate_dataset_payload(raw_payload, project_id=project_id)
            if split_name in dataset.split_names:
                raise ConflictError(f"Split '{split_name}' already exists.")

            split_result = self._compute_split(
                dataset=dataset,
                seed=payload.seed,
                ratios=payload.ratios,
            )
            dataset.split_names.append(split_name)
            split_index = len(dataset.split_names) - 1

            for image, subset in zip(dataset.images, split_result.assignments, strict=True):
                image.split.append(subset)

            return dataset.model_dump(mode="json")

        dataset_path = self.paths.dataset_metadata_file(project_id)
        self.store.update(dataset_path, update_dataset)
        if split_result is None or split_index is None:
            raise ValidationError("Split creation failed before assignments were produced.")

        logger.info(
            "Created split '%s' for project_id=%s at index=%d",
            split_name,
            project_id,
            split_index,
        )
        return SplitSummary(
            name=split_name,
            index=split_index,
            immutable=True,
            stats=self._to_split_counts(split_result.subset_counts),
            class_distribution=self._to_class_distribution(split_result.class_distribution),
        )

    def list_splits(self, project_id: str) -> list[SplitSummary]:
        """List all saved splits with derived per-subset statistics."""
        dataset = self._get_required_dataset(project_id)
        labels = self._extract_labels(dataset)
        summaries: list[SplitSummary] = []
        for index, split_name in enumerate(dataset.split_names):
            counts = {"train": 0, "val": 0, "test": 0, "none": 0}
            class_distribution: dict[str, dict[str, int]] = {}

            for class_name in dataset.classes:
                class_distribution[class_name] = {"train": 0, "val": 0, "test": 0, "none": 0}

            for class_name in sorted(set(labels)):
                class_distribution.setdefault(
                    class_name, {"train": 0, "val": 0, "test": 0, "none": 0}
                )

            for image_index, image in enumerate(dataset.images):
                subset = image.split[index]
                counts[subset] += 1
                class_distribution[labels[image_index]][subset] += 1

            summaries.append(
                SplitSummary(
                    name=split_name,
                    index=index,
                    immutable=True,
                    stats=self._to_split_counts(counts),
                    class_distribution=self._to_class_distribution(class_distribution),
                )
            )
        return summaries

    def get_split(self, project_id: str, split_name: str) -> SplitSummary:
        """Return one saved split summary by split name."""
        for summary in self.list_splits(project_id):
            if summary.name == split_name:
                return summary
        raise SplitNotFoundError(f"Split '{split_name}' was not found.")

    def delete_split(self, project_id: str, split_name: str) -> None:
        """Delete a saved split from dataset metadata."""
        self.project_service.get_project(project_id)
        self._get_required_dataset(project_id)

        def update_dataset(raw_payload: Any) -> dict[str, Any]:
            dataset = self._validate_dataset_payload(raw_payload, project_id=project_id)
            try:
                split_index = dataset.split_names.index(split_name)
            except ValueError as exc:
                raise SplitNotFoundError(f"Split '{split_name}' was not found.") from exc

            dataset.split_names.pop(split_index)
            for image in dataset.images:
                image.split.pop(split_index)
            return dataset.model_dump(mode="json")

        dataset_path = self.paths.dataset_metadata_file(project_id)
        self.store.update(dataset_path, update_dataset)
        logger.info("Deleted split '%s' for project_id=%s", split_name, project_id)

    def _compute_split(
        self,
        *,
        dataset: DatasetMetadata,
        seed: int | None,
        ratios: SplitRatios,
    ) -> SplitComputation:
        try:
            return compute_split_assignments(
                images=dataset.images,
                task=dataset.task,
                ratios=ratios,
                seed=seed,
                class_order=dataset.classes,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _validate_dataset_payload(self, payload: Any, *, project_id: str) -> DatasetMetadata:
        try:
            return DatasetMetadata.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(f"Dataset metadata is invalid for {project_id}.") from exc

    def _get_required_dataset(self, project_id: str) -> DatasetMetadata:
        try:
            return self.dataset_service.get_dataset(project_id)
        except NotFoundError as exc:
            raise DatasetNotImportedError(project_id) from exc

    def _extract_labels(self, dataset: DatasetMetadata) -> list[str]:
        labels: list[str] = []
        for image in dataset.images:
            if dataset.task == "anomaly_detection":
                found = False
                for annotation in image.annotations:
                    if annotation.type == "anomaly":
                        labels.append("anomalous" if annotation.is_anomalous else "normal")
                        found = True
                        break
                if found:
                    continue
                raise ValidationError("Anomaly dataset images must include anomaly annotations.")

            if dataset.task == "object_detection":
                bbox_label = None
                for annotation in image.annotations:
                    if annotation.type == "bbox":
                        bbox_label = annotation.class_name
                        break
                labels.append(bbox_label or "__background__")
                continue

            for annotation in image.annotations:
                if annotation.type == "label":
                    labels.append(annotation.class_name)
                    break
            else:
                raise ValidationError(
                    "Classification dataset images must include label annotations."
                )
        return labels

    def _resolve_seed(self, seed: int | None) -> int:
        return 0 if seed is None else seed

    def _to_split_counts(self, counts: dict[str, int]) -> SplitCounts:
        return SplitCounts(
            train=counts.get("train", 0),
            val=counts.get("val", 0),
            test=counts.get("test", 0),
            none=counts.get("none", 0),
        )

    def _to_class_distribution(
        self,
        class_distribution: dict[str, dict[str, int]],
    ) -> dict[str, SplitCounts]:
        return {
            class_name: self._to_split_counts(counts)
            for class_name, counts in class_distribution.items()
        }
