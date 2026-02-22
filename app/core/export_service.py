from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from app.core.dataset_service import DatasetService
from app.core.exceptions import (
    CheckpointNotFoundError,
    ConflictError,
    DatasetNotImportedError,
    ExperimentNotFoundError,
    NotFoundError,
    ValidationError,
)
from app.core.project_service import ProjectService
from app.core.training_service import TrainingService
from app.export import get_exporter
from app.export import list_formats as list_registered_formats
from app.schemas.export import (
    ExportCreate,
    ExportFormatInfo,
    ExportRecord,
    ExportsIndex,
    ExportSummary,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)

_EXPORT_ID_PREFIX = "export"
_DEFAULT_VERSION = "1.0"


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class ExportService:
    """Service for export lifecycle orchestration and metadata persistence."""

    def __init__(
        self,
        *,
        paths: WorkspacePaths | None = None,
        store: JsonStore | None = None,
        project_service: ProjectService | None = None,
        dataset_service: DatasetService | None = None,
        training_service: TrainingService | None = None,
    ) -> None:
        self.paths = paths or WorkspacePaths.from_settings()
        self.store = store or JsonStore()
        self.project_service = project_service or ProjectService(paths=self.paths, store=self.store)
        self.dataset_service = dataset_service or DatasetService(
            paths=self.paths,
            store=self.store,
            project_service=self.project_service,
        )
        self.training_service = training_service or TrainingService(
            paths=self.paths,
            store=self.store,
            project_service=self.project_service,
            dataset_service=self.dataset_service,
        )

    def list_formats(self) -> list[ExportFormatInfo]:
        """Return export format availability metadata."""
        return list_registered_formats()

    def list_exports(self, project_id: str) -> list[ExportSummary]:
        """List all exports for one project with refreshed summaries."""
        self.project_service.get_project(project_id)
        index = self._load_exports_index(project_id)

        refreshed: list[ExportSummary] = []
        changed = False
        for summary in index.exports:
            try:
                record = self._load_export(project_id, summary.id)
            except NotFoundError:
                logger.warning(
                    "Skipping missing export metadata for project_id=%s export_id=%s",
                    project_id,
                    summary.id,
                )
                changed = True
                continue

            current_summary = self._build_summary(record)
            refreshed.append(current_summary)
            if current_summary != summary:
                changed = True

        if changed:
            self._write_exports_index(
                project_id,
                ExportsIndex(version=index.version, exports=refreshed),
            )
        return refreshed

    def create_export(self, project_id: str, payload: ExportCreate) -> ExportRecord:
        """Create and execute one export for a completed experiment."""
        project = self.project_service.get_project(project_id)
        dataset = self._get_required_dataset(project_id)
        experiment = self._get_required_experiment(project_id, payload.experiment_id)

        if experiment.status != "completed":
            raise ConflictError(
                f"Experiment '{payload.experiment_id}' must be completed before export starts."
            )

        checkpoint_path = self._resolve_checkpoint_path(
            project_id=project_id,
            experiment_id=payload.experiment_id,
            checkpoint=payload.checkpoint,
        )

        try:
            exporter = get_exporter(payload.format)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        export_id = self._generate_export_id(project_id)
        output_file = f"model.{payload.format}"
        output_path = self.paths.export_dir(project_id, export_id) / output_file

        started_at = _utc_now()
        running = ExportRecord(
            id=export_id,
            experiment_id=payload.experiment_id,
            checkpoint=payload.checkpoint,
            format=payload.format,
            options=payload.options,
            status="running",
            created_at=started_at,
            started_at=started_at,
            completed_at=None,
            output_file=None,
            output_size_mb=None,
            validation=None,
            error=None,
        )
        self._write_export(project_id, running)
        self._upsert_export_summary(project_id, running)

        try:
            validation_result = exporter(
                task=project.task,
                model_config=experiment.model,
                num_classes=len(dataset.classes),
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                options=payload.options,
            )
            if not output_path.exists():
                raise ValidationError(
                    f"Export did not produce an output file for export '{export_id}'."
                )

            completed = running.model_copy(
                update={
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "output_file": output_file,
                    "output_size_mb": self._size_mb(output_path),
                    "validation": validation_result,
                    "error": None,
                }
            )
            self._write_export(project_id, completed)
            self._upsert_export_summary(project_id, completed)
            return completed
        except Exception as exc:
            failed = running.model_copy(
                update={
                    "status": "failed",
                    "completed_at": _utc_now(),
                    "error": str(exc),
                }
            )
            self._write_export(project_id, failed)
            self._upsert_export_summary(project_id, failed)
            logger.exception(
                "Export failed for project_id=%s export_id=%s",
                project_id,
                export_id,
            )
            if isinstance(exc, (ValidationError, ConflictError, CheckpointNotFoundError)):
                raise
            raise ValidationError(f"Export failed: {exc}") from exc

    def get_export(self, project_id: str, export_id: str) -> ExportRecord:
        """Return one export record by ID."""
        self.project_service.get_project(project_id)
        record = self._load_export(project_id, export_id)
        self._upsert_export_summary(project_id, record)
        return record

    def delete_export(self, project_id: str, export_id: str) -> None:
        """Delete one export metadata folder and remove it from the index."""
        self.get_export(project_id, export_id)
        export_dir = self.paths.export_dir(project_id, export_id)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        self._remove_export_summary(project_id, export_id)

    def resolve_output_file(self, project_id: str, export_id: str) -> Path:
        """Resolve the exported model file path for download operations."""
        record = self.get_export(project_id, export_id)
        if record.output_file is None:
            raise NotFoundError(f"Export '{export_id}' does not have an output file yet.")

        output_path = self.paths.export_dir(project_id, export_id) / record.output_file
        if not output_path.exists():
            raise NotFoundError(
                f"Export output file '{record.output_file}' is missing for '{export_id}'."
            )
        return output_path

    def _generate_export_id(self, project_id: str) -> str:
        for _ in range(20):
            candidate = f"{_EXPORT_ID_PREFIX}-{uuid.uuid4().hex[:8]}"
            if not self.paths.export_dir(project_id, candidate).exists():
                return candidate
        raise ConflictError("Unable to generate a unique export ID.")

    def _load_exports_index(self, project_id: str) -> ExportsIndex:
        index_path = self.paths.exports_index_file(project_id)
        payload = self.store.read(
            index_path,
            default=ExportsIndex(version=_DEFAULT_VERSION, exports=[]).model_dump(mode="json"),
        )
        try:
            return ExportsIndex.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Export index metadata is invalid for project '{project_id}'."
            ) from exc

    def _write_exports_index(self, project_id: str, index: ExportsIndex) -> None:
        self.store.write(
            self.paths.exports_index_file(project_id),
            index.model_dump(mode="json"),
        )

    def _load_export(self, project_id: str, export_id: str) -> ExportRecord:
        try:
            payload = self.store.read(self.paths.export_metadata_file(project_id, export_id))
        except FileNotFoundError as exc:
            raise NotFoundError(f"Export '{export_id}' was not found.") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        try:
            return ExportRecord.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(f"Export metadata is invalid for export '{export_id}'.") from exc

    def _write_export(self, project_id: str, export: ExportRecord) -> None:
        self.store.write(
            self.paths.export_metadata_file(project_id, export.id),
            export.model_dump(mode="json"),
        )

    def _upsert_export_summary(self, project_id: str, export: ExportRecord) -> None:
        index = self._load_exports_index(project_id)
        summary = self._build_summary(export)

        updated: list[ExportSummary] = []
        inserted = False
        for item in index.exports:
            if item.id == summary.id:
                updated.append(summary)
                inserted = True
            else:
                updated.append(item)
        if not inserted:
            updated.append(summary)

        self._write_exports_index(
            project_id,
            ExportsIndex(version=index.version, exports=updated),
        )

    def _remove_export_summary(self, project_id: str, export_id: str) -> None:
        index = self._load_exports_index(project_id)
        filtered = [item for item in index.exports if item.id != export_id]
        self._write_exports_index(
            project_id,
            ExportsIndex(version=index.version, exports=filtered),
        )

    def _build_summary(self, export: ExportRecord) -> ExportSummary:
        return ExportSummary(
            id=export.id,
            experiment_id=export.experiment_id,
            checkpoint=export.checkpoint,
            format=export.format,
            status=export.status,
            created_at=export.created_at,
            output_size_mb=export.output_size_mb,
        )

    def _resolve_checkpoint_path(
        self,
        *,
        project_id: str,
        experiment_id: str,
        checkpoint: str,
    ) -> Path:
        checkpoint_name = checkpoint if checkpoint.endswith(".ckpt") else f"{checkpoint}.ckpt"
        checkpoint_path = (
            self.paths.experiment_checkpoints_dir(project_id, experiment_id) / checkpoint_name
        )
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"Checkpoint '{checkpoint_name}' was not found for experiment '{experiment_id}'."
            )
        return checkpoint_path

    def _get_required_dataset(self, project_id: str):
        try:
            return self.dataset_service.get_dataset(project_id)
        except NotFoundError as exc:
            raise DatasetNotImportedError(project_id) from exc

    def _get_required_experiment(self, project_id: str, experiment_id: str):
        try:
            return self.training_service.get_experiment(project_id, experiment_id)
        except ExperimentNotFoundError:
            raise
        except NotFoundError as exc:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' was not found.") from exc

    def _size_mb(self, path: Path) -> float:
        return round(path.stat().st_size / (1024 * 1024), 3)
