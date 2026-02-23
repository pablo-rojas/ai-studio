from __future__ import annotations

import logging
import math
import shutil
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
from app.evaluation.evaluator import Evaluator
from app.schemas.evaluation import (
    ClassificationAggregateMetrics,
    ClassificationPerImageResult,
    EvaluationConfig,
    EvaluationPerImageResult,
    EvaluationProgress,
    EvaluationRecord,
    EvaluationResultsFile,
    EvaluationResultsPage,
    EvaluationResultsQuery,
    ObjectDetectionAggregateMetrics,
    ObjectDetectionPerImageResult,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationService:
    """Service for experiment-scoped evaluation lifecycle operations."""

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

    def start_evaluation(
        self,
        project_id: str,
        experiment_id: str,
        config: EvaluationConfig,
    ) -> EvaluationRecord:
        """Run evaluation for a completed experiment and persist outputs."""
        experiment = self._get_required_experiment(project_id, experiment_id)
        if experiment.status != "completed":
            raise ConflictError(
                f"Experiment '{experiment_id}' must be completed before evaluation starts."
            )
        if self._evaluation_exists(project_id, experiment_id):
            raise ConflictError(
                f"Evaluation already exists for experiment '{experiment_id}'. Reset it first."
            )

        dataset = self._get_required_dataset(project_id)
        checkpoint_path = self._resolve_checkpoint_path(
            project_id=project_id,
            experiment_id=experiment_id,
            checkpoint=config.checkpoint,
        )

        started_at = _utc_now()
        running = EvaluationRecord(
            checkpoint=config.checkpoint,
            split_subsets=config.split_subsets,
            batch_size=config.batch_size,
            device=config.device,
            status="running",
            progress=EvaluationProgress(processed=0, total=0),
            created_at=started_at,
            started_at=started_at,
            completed_at=None,
            error=None,
        )
        self._write_record(project_id, experiment_id, running)

        try:
            evaluator = Evaluator(
                config=config,
                project_id=project_id,
                experiment_id=experiment_id,
                dataset=dataset,
                experiment=experiment,
                images_dir=self.paths.dataset_images_dir(project_id),
                checkpoint_path=checkpoint_path,
                progress_callback=lambda processed, total: self._persist_progress(
                    project_id=project_id,
                    experiment_id=experiment_id,
                    processed=processed,
                    total=total,
                ),
            )
            output = evaluator.run()

            self.store.write(
                self.paths.experiment_evaluation_results_file(project_id, experiment_id),
                EvaluationResultsFile(results=output.results).model_dump(mode="json"),
            )
            self.store.write(
                self.paths.experiment_evaluation_aggregate_file(project_id, experiment_id),
                output.aggregate.model_dump(mode="json"),
            )

            completed = running.model_copy(
                update={
                    "status": "completed",
                    "progress": EvaluationProgress(
                        processed=len(output.results),
                        total=len(output.results),
                    ),
                    "completed_at": _utc_now(),
                    "error": None,
                }
            )
            self._write_record(project_id, experiment_id, completed)
            return completed
        except Exception as exc:
            failed = running.model_copy(
                update={
                    "status": "failed",
                    "completed_at": _utc_now(),
                    "error": str(exc),
                }
            )
            self._write_record(project_id, experiment_id, failed)
            logger.exception(
                "Evaluation failed for project_id=%s experiment_id=%s",
                project_id,
                experiment_id,
            )
            if isinstance(exc, (ValidationError, ConflictError, CheckpointNotFoundError)):
                raise
            raise ValidationError(f"Evaluation failed: {exc}") from exc

    def get_evaluation(self, project_id: str, experiment_id: str) -> EvaluationRecord:
        """Return persisted evaluation metadata for an experiment."""
        self._get_required_experiment(project_id, experiment_id)
        return self._load_record(project_id, experiment_id)

    def get_aggregate_metrics(
        self,
        project_id: str,
        experiment_id: str,
    ) -> ClassificationAggregateMetrics | ObjectDetectionAggregateMetrics | None:
        """Return aggregate metrics when available."""
        self.get_evaluation(project_id, experiment_id)
        aggregate_path = self.paths.experiment_evaluation_aggregate_file(project_id, experiment_id)
        payload = self.store.read(aggregate_path, default=None)
        if payload is None:
            return None
        dataset = self._get_required_dataset(project_id)
        try:
            if dataset.task == "object_detection":
                return ObjectDetectionAggregateMetrics.model_validate(payload)
            return ClassificationAggregateMetrics.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Aggregate evaluation metrics are invalid for experiment '{experiment_id}'."
            ) from exc

    def get_results(
        self,
        project_id: str,
        experiment_id: str,
        query: EvaluationResultsQuery,
    ) -> EvaluationResultsPage:
        """Return paginated and filterable per-image evaluation results."""
        self.get_evaluation(project_id, experiment_id)
        results_file = self._load_results_file(project_id, experiment_id)

        filtered = results_file.results
        if query.filter_subset is not None:
            filtered = [result for result in filtered if result.subset == query.filter_subset]
        if query.filter_correct is not None:
            filtered = [
                result
                for result in filtered
                if self._is_result_correct(result) == query.filter_correct
            ]
        if query.filter_class is not None:
            expected = query.filter_class.lower()
            filtered = [
                result
                for result in filtered
                if (self._extract_filter_class_name(result) or "").lower() == expected
            ]

        reverse = query.sort_order == "desc"
        if query.sort_by == "confidence":
            filtered.sort(key=self._result_confidence_sort_value, reverse=reverse)
        elif query.sort_by == "error":
            filtered.sort(key=self._result_error_sort_value, reverse=reverse)
        else:
            filtered.sort(key=lambda item: item.filename.lower(), reverse=reverse)

        total_items = len(filtered)
        total_pages = math.ceil(total_items / query.page_size) if total_items else 0
        start = (query.page - 1) * query.page_size
        end = start + query.page_size

        return EvaluationResultsPage(
            page=query.page,
            page_size=query.page_size,
            total_items=total_items,
            total_pages=total_pages,
            items=filtered[start:end],
        )

    def get_result(
        self,
        project_id: str,
        experiment_id: str,
        filename: str,
    ) -> EvaluationPerImageResult:
        """Return one per-image evaluation result by filename."""
        self.get_evaluation(project_id, experiment_id)
        safe_filename = self._normalize_result_filename(filename)
        results_file = self._load_results_file(project_id, experiment_id)
        for result in results_file.results:
            if result.filename == safe_filename:
                return result
        raise NotFoundError(
            f"Evaluation result '{safe_filename}' was not found for experiment '{experiment_id}'."
        )

    def reset_evaluation(self, project_id: str, experiment_id: str) -> None:
        """Delete the experiment's evaluation folder immediately."""
        self._get_required_experiment(project_id, experiment_id)
        evaluation_dir = self.paths.experiment_evaluation_dir(project_id, experiment_id)
        if evaluation_dir.exists():
            shutil.rmtree(evaluation_dir)

    def list_checkpoints(self, project_id: str, experiment_id: str) -> list[str]:
        """List checkpoint names that physically exist on disk."""
        self._get_required_experiment(project_id, experiment_id)
        checkpoints_dir = self.paths.experiment_checkpoints_dir(project_id, experiment_id)
        if not checkpoints_dir.exists():
            return []
        return [path.stem for path in sorted(checkpoints_dir.glob("*.ckpt"))]

    def _evaluation_exists(self, project_id: str, experiment_id: str) -> bool:
        return self.paths.experiment_evaluation_metadata_file(project_id, experiment_id).exists()

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
        self.project_service.get_project(project_id)
        try:
            return self.training_service.get_experiment(project_id, experiment_id)
        except ExperimentNotFoundError:
            raise
        except NotFoundError as exc:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' was not found.") from exc

    def _persist_progress(
        self,
        *,
        project_id: str,
        experiment_id: str,
        processed: int,
        total: int,
    ) -> None:
        try:
            current = self._load_record(project_id, experiment_id)
        except NotFoundError:
            return
        updated = current.model_copy(
            update={"progress": EvaluationProgress(processed=processed, total=total)}
        )
        self._write_record(project_id, experiment_id, updated)

    def _load_results_file(self, project_id: str, experiment_id: str) -> EvaluationResultsFile:
        payload = self.store.read(
            self.paths.experiment_evaluation_results_file(project_id, experiment_id),
            default={"results": []},
        )
        try:
            return EvaluationResultsFile.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Evaluation results are invalid for experiment '{experiment_id}'."
            ) from exc

    def _normalize_result_filename(self, filename: str) -> str:
        normalized = filename.strip()
        if not normalized:
            raise ValidationError("Filename cannot be empty.")
        if Path(normalized).name != normalized:
            raise ValidationError("Filename must not contain path separators.")
        return normalized

    def _is_result_correct(self, result: EvaluationPerImageResult) -> bool:
        if isinstance(result, ClassificationPerImageResult):
            return result.correct
        if isinstance(result, ObjectDetectionPerImageResult):
            return result.false_positives == 0 and result.false_negatives == 0
        return False

    def _extract_filter_class_name(self, result: EvaluationPerImageResult) -> str | None:
        if isinstance(result, ClassificationPerImageResult):
            return result.ground_truth.class_name
        if isinstance(result, ObjectDetectionPerImageResult):
            if result.ground_truth:
                return result.ground_truth[0].class_name
            if result.predictions:
                return result.predictions[0].class_name
        return None

    def _result_confidence_sort_value(self, result: EvaluationPerImageResult) -> float:
        if isinstance(result, ClassificationPerImageResult):
            return float(result.prediction.confidence)
        if isinstance(result, ObjectDetectionPerImageResult):
            if not result.predictions:
                return 0.0
            return max(float(prediction.confidence) for prediction in result.predictions)
        return 0.0

    def _result_error_sort_value(self, result: EvaluationPerImageResult) -> float:
        if isinstance(result, ClassificationPerImageResult):
            return 1.0 - float(result.prediction.confidence)
        if isinstance(result, ObjectDetectionPerImageResult):
            return float(result.false_positives + result.false_negatives)
        return 0.0

    def _load_record(self, project_id: str, experiment_id: str) -> EvaluationRecord:
        path = self.paths.experiment_evaluation_metadata_file(project_id, experiment_id)
        try:
            payload = self.store.read(path)
        except FileNotFoundError as exc:
            raise NotFoundError(f"Evaluation not found for experiment '{experiment_id}'.") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        try:
            return EvaluationRecord.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Evaluation metadata is invalid for experiment '{experiment_id}'."
            ) from exc

    def _write_record(self, project_id: str, experiment_id: str, record: EvaluationRecord) -> None:
        self.store.write(
            self.paths.experiment_evaluation_metadata_file(project_id, experiment_id),
            record.model_dump(mode="json"),
        )
