from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError as PydanticValidationError

from app.config import get_settings
from app.core.dataset_service import DatasetService
from app.core.exceptions import (
    CheckpointNotFoundError,
    ConflictError,
    DatasetNotImportedError,
    ExperimentNotFoundError,
    NotFoundError,
    SplitNotFoundError,
    TrainingInProgressError,
    ValidationError,
)
from app.core.project_service import ProjectService
from app.core.training_form_layout import TrainingSection, get_training_sections
from app.models.catalog import build_default_training_config
from app.schemas.dataset import DatasetMetadata
from app.schemas.training import (
    ExperimentCreate,
    ExperimentError,
    ExperimentMetrics,
    ExperimentRecord,
    ExperimentsIndex,
    ExperimentSummary,
    ExperimentUpdate,
    HardwareConfig,
    HyperparameterConfig,
    ModelConfig,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths
from app.training.hardware import list_device_options
from app.training.subprocess_runner import TrainingProcessHandle, TrainingSubprocessRunner
from app.training.worker import run_experiment_training

logger = logging.getLogger(__name__)

_EXPERIMENT_ID_PREFIX = "exp"
_DEFAULT_VERSION = "1.0"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_LOCKED_STATUSES = {"pending", "training", "completed", "failed", "cancelled"}


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ActiveTrainingRun:
    """In-memory metadata for the currently running training subprocess."""

    project_id: str
    experiment_id: str
    handle: TrainingProcessHandle


class TrainingService:
    """Service for experiment management and training lifecycle orchestration."""

    def __init__(
        self,
        *,
        paths: WorkspacePaths | None = None,
        store: JsonStore | None = None,
        project_service: ProjectService | None = None,
        dataset_service: DatasetService | None = None,
        runner: TrainingSubprocessRunner | None = None,
    ) -> None:
        self.paths = paths or WorkspacePaths.from_settings()
        self.store = store or JsonStore()
        self.project_service = project_service or ProjectService(paths=self.paths, store=self.store)
        self.dataset_service = dataset_service or DatasetService(
            paths=self.paths,
            store=self.store,
            project_service=self.project_service,
        )
        self.runner = runner or TrainingSubprocessRunner()
        self._active_lock = threading.Lock()
        self._active_run: ActiveTrainingRun | None = None

    def list_experiments(self, project_id: str) -> list[ExperimentSummary]:
        """List experiments with status/metric summaries for one project."""
        self.project_service.get_project(project_id)
        index = self._load_experiments_index(project_id)
        refreshed: list[ExperimentSummary] = []
        changed = False

        for summary in index.experiments:
            try:
                experiment = self._load_experiment(project_id, summary.id)
            except ExperimentNotFoundError:
                logger.warning(
                    "Skipping missing experiment metadata for project_id=%s experiment_id=%s",
                    project_id,
                    summary.id,
                )
                changed = True
                continue

            current_summary = self._build_summary(experiment)
            refreshed.append(current_summary)
            if current_summary != summary:
                changed = True

        if changed:
            self._write_experiments_index(
                project_id,
                ExperimentsIndex(version=index.version, experiments=refreshed),
            )
        return refreshed

    def create_experiment(
        self,
        project_id: str,
        payload: ExperimentCreate | None = None,
    ) -> ExperimentRecord:
        """Create an experiment with task defaults and optional overrides."""
        payload = payload or ExperimentCreate()
        project = self.project_service.get_project(project_id)
        dataset = self._get_required_dataset(project_id)
        split_name = payload.split_name or self._resolve_default_split_name(dataset)
        self._validate_split_name(dataset, split_name)

        default_config = build_default_training_config(project.task)
        model = payload.model or default_config.model
        hyperparameters = payload.hyperparameters or default_config.hyperparameters
        augmentations = payload.augmentations or default_config.augmentations
        model, hyperparameters = self._normalize_dropout_alignment(model, hyperparameters)

        hardware = payload.hardware or HardwareConfig(
            selected_devices=self._default_selected_devices()
        )
        experiment = ExperimentRecord(
            id=self._generate_experiment_id(project_id),
            name=payload.name or self._next_default_experiment_name(project_id),
            description=payload.description,
            created_at=_utc_now(),
            status="created",
            split_name=split_name,
            model=model,
            hyperparameters=hyperparameters,
            augmentations=augmentations,
            hardware=hardware,
        )
        self._write_experiment(project_id, experiment)
        self._upsert_experiment_summary(project_id, experiment)
        return experiment

    def get_experiment(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        """Return one experiment record by ID."""
        self.project_service.get_project(project_id)
        experiment = self._load_experiment(project_id, experiment_id)
        self._upsert_experiment_summary(project_id, experiment)
        return experiment

    def update_experiment(
        self,
        project_id: str,
        experiment_id: str,
        payload: ExperimentUpdate,
    ) -> ExperimentRecord:
        """Patch editable configuration fields for a created experiment."""
        experiment = self.get_experiment(project_id, experiment_id)
        if experiment.status in _LOCKED_STATUSES:
            raise ConflictError(
                "Experiment configuration is locked once training has started. "
                "Restart the experiment to edit config."
            )

        split_name = experiment.split_name
        if payload.split_name is not None:
            dataset = self._get_required_dataset(project_id)
            self._validate_split_name(dataset, payload.split_name)
            split_name = payload.split_name

        model = payload.model or experiment.model
        hyperparameters = payload.hyperparameters or experiment.hyperparameters
        model, hyperparameters = self._normalize_dropout_alignment(model, hyperparameters)

        updated = experiment.model_copy(
            update={
                "name": payload.name if payload.name is not None else experiment.name,
                "description": (
                    payload.description
                    if payload.description is not None
                    else experiment.description
                ),
                "split_name": split_name,
                "model": model,
                "hyperparameters": hyperparameters,
                "augmentations": payload.augmentations or experiment.augmentations,
                "hardware": payload.hardware or experiment.hardware,
            }
        )
        self._write_experiment(project_id, updated)
        self._upsert_experiment_summary(project_id, updated)
        return updated

    def delete_experiment(self, project_id: str, experiment_id: str) -> None:
        """Delete experiment metadata and all generated artifacts."""
        self.get_experiment(project_id, experiment_id)
        active = self._get_active_run()
        if active is not None and active.experiment_id == experiment_id:
            raise ConflictError("Cannot delete an experiment while it is training.")

        experiment_dir = self.paths.experiment_dir(project_id, experiment_id)
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        self._remove_experiment_summary(project_id, experiment_id)

    def start_training(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        """Start training for an experiment from status `created`."""
        experiment = self.get_experiment(project_id, experiment_id)
        if experiment.status != "created":
            raise ConflictError(
                f"Experiment '{experiment_id}' cannot start from status '{experiment.status}'."
            )
        return self._start_training_internal(
            project_id=project_id,
            experiment=experiment,
            resume=False,
        )

    def resume_training(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        """Resume training from `last.ckpt` for failed/cancelled experiments."""
        experiment = self.get_experiment(project_id, experiment_id)
        if experiment.status not in {"failed", "cancelled"}:
            raise ConflictError(
                f"Experiment '{experiment_id}' cannot resume from status '{experiment.status}'."
            )

        last_checkpoint = (
            self.paths.experiment_checkpoints_dir(project_id, experiment_id) / "last.ckpt"
        )
        if not last_checkpoint.exists():
            raise CheckpointNotFoundError(
                f"Checkpoint 'last.ckpt' was not found for experiment '{experiment_id}'."
            )

        return self._start_training_internal(
            project_id=project_id,
            experiment=experiment,
            resume=True,
        )

    def stop_training(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        """Request graceful cancellation for a running training subprocess."""
        experiment = self.get_experiment(project_id, experiment_id)
        active = self._get_active_run()

        if active is None:
            if experiment.status in _TERMINAL_STATUSES:
                return experiment
            raise ConflictError("No experiment is currently training.")

        if active.project_id != project_id or active.experiment_id != experiment_id:
            raise ConflictError(f"Experiment '{experiment_id}' is not currently training.")

        self.runner.request_stop()
        return self.get_experiment(project_id, experiment_id)

    def restart_experiment(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        """Reset an experiment to `created` and delete all generated artifacts."""
        experiment = self.get_experiment(project_id, experiment_id)
        active = self._get_active_run()
        if active is not None and active.experiment_id == experiment_id:
            raise ConflictError("Cannot restart an experiment while it is training.")
        if experiment.status in {"pending", "training"}:
            raise ConflictError("Cannot restart an experiment while it is training.")

        metrics_file = self.paths.experiment_metrics_file(project_id, experiment_id)
        checkpoints_dir = self.paths.experiment_checkpoints_dir(project_id, experiment_id)
        logs_dir = self.paths.experiment_logs_dir(project_id, experiment_id)

        if metrics_file.exists():
            metrics_file.unlink()
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
        if logs_dir.exists():
            shutil.rmtree(logs_dir)

        reset = experiment.model_copy(
            update={
                "status": "created",
                "started_at": None,
                "completed_at": None,
                "best_epoch": None,
                "best_checkpoint_path": None,
                "best_metric": None,
                "final_metrics": None,
                "error": None,
            }
        )
        self._write_experiment(project_id, reset)
        self._upsert_experiment_summary(project_id, reset)
        return reset

    def get_metrics(self, project_id: str, experiment_id: str) -> ExperimentMetrics:
        """Return persisted epoch metrics for one experiment."""
        self.get_experiment(project_id, experiment_id)
        metrics_path = self.paths.experiment_metrics_file(project_id, experiment_id)
        raw_payload = self.store.read(metrics_path, default={"epochs": []})
        try:
            return ExperimentMetrics.model_validate(raw_payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Metrics metadata is invalid for experiment '{experiment_id}'."
            ) from exc

    def get_training_form_sections(self, task: str) -> tuple[TrainingSection, ...]:
        """Return collapsible section metadata for the training configuration form."""
        return get_training_sections(task)

    def list_available_devices(self) -> list[dict[str, object]]:
        """Return hardware options for the training configuration form."""
        return [option.to_payload() for option in list_device_options()]

    def _start_training_internal(
        self,
        *,
        project_id: str,
        experiment: ExperimentRecord,
        resume: bool,
    ) -> ExperimentRecord:
        dataset = self._get_required_dataset(project_id)
        self._validate_split_name(dataset, experiment.split_name)
        active = self._get_active_run()
        if active is not None:
            if active.project_id == project_id and active.experiment_id == experiment.id:
                raise ConflictError(f"Experiment '{experiment.id}' is already training.")
            raise TrainingInProgressError()

        pending_experiment = experiment.model_copy(
            update={
                "status": "pending",
                "completed_at": None,
                "error": None,
            }
        )
        self._write_experiment(project_id, pending_experiment)
        self._upsert_experiment_summary(project_id, pending_experiment)

        checkpoints_dir = self.paths.experiment_checkpoints_dir(project_id, experiment.id)
        logs_dir = self.paths.experiment_logs_dir(project_id, experiment.id)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        handle = self.runner.start(
            run_experiment_training,
            str(self.paths.root),
            project_id,
            experiment.id,
            resume,
        )
        with self._active_lock:
            self._active_run = ActiveTrainingRun(
                project_id=project_id,
                experiment_id=experiment.id,
                handle=handle,
            )

        monitor = threading.Thread(
            target=self._monitor_active_training,
            args=(project_id, experiment.id, handle),
            daemon=True,
            name=f"training-monitor-{project_id}-{experiment.id}",
        )
        monitor.start()
        return pending_experiment

    def _monitor_active_training(
        self,
        project_id: str,
        experiment_id: str,
        handle: TrainingProcessHandle,
    ) -> None:
        handle.wait()
        self.runner.get_active()
        with self._active_lock:
            if self._active_run is not None and self._active_run.experiment_id == experiment_id:
                self._active_run = None

        try:
            experiment = self._load_experiment(project_id, experiment_id)
        except ExperimentNotFoundError:
            return

        if experiment.status in _TERMINAL_STATUSES:
            self._upsert_experiment_summary(project_id, experiment)
            return

        # Fallback for abrupt worker exits where status was not persisted.
        fallback_status = "cancelled" if handle.stop_event.is_set() else "failed"
        fallback_error: ExperimentError | None = None
        if fallback_status == "failed":
            fallback_error = ExperimentError(
                type="RuntimeError",
                message=(
                    f"Training subprocess exited unexpectedly with code {handle.process.exitcode}."
                ),
                traceback="Training subprocess terminated before writing a terminal status.",
            )

        updated = experiment.model_copy(
            update={
                "status": fallback_status,
                "completed_at": _utc_now(),
                "error": fallback_error,
            }
        )
        self._write_experiment(project_id, updated)
        self._upsert_experiment_summary(project_id, updated)

    def _get_active_run(self) -> ActiveTrainingRun | None:
        handle = self.runner.get_active()
        with self._active_lock:
            if handle is None:
                self._active_run = None
                return None
            if self._active_run is None:
                return None
            self._active_run.handle = handle
            return self._active_run

    def _load_experiments_index(self, project_id: str) -> ExperimentsIndex:
        index_path = self.paths.experiments_index_file(project_id)
        self.store.update(
            index_path,
            lambda payload: payload,
            default_factory=lambda: ExperimentsIndex(
                version=_DEFAULT_VERSION,
                experiments=[],
            ).model_dump(mode="json"),
        )
        try:
            payload = self.store.read(index_path)
        except FileNotFoundError as exc:
            raise ValidationError("Experiments index is missing.") from exc

        try:
            return ExperimentsIndex.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError("Experiments index is invalid.") from exc

    def _write_experiments_index(self, project_id: str, index: ExperimentsIndex) -> None:
        self.store.write(
            self.paths.experiments_index_file(project_id),
            index.model_dump(mode="json"),
        )

    def _load_experiment(self, project_id: str, experiment_id: str) -> ExperimentRecord:
        experiment_path = self.paths.experiment_metadata_file(project_id, experiment_id)
        try:
            payload = self.store.read(experiment_path)
        except FileNotFoundError as exc:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' was not found.") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        try:
            return ExperimentRecord.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Experiment metadata is invalid for experiment '{experiment_id}'."
            ) from exc

    def _write_experiment(self, project_id: str, experiment: ExperimentRecord) -> None:
        self.store.write(
            self.paths.experiment_metadata_file(project_id, experiment.id),
            experiment.model_dump(mode="json"),
        )

    def _upsert_experiment_summary(self, project_id: str, experiment: ExperimentRecord) -> None:
        index = self._load_experiments_index(project_id)
        summary = self._build_summary(experiment)
        updated: list[ExperimentSummary] = []
        inserted = False
        for item in index.experiments:
            if item.id == summary.id:
                updated.append(summary)
                inserted = True
            else:
                updated.append(item)
        if not inserted:
            updated.append(summary)
        self._write_experiments_index(
            project_id,
            ExperimentsIndex(version=index.version, experiments=updated),
        )

    def _remove_experiment_summary(self, project_id: str, experiment_id: str) -> None:
        index = self._load_experiments_index(project_id)
        filtered = [item for item in index.experiments if item.id != experiment_id]
        self._write_experiments_index(
            project_id,
            ExperimentsIndex(version=index.version, experiments=filtered),
        )

    def _build_summary(self, experiment: ExperimentRecord) -> ExperimentSummary:
        best_metric_value = self._extract_best_metric_value(experiment.best_metric)
        return ExperimentSummary(
            id=experiment.id,
            name=experiment.name,
            created_at=experiment.created_at,
            status=experiment.status,
            best_metric_value=best_metric_value,
        )

    def _extract_best_metric_value(self, best_metric: dict[str, float] | None) -> float | None:
        if not best_metric:
            return None
        for value in best_metric.values():
            return float(value)
        return None

    def _get_required_dataset(self, project_id: str) -> DatasetMetadata:
        try:
            return self.dataset_service.get_dataset(project_id)
        except NotFoundError as exc:
            raise DatasetNotImportedError(project_id) from exc

    def _resolve_default_split_name(self, dataset: DatasetMetadata) -> str:
        if not dataset.split_names:
            raise SplitNotFoundError("No split is available. Create a split before training.")
        return dataset.split_names[0]

    def _validate_split_name(self, dataset: DatasetMetadata, split_name: str) -> None:
        if split_name not in dataset.split_names:
            raise SplitNotFoundError(
                f"Split '{split_name}' was not found in dataset '{dataset.id}'."
            )

    def _generate_experiment_id(self, project_id: str) -> str:
        for _ in range(20):
            candidate = f"{_EXPERIMENT_ID_PREFIX}-{uuid.uuid4().hex[:8]}"
            if not self.paths.experiment_dir(project_id, candidate).exists():
                return candidate
        raise ConflictError("Unable to generate a unique experiment ID.")

    def _next_default_experiment_name(self, project_id: str) -> str:
        existing = {experiment.name for experiment in self.list_experiments(project_id)}
        index = 1
        while True:
            candidate = f"Experiment {index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _normalize_dropout_alignment(
        self,
        model: ModelConfig,
        hyperparameters: HyperparameterConfig,
    ) -> tuple[ModelConfig, HyperparameterConfig]:
        if model.dropout != hyperparameters.dropout:
            model = model.model_copy(update={"dropout": hyperparameters.dropout})
        return model, hyperparameters

    def _default_selected_devices(self) -> list[str]:
        default_device = get_settings().default_device.strip().lower()
        if default_device in {"cuda", "gpu"}:
            return ["gpu:0"]
        if default_device.startswith(("cuda:", "gpu:")):
            _, _, raw_index = default_device.partition(":")
            if raw_index.isdigit():
                return [f"gpu:{int(raw_index)}"]
        return ["cpu"]
