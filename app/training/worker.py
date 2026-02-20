from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.datasets.base import AIStudioDataModule
from app.models.catalog import create_model, get_task_config
from app.schemas.dataset import DatasetMetadata
from app.schemas.project import ProjectResponse
from app.schemas.training import (
    ExperimentError,
    ExperimentMetrics,
    ExperimentRecord,
    ExperimentsIndex,
    ExperimentSummary,
    TrainingConfig,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths
from app.training.lightning_module import AIStudioModule
from app.training.losses import build_loss
from app.training.trainer_factory import build_trainer

_DEFAULT_VERSION = "1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_experiment_training(
    stop_event,
    workspace_root: str,
    project_id: str,
    experiment_id: str,
    resume: bool,
) -> None:
    """Subprocess entrypoint that runs one experiment training job."""
    paths = WorkspacePaths(root=Path(workspace_root))
    store = JsonStore()

    project = _load_project(store, paths, project_id)
    dataset = _load_dataset(store, paths, project_id)
    experiment = _load_experiment(store, paths, project_id, experiment_id)

    monitor_metric = f"val_{get_task_config(project.task).primary_metric}"
    experiment = experiment.model_copy(
        update={
            "status": "training",
            "started_at": _utc_now(),
            "completed_at": None,
            "error": None,
        }
    )
    _write_experiment(store, paths, project_id, experiment)
    _upsert_summary(store, paths, project_id, experiment)

    try:
        training_config = TrainingConfig(
            model=experiment.model,
            hyperparameters=experiment.hyperparameters,
            augmentations=experiment.augmentations,
        )

        model = create_model(
            task=project.task,
            architecture=experiment.model.backbone,
            config=experiment.model,
            num_classes=len(dataset.classes),
        )
        loss_fn = build_loss(
            project.task,
            experiment.hyperparameters.loss,
            label_smoothing=experiment.hyperparameters.label_smoothing,
        )
        module = AIStudioModule(
            task=project.task,
            model=model,
            loss_fn=loss_fn,
            hyperparameters=experiment.hyperparameters,
            num_classes=len(dataset.classes),
        )
        data_module = AIStudioDataModule(
            dataset=dataset,
            images_dir=paths.dataset_images_dir(project_id),
            split_name=experiment.split_name,
            augmentations=experiment.augmentations,
            batch_size=experiment.hyperparameters.batch_size,
        )
        data_module.setup()

        trainer = build_trainer(
            experiment_dir=paths.experiment_dir(project_id, experiment_id),
            training_config=training_config,
            selected_devices=experiment.hardware.selected_devices,
            metrics_file=paths.experiment_metrics_file(project_id, experiment_id),
            monitor_metric=monitor_metric,
            enable_progress_bar=False,
            stop_requested=stop_event.is_set,
        )

        checkpoint_path: str | None = None
        if resume:
            last_checkpoint = (
                paths.experiment_checkpoints_dir(project_id, experiment_id) / "last.ckpt"
            )
            if last_checkpoint.exists():
                checkpoint_path = str(last_checkpoint)

        trainer.fit(module=module, datamodule=data_module, ckpt_path=checkpoint_path)

        metrics = _load_metrics(store, paths, project_id, experiment_id)
        final_metrics = _final_metrics(metrics)
        best_epoch, best_metric = _best_metric(metrics, monitor_metric)
        best_checkpoint = paths.experiment_checkpoints_dir(project_id, experiment_id) / "best.ckpt"
        status = "cancelled" if stop_event.is_set() else "completed"

        completed = experiment.model_copy(
            update={
                "status": status,
                "completed_at": _utc_now(),
                "best_epoch": best_epoch,
                "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else None,
                "best_metric": best_metric,
                "final_metrics": final_metrics,
                "error": None,
            }
        )
        _write_experiment(store, paths, project_id, completed)
        _upsert_summary(store, paths, project_id, completed)
    except Exception as exc:
        status = "cancelled" if stop_event.is_set() else "failed"
        failed = experiment.model_copy(
            update={
                "status": status,
                "completed_at": _utc_now(),
                "error": ExperimentError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                ),
            }
        )
        _write_experiment(store, paths, project_id, failed)
        _upsert_summary(store, paths, project_id, failed)


def _load_project(store: JsonStore, paths: WorkspacePaths, project_id: str) -> ProjectResponse:
    payload = store.read(paths.project_metadata_file(project_id))
    try:
        return ProjectResponse.model_validate(payload)
    except PydanticValidationError as exc:
        raise RuntimeError(f"Invalid project metadata for {project_id}.") from exc


def _load_dataset(store: JsonStore, paths: WorkspacePaths, project_id: str) -> DatasetMetadata:
    payload = store.read(paths.dataset_metadata_file(project_id))
    try:
        return DatasetMetadata.model_validate(payload)
    except PydanticValidationError as exc:
        raise RuntimeError(f"Invalid dataset metadata for {project_id}.") from exc


def _load_experiment(
    store: JsonStore,
    paths: WorkspacePaths,
    project_id: str,
    experiment_id: str,
) -> ExperimentRecord:
    payload = store.read(paths.experiment_metadata_file(project_id, experiment_id))
    try:
        return ExperimentRecord.model_validate(payload)
    except PydanticValidationError as exc:
        raise RuntimeError(f"Invalid experiment metadata for {experiment_id}.") from exc


def _write_experiment(
    store: JsonStore,
    paths: WorkspacePaths,
    project_id: str,
    experiment: ExperimentRecord,
) -> None:
    store.write(
        paths.experiment_metadata_file(project_id, experiment.id),
        experiment.model_dump(mode="json"),
    )


def _upsert_summary(
    store: JsonStore,
    paths: WorkspacePaths,
    project_id: str,
    experiment: ExperimentRecord,
) -> None:
    index_path = paths.experiments_index_file(project_id)
    payload = store.read(
        index_path,
        default=ExperimentsIndex(version=_DEFAULT_VERSION, experiments=[]).model_dump(mode="json"),
    )
    index = ExperimentsIndex.model_validate(payload)
    summary = ExperimentSummary(
        id=experiment.id,
        name=experiment.name,
        created_at=experiment.created_at,
        status=experiment.status,
        best_metric_value=_extract_best_metric_value(experiment.best_metric),
    )

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
    store.write(
        index_path,
        ExperimentsIndex(version=index.version, experiments=updated).model_dump(mode="json"),
    )


def _extract_best_metric_value(best_metric: dict[str, float] | None) -> float | None:
    if not best_metric:
        return None
    for value in best_metric.values():
        return float(value)
    return None


def _load_metrics(
    store: JsonStore,
    paths: WorkspacePaths,
    project_id: str,
    experiment_id: str,
) -> ExperimentMetrics:
    payload = store.read(
        paths.experiment_metrics_file(project_id, experiment_id),
        default={"epochs": []},
    )
    try:
        return ExperimentMetrics.model_validate(payload)
    except PydanticValidationError:
        return ExperimentMetrics(epochs=[])


def _final_metrics(metrics: ExperimentMetrics) -> dict[str, float] | None:
    if not metrics.epochs:
        return None
    last_epoch = metrics.epochs[-1]
    final: dict[str, float] = {}
    for key, value in last_epoch.items():
        if key in {"epoch", "duration_s"}:
            continue
        if isinstance(value, (float, int)):
            final[key] = float(value)
    return final or None


def _best_metric(
    metrics: ExperimentMetrics,
    metric_name: str,
) -> tuple[int | None, dict[str, float] | None]:
    selected_epoch: int | None = None
    selected_value: float | None = None
    should_minimize = _should_minimize(metric_name)

    for epoch_payload in metrics.epochs:
        raw_metric: Any = epoch_payload.get(metric_name)
        if not isinstance(raw_metric, (float, int)):
            continue
        value = float(raw_metric)
        should_replace = False
        if selected_value is None:
            should_replace = True
        elif should_minimize and value < selected_value:
            should_replace = True
        elif not should_minimize and value > selected_value:
            should_replace = True

        if should_replace:
            selected_value = value
            raw_epoch = epoch_payload.get("epoch")
            selected_epoch = int(raw_epoch) if isinstance(raw_epoch, int) else None

    if selected_value is None:
        return None, None
    return selected_epoch, {metric_name: selected_value}


def _should_minimize(metric_name: str) -> bool:
    normalized = metric_name.strip().lower()
    return any(token in normalized for token in ("loss", "error", "mae", "rmse"))
