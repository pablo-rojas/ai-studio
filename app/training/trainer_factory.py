from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from app.config import get_settings
from app.schemas.training import TrainingConfig
from app.training.callbacks import GracefulStopCallback, JSONMetricLogger, SSENotifier


@dataclass(frozen=True, slots=True)
class ResolvedHardware:
    """Resolved trainer hardware arguments."""

    accelerator: str
    devices: int | list[int] | str
    strategy: str


def resolve_hardware(
    selected_devices: Sequence[str] | None,
    *,
    default_device: str | None = None,
) -> ResolvedHardware:
    """Resolve user-selected devices into Lightning trainer args."""
    requested = _normalize_device_labels(selected_devices)
    if not requested:
        requested = [default_device or get_settings().default_device]

    gpu_indices = _extract_gpu_indices(requested)
    if gpu_indices and torch.cuda.is_available():
        return ResolvedHardware(
            accelerator="gpu",
            devices=gpu_indices,
            strategy="auto",
        )

    return ResolvedHardware(
        accelerator="cpu",
        devices=1,
        strategy="auto",
    )


def build_trainer(
    *,
    experiment_dir: Path,
    training_config: TrainingConfig,
    selected_devices: Sequence[str] | None = None,
    default_device: str | None = None,
    metrics_file: Path | None = None,
    monitor_metric: str = "val_accuracy",
    enable_progress_bar: bool = True,
    publish_event: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> pl.Trainer:
    """Build a Lightning trainer configured from training settings."""
    hyperparameters = training_config.hyperparameters
    resolved_hardware = resolve_hardware(
        selected_devices=selected_devices,
        default_device=default_device,
    )

    checkpoints_dir = experiment_dir / "checkpoints"
    metrics_path = metrics_file or (experiment_dir / "metrics.json")
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="best",
            monitor=monitor_metric,
            mode=_resolve_monitor_mode(monitor_metric),
            save_last=True,
            save_top_k=1,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        JSONMetricLogger(output_path=metrics_path),
    ]

    if hyperparameters.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=hyperparameters.early_stopping_patience,
                mode="min",
            )
        )
    if publish_event is not None:
        callbacks.append(SSENotifier(publish_event))
    if stop_requested is not None:
        callbacks.append(GracefulStopCallback(stop_requested))

    return pl.Trainer(
        default_root_dir=str(experiment_dir),
        max_epochs=hyperparameters.max_epochs,
        accumulate_grad_batches=hyperparameters.batch_multiplier,
        accelerator=resolved_hardware.accelerator,
        devices=resolved_hardware.devices,
        strategy=resolved_hardware.strategy,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )


def _resolve_monitor_mode(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    if "loss" in normalized or "mae" in normalized or "rmse" in normalized:
        return "min"
    return "max"


def _normalize_device_labels(selected_devices: Sequence[str] | None) -> list[str]:
    if not selected_devices:
        return []
    normalized: list[str] = []
    for item in selected_devices:
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        normalized.append(cleaned)
    return normalized


def _extract_gpu_indices(labels: Sequence[str]) -> list[int]:
    indices: set[int] = set()
    for label in labels:
        if label in {"gpu", "cuda"}:
            indices.add(0)
            continue
        if label.startswith(("gpu:", "cuda:")):
            _, _, raw_index = label.partition(":")
            if raw_index.isdigit():
                indices.add(int(raw_index))
    return sorted(indices)


__all__ = ["ResolvedHardware", "build_trainer", "resolve_hardware"]
