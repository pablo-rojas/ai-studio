from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from app.storage.json_store import JsonStore

logger = logging.getLogger(__name__)


class JSONMetricLogger(pl.Callback):
    """Persist per-epoch training metrics to `metrics.json`."""

    def __init__(
        self,
        output_path: Path,
        *,
        store: JsonStore | None = None,
    ) -> None:
        self.output_path = output_path
        self.store = store or JsonStore()
        self._epochs: list[dict[str, Any]] = []
        self._epoch_started_at: float | None = None
        self._last_written_epoch = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Load existing metrics (resume) and ensure output file exists."""
        del trainer
        del pl_module
        payload = self.store.read(self.output_path, default={"epochs": []})
        raw_epochs: Any = payload.get("epochs", []) if isinstance(payload, dict) else []
        self._epochs = [item for item in raw_epochs if isinstance(item, dict)]
        if self._epochs:
            last_epoch = self._epochs[-1].get("epoch")
            if isinstance(last_epoch, int):
                self._last_written_epoch = last_epoch
        self._write_metrics()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start a timer for epoch duration tracking."""
        del trainer
        del pl_module
        self._epoch_started_at = time.perf_counter()

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Write one metrics record once validation finishes."""
        del pl_module
        if trainer.sanity_checking:
            return
        self._append_epoch_metrics(trainer)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Write metrics for runs without validation dataloaders."""
        del pl_module
        if trainer.sanity_checking:
            return
        if _has_validation_batches(trainer):
            return
        self._append_epoch_metrics(trainer)

    def _append_epoch_metrics(self, trainer: pl.Trainer) -> None:
        epoch = trainer.current_epoch + 1
        if epoch <= self._last_written_epoch:
            return

        epoch_data: dict[str, Any] = {"epoch": epoch}
        epoch_data.update(_extract_scalar_metrics(trainer.callback_metrics))

        learning_rate = _extract_learning_rate(trainer)
        if learning_rate is not None:
            epoch_data["lr"] = learning_rate

        duration_s = _compute_duration_s(self._epoch_started_at)
        if duration_s is not None:
            epoch_data["duration_s"] = duration_s

        self._epochs.append(epoch_data)
        self._last_written_epoch = epoch
        self._write_metrics()

    def _write_metrics(self) -> None:
        self.store.write(self.output_path, {"epochs": self._epochs})


class SSENotifier(pl.Callback):
    """Emit training events through a user-provided publisher callback."""

    def __init__(self, publish_event: Callable[[dict[str, Any]], None]) -> None:
        self.publish_event = publish_event

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._publish(
            {
                "type": "training_started",
                "epoch": trainer.current_epoch + 1,
                "max_epochs": trainer.max_epochs,
            }
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        self._publish(
            {
                "type": "epoch_end",
                "epoch": trainer.current_epoch + 1,
                "max_epochs": trainer.max_epochs,
                "metrics": _extract_scalar_metrics(trainer.callback_metrics),
            }
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        del pl_module
        self._publish({"type": "training_complete"})

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer
        del pl_module
        self._publish({"type": "training_error", "message": str(exception)})

    def _publish(self, payload: dict[str, Any]) -> None:
        try:
            self.publish_event(payload)
        except Exception:
            logger.exception("Failed to publish SSE training event.")


class GracefulStopCallback(pl.Callback):
    """Stop training gracefully when an external stop flag is set."""

    def __init__(self, stop_requested: Callable[[], bool]) -> None:
        self.stop_requested = stop_requested
        self._stop_triggered = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module
        del outputs
        del batch
        del batch_idx
        if self._stop_triggered:
            return
        if self.stop_requested():
            self._stop_triggered = True
            trainer.should_stop = True


def _extract_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = _to_float(value)
        if numeric is None:
            continue
        parsed[key] = numeric
    return parsed


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    return None


def _extract_learning_rate(trainer: pl.Trainer) -> float | None:
    if not trainer.optimizers:
        return None
    param_groups = trainer.optimizers[0].param_groups
    if not param_groups:
        return None
    raw = param_groups[0].get("lr")
    if not isinstance(raw, (float, int)):
        return None
    return float(raw)


def _compute_duration_s(epoch_started_at: float | None) -> float | None:
    if epoch_started_at is None:
        return None
    return round(time.perf_counter() - epoch_started_at, 6)


def _has_validation_batches(trainer: pl.Trainer) -> bool:
    raw = trainer.num_val_batches
    if isinstance(raw, int):
        return raw > 0
    if isinstance(raw, list):
        return sum(raw) > 0
    return bool(raw)


__all__ = ["GracefulStopCallback", "JSONMetricLogger", "SSENotifier"]
