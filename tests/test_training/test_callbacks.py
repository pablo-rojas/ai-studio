from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.training.callbacks import GracefulStopCallback, JSONMetricLogger, SSENotifier


class _FakeTrainer:
    def __init__(self) -> None:
        self.sanity_checking = False
        self.current_epoch = 0
        self.max_epochs = 5
        self.should_stop = False
        self.callback_metrics = {
            "train_loss": torch.tensor(1.23),
            "val_loss": torch.tensor(0.98),
            "val_accuracy": torch.tensor(0.75),
            "confusion_matrix": torch.tensor([[1, 0], [0, 1]]),
        }
        self.optimizers = [SimpleNamespace(param_groups=[{"lr": 0.001}])]
        self.num_val_batches = 1


class _FakeModule:
    pass


def test_json_metric_logger_writes_epoch_data(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    trainer = _FakeTrainer()
    callback = JSONMetricLogger(output_path=output_path)

    callback.on_fit_start(trainer, _FakeModule())
    callback.on_train_epoch_start(trainer, _FakeModule())
    callback.on_validation_epoch_end(trainer, _FakeModule())

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "epochs" in payload
    assert len(payload["epochs"]) == 1
    first_epoch = payload["epochs"][0]
    assert first_epoch["epoch"] == 1
    assert first_epoch["train_loss"] == pytest.approx(1.23)
    assert first_epoch["val_loss"] == pytest.approx(0.98)
    assert first_epoch["val_accuracy"] == pytest.approx(0.75)
    assert "confusion_matrix" not in first_epoch
    assert first_epoch["lr"] == 0.001
    assert "duration_s" in first_epoch


def test_sse_notifier_emits_expected_events() -> None:
    trainer = _FakeTrainer()
    events: list[dict[str, object]] = []
    callback = SSENotifier(events.append)

    callback.on_train_start(trainer, _FakeModule())
    callback.on_validation_epoch_end(trainer, _FakeModule())
    callback.on_train_end(trainer, _FakeModule())

    assert [event["type"] for event in events] == [
        "training_started",
        "epoch_end",
        "training_complete",
    ]


def test_graceful_stop_callback_sets_should_stop() -> None:
    trainer = _FakeTrainer()
    should_stop = {"value": False}

    def _stop_requested() -> bool:
        return should_stop["value"]

    callback = GracefulStopCallback(stop_requested=_stop_requested)
    callback.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=0)
    assert trainer.should_stop is False

    should_stop["value"] = True
    callback.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=1)
    assert trainer.should_stop is True
