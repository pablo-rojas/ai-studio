from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from app.models.catalog import build_default_training_config
from app.training.callbacks import JSONMetricLogger
from app.training.trainer_factory import build_trainer, resolve_hardware


def test_resolve_hardware_cpu_selection() -> None:
    resolved = resolve_hardware(["cpu"])

    assert resolved.accelerator == "cpu"
    assert resolved.devices == 1
    assert resolved.strategy == "auto"


def test_resolve_hardware_gpu_selection_falls_back_without_cuda() -> None:
    resolved = resolve_hardware(["gpu:0"])

    if torch.cuda.is_available():
        assert resolved.accelerator == "gpu"
        assert resolved.devices == [0]
    else:
        assert resolved.accelerator == "cpu"
        assert resolved.devices == 1


def test_build_trainer_registers_expected_callbacks(tmp_path: Path) -> None:
    training_config = build_default_training_config("classification", backbone="resnet18")
    trainer = build_trainer(
        experiment_dir=tmp_path,
        training_config=training_config,
        selected_devices=["cpu"],
        enable_progress_bar=False,
    )

    assert trainer.max_epochs == training_config.hyperparameters.max_epochs
    assert trainer.accumulate_grad_batches == training_config.hyperparameters.batch_multiplier
    assert any(isinstance(callback, ModelCheckpoint) for callback in trainer.callbacks)
    assert any(isinstance(callback, LearningRateMonitor) for callback in trainer.callbacks)
    assert any(isinstance(callback, JSONMetricLogger) for callback in trainer.callbacks)
    assert any(isinstance(callback, EarlyStopping) for callback in trainer.callbacks)
