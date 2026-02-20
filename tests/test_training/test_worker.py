from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import app.training.worker as worker
from app.models.catalog import build_default_training_config
from app.schemas.training import ExperimentMetrics, ExperimentRecord


class _NeverStopEvent:
    def is_set(self) -> bool:
        return False


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root

    def dataset_images_dir(self, project_id: str) -> Path:
        del project_id
        path = self.root / "dataset" / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        path = self.root / project_id / "experiments" / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def experiment_metrics_file(self, project_id: str, experiment_id: str) -> Path:
        return self.experiment_dir(project_id, experiment_id) / "metrics.json"

    def experiment_checkpoints_dir(self, project_id: str, experiment_id: str) -> Path:
        path = self.experiment_dir(project_id, experiment_id) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path


class _FakeModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeDataModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True


class _FakeTrainer:
    def __init__(self) -> None:
        self.fit_kwargs: dict[str, object] | None = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


def test_worker_calls_trainer_fit_with_model_keyword(tmp_path: Path, monkeypatch) -> None:
    defaults = build_default_training_config("classification", backbone="resnet18")
    experiment = ExperimentRecord(
        id="exp-12345678",
        name="Fit Keyword",
        created_at=datetime.now(timezone.utc),
        split_name="80-10-10",
        model=defaults.model,
        hyperparameters=defaults.hyperparameters.model_copy(update={"max_epochs": 1}),
        augmentations=defaults.augmentations,
    )

    fake_trainer = _FakeTrainer()

    monkeypatch.setattr(worker, "WorkspacePaths", _FakePaths)
    monkeypatch.setattr(worker, "JsonStore", lambda: SimpleNamespace())
    monkeypatch.setattr(
        worker,
        "_load_project",
        lambda *args, **kwargs: SimpleNamespace(task="classification"),
    )
    monkeypatch.setattr(
        worker,
        "_load_dataset",
        lambda *args, **kwargs: SimpleNamespace(classes=["cat", "dog"]),
    )
    monkeypatch.setattr(worker, "_load_experiment", lambda *args, **kwargs: experiment)
    monkeypatch.setattr(worker, "_write_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_upsert_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "create_model", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(worker, "build_loss", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(worker, "AIStudioModule", _FakeModule)
    monkeypatch.setattr(worker, "AIStudioDataModule", _FakeDataModule)
    monkeypatch.setattr(
        worker,
        "_load_metrics",
        lambda *args, **kwargs: ExperimentMetrics(epochs=[{"epoch": 1, "val_accuracy": 0.82}]),
    )
    monkeypatch.setattr(worker, "build_trainer", lambda *args, **kwargs: fake_trainer)

    worker.run_experiment_training(
        _NeverStopEvent(),
        str(tmp_path),
        "proj-12345678",
        experiment.id,
        False,
    )

    assert fake_trainer.fit_kwargs is not None
    assert "model" in fake_trainer.fit_kwargs
    assert "module" not in fake_trainer.fit_kwargs
    assert fake_trainer.fit_kwargs["ckpt_path"] is None
    assert isinstance(fake_trainer.fit_kwargs["datamodule"], _FakeDataModule)
