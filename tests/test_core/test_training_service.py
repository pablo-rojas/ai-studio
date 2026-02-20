from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from app.core.dataset_service import DatasetService
from app.core.exceptions import DatasetNotImportedError, TrainingInProgressError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImportRequest
from app.schemas.project import ProjectCreate
from app.schemas.split import SplitCreateRequest, SplitRatios
from app.schemas.training import ExperimentCreate, ExperimentUpdate
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


@dataclass
class _FakeStopEvent:
    value: bool = False

    def set(self) -> None:
        self.value = True

    def is_set(self) -> bool:
        return self.value


class _FakeHandle:
    def __init__(self) -> None:
        self.process = SimpleNamespace(exitcode=0)
        self.stop_event = _FakeStopEvent()
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def request_stop(self) -> None:
        self.stop_event.set()
        self._alive = False

    def wait(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self._alive = False


class _FakeRunner:
    def __init__(self) -> None:
        self.active: _FakeHandle | None = None
        self.starts: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def start(self, target, *args: Any, **kwargs: Any) -> _FakeHandle:
        if self.active is not None and self.active.is_alive():
            raise RuntimeError("A training process is already running.")
        handle = _FakeHandle()
        self.active = handle
        self.starts.append((target, args, kwargs))
        return handle

    def get_active(self) -> _FakeHandle | None:
        if self.active is None:
            return None
        if self.active.is_alive():
            return self.active
        self.active = None
        return None

    def request_stop(self) -> None:
        if self.active is not None:
            self.active.request_stop()


def test_training_service_crud_and_metrics_defaults(workspace: Path) -> None:
    source_root = workspace.parent / "training_service_source"
    _build_classification_source(source_root)

    paths = WorkspacePaths(root=workspace)
    store = JsonStore()
    project_service = ProjectService(paths=paths, store=store)
    dataset_service = DatasetService(paths=paths, store=store, project_service=project_service)
    split_service = SplitService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    training_service = TrainingService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
        runner=_FakeRunner(),
    )

    project = project_service.create_project(
        ProjectCreate(name="Training Core Project", task="classification")
    )
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )
    split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="80-10-10",
            ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
            seed=42,
        ),
    )

    created = training_service.create_experiment(
        project.id,
        ExperimentCreate(name="Baseline"),
    )
    assert created.id.startswith("exp-")
    assert created.status == "created"
    assert created.split_name == "80-10-10"

    listed = training_service.list_experiments(project.id)
    assert [item.id for item in listed] == [created.id]
    assert listed[0].name == "Baseline"

    updated = training_service.update_experiment(
        project.id,
        created.id,
        payload=ExperimentUpdate(name="Baseline Updated"),
    )
    assert updated.name == "Baseline Updated"

    metrics = training_service.get_metrics(project.id, created.id)
    assert metrics.epochs == []

    training_service.delete_experiment(project.id, created.id)
    assert training_service.list_experiments(project.id) == []


def test_training_service_start_enforces_single_active_run(workspace: Path, monkeypatch) -> None:
    source_root = workspace.parent / "training_service_start_source"
    _build_classification_source(source_root)

    paths = WorkspacePaths(root=workspace)
    store = JsonStore()
    project_service = ProjectService(paths=paths, store=store)
    dataset_service = DatasetService(paths=paths, store=store, project_service=project_service)
    split_service = SplitService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    runner = _FakeRunner()
    training_service = TrainingService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=dataset_service,
        runner=runner,
    )

    monkeypatch.setattr(training_service, "_monitor_active_training", lambda *args, **kwargs: None)

    project = project_service.create_project(
        ProjectCreate(name="Training Start Project", task="classification")
    )
    dataset_service.import_dataset(
        project.id,
        DatasetImportRequest(source_path=str(source_root), source_format="image_folders"),
    )
    split_service.create_split(
        project.id,
        SplitCreateRequest(
            name="80-10-10",
            ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
            seed=7,
        ),
    )

    first = training_service.create_experiment(project.id, ExperimentCreate(name="First"))
    second = training_service.create_experiment(project.id, ExperimentCreate(name="Second"))

    pending = training_service.start_training(project.id, first.id)
    assert pending.status == "pending"
    assert len(runner.starts) == 1

    with pytest.raises(TrainingInProgressError):
        training_service.start_training(project.id, second.id)

    training_service.stop_training(project.id, first.id)


def test_training_service_requires_imported_dataset(workspace: Path) -> None:
    paths = WorkspacePaths(root=workspace)
    store = JsonStore()
    project_service = ProjectService(paths=paths, store=store)
    training_service = TrainingService(
        paths=paths,
        store=store,
        project_service=project_service,
        dataset_service=DatasetService(paths=paths, store=store, project_service=project_service),
        runner=_FakeRunner(),
    )

    project = project_service.create_project(
        ProjectCreate(name="No Dataset Training Project", task="classification")
    )

    with pytest.raises(DatasetNotImportedError):
        training_service.create_experiment(project.id)


def _build_classification_source(source_root: Path) -> None:
    cats_dir = source_root / "cats"
    dogs_dir = source_root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for index in range(20):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(220, 120, 120))
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 220))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)
