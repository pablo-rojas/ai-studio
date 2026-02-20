from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import NotFoundError
from app.core.project_service import ProjectService
from app.schemas.project import ProjectCreate, ProjectRename
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


def test_workspace_is_initialized_on_first_use(
    project_service: ProjectService, workspace: Path
) -> None:
    assert not workspace.exists()

    projects = project_service.list_projects()

    assert projects == []
    assert workspace.exists()
    assert (workspace / "projects").exists()
    assert (workspace / "workspace.json").exists()


def test_project_crud_round_trip(project_service: ProjectService) -> None:
    created = project_service.create_project(
        ProjectCreate(
            name="PCB Defect Classifier",
            task="classification",
            description="PCB defect model",
        )
    )
    assert created.id.startswith("proj-")
    assert len(created.id) == len("proj-") + 8
    assert created.name == "PCB Defect Classifier"
    assert created.task == "classification"
    assert created.description == "PCB defect model"

    listed_projects = project_service.list_projects()
    assert [project.id for project in listed_projects] == [created.id]

    renamed = project_service.rename_project(
        created.id,
        ProjectRename(name="PCB Defect Classifier v2"),
    )
    assert renamed.id == created.id
    assert renamed.name == "PCB Defect Classifier v2"
    assert renamed.updated_at >= created.updated_at

    fetched = project_service.get_project(created.id)
    assert fetched.name == "PCB Defect Classifier v2"

    project_service.delete_project(created.id)
    assert project_service.list_projects() == []

    with pytest.raises(NotFoundError):
        project_service.get_project(created.id)


def test_project_data_persists_across_service_instances(workspace: Path) -> None:
    first_service = ProjectService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    created = first_service.create_project(
        ProjectCreate(name="Persistent Project", task="classification")
    )

    second_service = ProjectService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    listed_projects = second_service.list_projects()

    assert len(listed_projects) == 1
    assert listed_projects[0].id == created.id
    assert listed_projects[0].name == "Persistent Project"
