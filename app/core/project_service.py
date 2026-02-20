from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.schemas.project import (
    ProjectCreate,
    ProjectRename,
    ProjectResponse,
    WorkspaceMetadata,
)
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)

_PROJECT_ID_PREFIX = "proj"
_DEFAULT_VERSION = "1.0"
_EMPTY_EXPERIMENTS_INDEX: dict[str, Any] = {"version": _DEFAULT_VERSION, "experiments": []}
_EMPTY_EVALUATIONS_INDEX: dict[str, Any] = {"version": _DEFAULT_VERSION, "evaluations": []}
_EMPTY_EXPORTS_INDEX: dict[str, Any] = {"version": _DEFAULT_VERSION, "exports": []}


def _utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class ProjectService:
    """Service for project lifecycle management."""

    def __init__(
        self,
        *,
        paths: WorkspacePaths | None = None,
        store: JsonStore | None = None,
    ) -> None:
        self.paths = paths or WorkspacePaths.from_settings()
        self.store = store or JsonStore()

    def initialize_workspace(self) -> None:
        """Ensure the workspace folder and workspace metadata exist."""
        self.paths.ensure_workspace_layout()
        if self.paths.workspace_metadata_exists():
            return

        metadata = WorkspaceMetadata(
            version=_DEFAULT_VERSION,
            created_at=_utc_now(),
            projects=[],
        )
        self.store.write(self.paths.workspace_metadata_file(), metadata.model_dump(mode="json"))

    def create_project(self, payload: ProjectCreate) -> ProjectResponse:
        """Create a project and persist all metadata."""
        self.initialize_workspace()

        project_id = self._generate_project_id()
        timestamp = _utc_now()
        project = ProjectResponse(
            id=project_id,
            name=payload.name,
            task=payload.task,
            description=payload.description,
            created_at=timestamp,
            updated_at=timestamp,
        )

        self.paths.ensure_project_layout(project_id)
        self._initialize_project_indexes(project_id)
        self.store.write(
            self.paths.project_metadata_file(project_id),
            project.model_dump(mode="json"),
        )
        self._add_project_to_workspace(project_id)
        return project

    def list_projects(self) -> list[ProjectResponse]:
        """List all projects from workspace metadata."""
        self.initialize_workspace()
        workspace = self._load_workspace_metadata()
        projects: list[ProjectResponse] = []

        for project_id in workspace.projects:
            try:
                projects.append(self.get_project(project_id))
            except NotFoundError:
                logger.warning("Skipping missing project metadata for id=%s", project_id)

        return projects

    def get_project(self, project_id: str) -> ProjectResponse:
        """Return a single project's metadata."""
        self.initialize_workspace()
        try:
            project_data = self.store.read(self.paths.project_metadata_file(project_id))
        except FileNotFoundError as exc:
            raise NotFoundError(f"Project {project_id} not found.") from exc
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        try:
            return ProjectResponse.model_validate(project_data)
        except PydanticValidationError as exc:
            raise ValidationError(f"Project metadata is invalid for {project_id}.") from exc

    def rename_project(self, project_id: str, payload: ProjectRename) -> ProjectResponse:
        """Rename an existing project."""
        project = self.get_project(project_id)
        updated_project = project.model_copy(
            update={"name": payload.name, "updated_at": _utc_now()}
        )
        self.store.write(
            self.paths.project_metadata_file(project_id),
            updated_project.model_dump(mode="json"),
        )
        return updated_project

    def delete_project(self, project_id: str) -> None:
        """Delete a project and remove it from workspace metadata."""
        self.get_project(project_id)
        self.paths.remove_project(project_id)
        self._remove_project_from_workspace(project_id)

    def _generate_project_id(self) -> str:
        for _ in range(20):
            candidate = f"{_PROJECT_ID_PREFIX}-{uuid.uuid4().hex[:8]}"
            if not self.paths.project_exists(candidate):
                return candidate
        raise ConflictError("Unable to generate a unique project ID.")

    def _load_workspace_metadata(self) -> WorkspaceMetadata:
        workspace_file = self.paths.workspace_metadata_file()
        workspace_data = self.store.read(workspace_file)
        try:
            return WorkspaceMetadata.model_validate(workspace_data)
        except PydanticValidationError as exc:
            raise ValidationError("Workspace metadata is invalid.") from exc

    def _add_project_to_workspace(self, project_id: str) -> None:
        workspace_file = self.paths.workspace_metadata_file()

        def update_workspace(payload: Any) -> dict[str, Any]:
            metadata = WorkspaceMetadata.model_validate(payload)
            if project_id not in metadata.projects:
                metadata.projects.append(project_id)
            return metadata.model_dump(mode="json")

        self.store.update(
            workspace_file,
            update_workspace,
            default_factory=lambda: WorkspaceMetadata(
                version=_DEFAULT_VERSION, created_at=_utc_now(), projects=[]
            ).model_dump(mode="json"),
        )

    def _remove_project_from_workspace(self, project_id: str) -> None:
        workspace_file = self.paths.workspace_metadata_file()

        def update_workspace(payload: Any) -> dict[str, Any]:
            metadata = WorkspaceMetadata.model_validate(payload)
            metadata.projects = [
                existing_id for existing_id in metadata.projects if existing_id != project_id
            ]
            return metadata.model_dump(mode="json")

        self.store.update(
            workspace_file,
            update_workspace,
            default_factory=lambda: WorkspaceMetadata(
                version=_DEFAULT_VERSION, created_at=_utc_now(), projects=[]
            ).model_dump(mode="json"),
        )

    def _initialize_project_indexes(self, project_id: str) -> None:
        self.store.update(
            self.paths.experiments_index_file(project_id),
            lambda payload: payload,
            default_factory=lambda: _EMPTY_EXPERIMENTS_INDEX.copy(),
        )
        self.store.update(
            self.paths.evaluations_index_file(project_id),
            lambda payload: payload,
            default_factory=lambda: _EMPTY_EVALUATIONS_INDEX.copy(),
        )
        self.store.update(
            self.paths.exports_index_file(project_id),
            lambda payload: payload,
            default_factory=lambda: _EMPTY_EXPORTS_INDEX.copy(),
        )
