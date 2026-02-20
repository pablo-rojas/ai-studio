from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.project_service import ProjectService
from app.main import create_app
from app.schemas.project import ProjectCreate, ProjectResponse
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide a clean temporary workspace root."""
    return tmp_path / "workspace"


@pytest.fixture
def project_service(workspace: Path) -> ProjectService:
    """Provide a project service bound to a temporary workspace."""
    return ProjectService(paths=WorkspacePaths(root=workspace), store=JsonStore())


@pytest_asyncio.fixture
async def test_client(workspace: Path) -> AsyncClient:
    """Provide an httpx AsyncClient pointing at the test app."""
    app = create_app(paths=WorkspacePaths(root=workspace), store=JsonStore())
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def sample_project(project_service: ProjectService) -> ProjectResponse:
    """Create and return a sample project."""
    return project_service.create_project(
        ProjectCreate(name="Sample Project", task="classification")
    )


@pytest.fixture
def sample_dataset(
    sample_project: ProjectResponse,
    project_service: ProjectService,
) -> dict[str, Path]:
    """Provide a placeholder dataset fixture for future phases."""
    return {
        "project_id": sample_project.id,
        "dataset_dir": project_service.paths.dataset_dir(sample_project.id),
    }
