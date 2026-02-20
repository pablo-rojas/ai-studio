from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.core.dataset_service import DatasetService
from app.core.project_service import ProjectService
from app.main import create_app
from app.schemas.dataset import DatasetImportRequest, DatasetMetadata
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
    workspace: Path,
) -> DatasetMetadata:
    """Import and return a small sample dataset."""
    source_root = workspace.parent / "source_dataset"
    cat_dir = source_root / "cat"
    dog_dir = source_root / "dog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    dog_dir.mkdir(parents=True, exist_ok=True)

    _create_test_image(cat_dir / "cat_1.png", size=(240, 180), color=(220, 120, 120))
    _create_test_image(dog_dir / "dog_1.png", size=(300, 200), color=(120, 120, 220))

    service = DatasetService(paths=WorkspacePaths(root=workspace), store=JsonStore())
    return service.import_dataset(
        sample_project.id,
        DatasetImportRequest(
            source_path=str(source_root),
            source_format="image_folders",
        ),
    )


def _create_test_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size, color)
    image.save(path)
