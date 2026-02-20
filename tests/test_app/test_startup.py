from __future__ import annotations

from pathlib import Path

import pytest

from app.main import create_app
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


@pytest.mark.asyncio
async def test_startup_creates_workspace_directory(workspace: Path) -> None:
    app = create_app(paths=WorkspacePaths(root=workspace), store=JsonStore())
    assert not workspace.exists()

    async with app.router.lifespan_context(app):
        assert workspace.exists()
        assert (workspace / "projects").exists()
        assert (workspace / "workspace.json").exists()
