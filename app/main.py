from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.project_service import ProjectService
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths


def create_app(
    *,
    paths: WorkspacePaths | None = None,
    store: JsonStore | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    project_service = ProjectService(paths=paths, store=store)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        project_service.initialize_workspace()
        yield

    return FastAPI(title="AI Studio", version="0.1.0", lifespan=lifespan)


app = create_app()
