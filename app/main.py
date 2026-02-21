from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import datasets, evaluation, pages, projects, splits, training
from app.config import Settings, get_settings
from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.exceptions import AIStudioError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.storage.json_store import JsonStore
from app.storage.paths import WorkspacePaths

logger = logging.getLogger(__name__)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


def _render_hx_error(
    request: Request,
    *,
    status_code: int,
    message: str,
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/error_toast.html",
        {"error_message": message},
        status_code=status_code,
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AIStudioError)
    async def aistudio_error_handler(request: Request, exc: AIStudioError):
        if request.headers.get("HX-Request"):
            return _render_hx_error(request, status_code=exc.status_code, message=exc.message)
        return _error_response(
            status_code=exc.status_code,
            code=exc.error_code,
            message=exc.message,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        message = "Request validation failed."
        if request.headers.get("HX-Request"):
            return _render_hx_error(request, status_code=422, message=message)
        return _error_response(
            status_code=422,
            code="VALIDATION_ERROR",
            message=message,
            details=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, _: Exception):
        logger.exception("Unhandled exception while processing request.")
        message = "An unexpected error occurred."
        if request.headers.get("HX-Request"):
            return _render_hx_error(request, status_code=500, message=message)
        return _error_response(
            status_code=500,
            code="INTERNAL_ERROR",
            message=message,
        )


def create_app(
    *,
    paths: WorkspacePaths | None = None,
    store: JsonStore | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings: Settings = get_settings()
    workspace_paths = paths or WorkspacePaths.from_settings()
    json_store = store or JsonStore()

    project_service = ProjectService(paths=workspace_paths, store=json_store)
    dataset_service = DatasetService(
        paths=workspace_paths,
        store=json_store,
        project_service=project_service,
    )
    split_service = SplitService(
        paths=workspace_paths,
        store=json_store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    training_service = TrainingService(
        paths=workspace_paths,
        store=json_store,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    evaluation_service = EvaluationService(
        paths=workspace_paths,
        store=json_store,
        project_service=project_service,
        dataset_service=dataset_service,
        training_service=training_service,
    )

    app_root = Path(__file__).resolve().parent
    templates_dir = app_root / "templates"
    static_dir = app_root / "static"
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        project_service.initialize_workspace()
        yield

    app = FastAPI(title="AI Studio", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.project_service = project_service
    app.state.dataset_service = dataset_service
    app.state.split_service = split_service
    app.state.training_service = training_service
    app.state.evaluation_service = evaluation_service
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(splits.router, prefix="/api/splits", tags=["splits"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
    app.include_router(pages.router, tags=["pages"])

    _register_exception_handlers(app)
    return app


app = create_app()
