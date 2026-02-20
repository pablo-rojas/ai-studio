from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.dependencies import get_project_service
from app.core.project_service import ProjectService

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]


@router.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect the application root to the project page."""
    return RedirectResponse(url="/projects", status_code=307)


@router.get("/projects", include_in_schema=False)
async def projects_page(
    request: Request,
    project_service: ProjectServiceDep,
):
    """Render the project management page."""
    templates: Jinja2Templates = request.app.state.templates
    projects = project_service.list_projects()
    return templates.TemplateResponse(
        request,
        "pages/projects.html",
        {
            "projects": projects,
            "active_page": "projects",
            "project": None,
        },
    )


@router.get("/projects/{project_id}/dataset", include_in_schema=False)
async def dataset_page(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
):
    """Render a minimal dataset page placeholder for project navigation."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)
    return templates.TemplateResponse(
        request,
        "pages/dataset.html",
        {
            "project": project,
            "active_page": "dataset",
        },
    )
