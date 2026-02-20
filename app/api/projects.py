from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from app.api.dependencies import get_project_service
from app.api.responses import is_hx_request, ok_response
from app.core.project_service import ProjectService
from app.schemas.project import ProjectCreate, ProjectRename

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]


def _render_project_list_fragment(
    request: Request,
    project_service: ProjectService,
):
    projects = project_service.list_projects()
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        "fragments/project_list.html",
        {"request": request, "projects": projects},
    )


@router.get("")
async def list_projects(
    request: Request,
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """List all projects."""
    projects = project_service.list_projects()
    if is_hx_request(request):
        return _render_project_list_fragment(request, project_service)
    return ok_response({"projects": [project.model_dump(mode="json") for project in projects]})


@router.post("")
async def create_project(
    request: Request,
    payload: ProjectCreate,
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Create a new project."""
    project = project_service.create_project(payload)
    if is_hx_request(request):
        return _render_project_list_fragment(request, project_service)
    return ok_response(project.model_dump(mode="json"))


@router.get("/{project_id}")
async def get_project(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Return one project by ID."""
    project = project_service.get_project(project_id)
    if is_hx_request(request):
        return _render_project_list_fragment(request, project_service)
    return ok_response(project.model_dump(mode="json"))


@router.patch("/{project_id}")
async def rename_project(
    request: Request,
    project_id: str,
    payload: ProjectRename,
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Rename an existing project."""
    project = project_service.rename_project(project_id, payload)
    if is_hx_request(request):
        return _render_project_list_fragment(request, project_service)
    return ok_response(project.model_dump(mode="json"))


@router.delete("/{project_id}")
async def delete_project(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Delete a project."""
    project_service.delete_project(project_id)
    if is_hx_request(request):
        return _render_project_list_fragment(request, project_service)
    return ok_response({"project_id": project_id, "deleted": True})
