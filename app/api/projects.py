from __future__ import annotations

from json import JSONDecodeError
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_project_service
from app.api.responses import is_hx_request, ok_response
from app.core.project_service import ProjectService
from app.schemas.project import ProjectCreate, ProjectRename

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
ModelT = TypeVar("ModelT")


def _render_project_list_fragment(
    request: Request,
    project_service: ProjectService,
):
    projects = project_service.list_projects()
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/project_list.html",
        {"projects": projects},
    )


async def _parse_request_model(request: Request, model_type: type[ModelT]) -> ModelT:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            raw_payload = await request.json()
        except JSONDecodeError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body",),
                        "msg": "JSON decode error.",
                        "input": None,
                    }
                ]
            ) from exc
    else:
        form = await request.form()
        raw_payload = dict(form)

    try:
        return model_type.model_validate(raw_payload)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


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
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Create a new project."""
    payload = await _parse_request_model(request, ProjectCreate)
    project = project_service.create_project(payload)
    if is_hx_request(request):
        response = _render_project_list_fragment(request, project_service)
        response.headers["HX-Redirect"] = f"/projects/{project.id}/dataset"
        return response
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
    project_service: ProjectServiceDep,
) -> dict[str, object]:
    """Rename an existing project."""
    payload = await _parse_request_model(request, ProjectRename)
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
