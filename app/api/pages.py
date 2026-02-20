from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.dependencies import get_dataset_service, get_project_service
from app.core.dataset_service import DatasetService
from app.core.exceptions import NotFoundError
from app.core.project_service import ProjectService
from app.schemas.dataset import DatasetImageListQuery

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]


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
    dataset_service: DatasetServiceDep,
):
    """Render the dataset browsing page for an existing project."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)
    query = DatasetImageListQuery()

    dataset: dict[str, object] | None = None
    listing: dict[str, object] | None = None

    try:
        dataset_model = dataset_service.get_dataset(project_id)
        listing_model = dataset_service.list_images(project_id, query)
    except NotFoundError:
        dataset_model = None
        listing_model = None

    if dataset_model is not None:
        dataset = dataset_model.model_dump(mode="json")
    if listing_model is not None:
        listing = listing_model.model_dump(mode="json")

    return templates.TemplateResponse(
        request,
        "pages/dataset.html",
        {
            "project": project,
            "project_id": project_id,
            "dataset": dataset,
            "listing": listing,
            "query": query.model_dump(mode="json"),
            "active_page": "dataset",
        },
    )
