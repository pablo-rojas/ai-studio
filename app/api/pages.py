from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.datasets import build_class_badge_map
from app.api.dependencies import get_dataset_service, get_project_service, get_split_service
from app.core.dataset_service import DatasetService
from app.core.exceptions import NotFoundError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.schemas.dataset import DatasetImageListQuery

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
SplitServiceDep = Annotated[SplitService, Depends(get_split_service)]


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
    query = DatasetImageListQuery(split_name=request.query_params.get("split_name"))

    dataset: dict[str, object] | None = None
    listing: dict[str, object] | None = None
    query_payload = query.model_dump(mode="json")

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
        query_payload["split_name"] = listing_model.selected_split_name
    else:
        query_payload["split_name"] = ""
    class_badge_map = build_class_badge_map(dataset_model.classes if dataset_model else [])

    return templates.TemplateResponse(
        request,
        "pages/dataset.html",
        {
            "project": project,
            "project_id": project_id,
            "dataset": dataset,
            "listing": listing,
            "query": query_payload,
            "class_badge_map": class_badge_map,
            "active_page": "dataset",
        },
    )


@router.get("/projects/{project_id}/split", include_in_schema=False)
async def split_page(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
):
    """Render the split management page for an existing project."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)

    dataset: dict[str, object] | None = None
    splits: list[dict[str, object]] = []

    try:
        dataset_model = dataset_service.get_dataset(project_id)
    except NotFoundError:
        dataset_model = None

    if dataset_model is not None:
        dataset = dataset_model.model_dump(mode="json")
        splits = [split.model_dump(mode="json") for split in split_service.list_splits(project_id)]

    return templates.TemplateResponse(
        request,
        "pages/split.html",
        {
            "project": project,
            "project_id": project_id,
            "dataset": dataset,
            "splits": splits,
            "active_page": "split",
        },
    )
