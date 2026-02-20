from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.datasets import build_class_badge_map
from app.api.dependencies import (
    get_dataset_service,
    get_project_service,
    get_split_service,
    get_training_service,
)
from app.core.dataset_service import DatasetService
from app.core.exceptions import NotFoundError
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_form_layout import TrainingSection
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImageListQuery

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
SplitServiceDep = Annotated[SplitService, Depends(get_split_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]

_CLASSIFICATION_BACKBONES: tuple[tuple[str, str], ...] = (
    ("resnet18", "ResNet-18"),
    ("resnet34", "ResNet-34"),
    ("resnet50", "ResNet-50"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("efficientnet_b3", "EfficientNet-B3"),
    ("mobilenet_v3_small", "MobileNetV3-Small"),
    ("mobilenet_v3_large", "MobileNetV3-Large"),
)


def _serialize_training_sections(
    sections: tuple[TrainingSection, ...],
) -> dict[str, dict[str, object]]:
    return {
        section.id: {
            "title": section.title,
            "description": section.description,
            "default_open": section.default_open,
        }
        for section in sections
    }


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


@router.get("/projects/{project_id}/training", include_in_schema=False)
async def training_page(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
    dataset_service: DatasetServiceDep,
    training_service: TrainingServiceDep,
):
    """Render the experiment configuration and training page."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)

    dataset: dict[str, object] | None = None
    split_names: list[str] = []
    experiments: list[dict[str, object]] = []
    selected_experiment: dict[str, object] | None = None
    selected_metrics: dict[str, object] = {"epochs": []}
    selected_experiment_id = request.query_params.get("experiment_id")
    training_sections = _serialize_training_sections(
        training_service.get_training_form_sections(project.task)
    )
    available_devices = training_service.list_available_devices()

    try:
        dataset_model = dataset_service.get_dataset(project_id)
    except NotFoundError:
        dataset_model = None

    if dataset_model is not None:
        dataset = dataset_model.model_dump(mode="json")
        split_names = list(dataset_model.split_names)
        experiments = [
            experiment.model_dump(mode="json")
            for experiment in training_service.list_experiments(project_id)
        ]

        if selected_experiment_id is None and experiments:
            selected_experiment_id = str(experiments[0]["id"])

        if selected_experiment_id is not None:
            try:
                experiment_model = training_service.get_experiment(
                    project_id,
                    selected_experiment_id,
                )
                selected_experiment = experiment_model.model_dump(mode="json")
                selected_metrics = training_service.get_metrics(
                    project_id,
                    selected_experiment_id,
                ).model_dump(mode="json")
            except NotFoundError:
                selected_experiment_id = None

    return templates.TemplateResponse(
        request,
        "pages/training.html",
        {
            "project": project,
            "project_id": project_id,
            "dataset": dataset,
            "split_names": split_names,
            "experiments": experiments,
            "selected_experiment": selected_experiment,
            "selected_experiment_id": selected_experiment_id,
            "selected_metrics": selected_metrics,
            "backbone_options": _CLASSIFICATION_BACKBONES,
            "training_sections": training_sections,
            "available_devices": available_devices,
            "active_page": "training",
        },
    )
