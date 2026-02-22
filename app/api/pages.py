from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.datasets import build_class_badge_map
from app.api.dependencies import (
    get_dataset_service,
    get_evaluation_service,
    get_export_service,
    get_project_service,
    get_split_service,
    get_training_service,
)
from app.api.export_page_context import build_export_page_context
from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.exceptions import NotFoundError
from app.core.export_service import ExportService
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_form_layout import TrainingSection
from app.core.training_service import TrainingService
from app.schemas.dataset import DatasetImageListQuery
from app.schemas.evaluation import EvaluationConfig, EvaluationResultsQuery

router = APIRouter()
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
SplitServiceDep = Annotated[SplitService, Depends(get_split_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]

_CLASSIFICATION_BACKBONES: tuple[tuple[str, str], ...] = (
    ("resnet18", "ResNet-18"),
    ("resnet34", "ResNet-34"),
    ("resnet50", "ResNet-50"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("efficientnet_b3", "EfficientNet-B3"),
    ("mobilenet_v3_small", "MobileNetV3-Small"),
    ("mobilenet_v3_large", "MobileNetV3-Large"),
)
_EVALUATION_SUBSET_ORDER: tuple[str, str, str] = ("test", "val", "train")


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


def _build_evaluation_subset_options(
    split_service: SplitService,
    *,
    project_id: str,
    split_name: str,
) -> list[dict[str, object]]:
    try:
        split_summary = split_service.get_split(project_id, split_name)
    except NotFoundError:
        split_summary = None

    if split_summary is None:
        return [
            {"name": subset, "label": subset.title(), "count": None}
            for subset in _EVALUATION_SUBSET_ORDER
        ]

    options: list[dict[str, object]] = []
    for subset in _EVALUATION_SUBSET_ORDER:
        count = int(getattr(split_summary.stats, subset))
        if count <= 0:
            continue
        options.append({"name": subset, "label": subset.title(), "count": count})
    if options:
        return options
    return [
        {"name": subset, "label": subset.title(), "count": None}
        for subset in _EVALUATION_SUBSET_ORDER
    ]


def _resolve_default_evaluation_subsets(subset_options: list[dict[str, object]]) -> list[str]:
    available = [str(option["name"]) for option in subset_options]
    if "test" in available:
        return ["test"]
    if available:
        return [available[0]]
    return ["test"]


def _resolve_default_device_id(available_devices: list[dict[str, object]]) -> str:
    for option in available_devices:
        candidate = option.get("id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return "cpu"


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


@router.get("/projects/{project_id}/evaluation", include_in_schema=False)
async def evaluation_page(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
    training_service: TrainingServiceDep,
    evaluation_service: EvaluationServiceDep,
):
    """Render the experiment-scoped evaluation page."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)

    dataset: dict[str, object] | None = None
    class_options: list[str] = []
    experiments: list[dict[str, object]] = []
    completed_experiments: list[dict[str, object]] = []
    selected_experiment_id = request.query_params.get("experiment_id")
    selected_experiment: dict[str, object] | None = None
    available_devices = training_service.list_available_devices()
    default_device = _resolve_default_device_id(available_devices)
    checkpoints: list[str] = []
    subset_options: list[dict[str, object]] = [
        {"name": subset, "label": subset.title(), "count": None}
        for subset in _EVALUATION_SUBSET_ORDER
    ]
    evaluation: dict[str, object] | None = None
    aggregate: dict[str, object] | None = None
    results_query = EvaluationResultsQuery().model_dump(mode="json")
    results_page = {
        "page": 1,
        "page_size": 50,
        "total_items": 0,
        "total_pages": 0,
        "items": [],
    }
    evaluation_config = EvaluationConfig(
        checkpoint="best",
        split_subsets=["test"],
        batch_size=32,
        device=default_device,
    ).model_dump(mode="json")

    try:
        dataset_model = dataset_service.get_dataset(project_id)
    except NotFoundError:
        dataset_model = None

    if dataset_model is not None:
        dataset = dataset_model.model_dump(mode="json")
        class_options = list(dataset_model.classes)
        experiments = [
            experiment.model_dump(mode="json")
            for experiment in training_service.list_experiments(project_id)
        ]
        completed_experiments = [
            experiment for experiment in experiments if experiment.get("status") == "completed"
        ]
        completed_ids = {str(experiment["id"]) for experiment in completed_experiments}
        if selected_experiment_id not in completed_ids:
            selected_experiment_id = None
        if selected_experiment_id is None and completed_experiments:
            selected_experiment_id = str(completed_experiments[0]["id"])

        if selected_experiment_id is not None:
            experiment_model = training_service.get_experiment(project_id, selected_experiment_id)
            if experiment_model.status == "completed":
                selected_experiment = experiment_model.model_dump(mode="json")
                checkpoints = evaluation_service.list_checkpoints(
                    project_id,
                    selected_experiment_id,
                )
                subset_options = _build_evaluation_subset_options(
                    split_service,
                    project_id=project_id,
                    split_name=experiment_model.split_name,
                )
                default_subsets = _resolve_default_evaluation_subsets(subset_options)
                default_checkpoint = checkpoints[0] if checkpoints else "best"
                evaluation_config = EvaluationConfig(
                    checkpoint=default_checkpoint,
                    split_subsets=default_subsets,
                    batch_size=32,
                    device=default_device,
                ).model_dump(mode="json")

                try:
                    record = evaluation_service.get_evaluation(project_id, selected_experiment_id)
                    evaluation = record.model_dump(mode="json")
                    evaluation_config = EvaluationConfig(
                        checkpoint=record.checkpoint,
                        split_subsets=list(record.split_subsets),
                        batch_size=record.batch_size,
                        device=record.device,
                    ).model_dump(mode="json")
                    aggregate_model = evaluation_service.get_aggregate_metrics(
                        project_id,
                        selected_experiment_id,
                    )
                    aggregate = (
                        aggregate_model.model_dump(mode="json")
                        if aggregate_model is not None
                        else None
                    )
                    results_page = evaluation_service.get_results(
                        project_id,
                        selected_experiment_id,
                        EvaluationResultsQuery(),
                    ).model_dump(mode="json")
                except NotFoundError:
                    evaluation = None

    return templates.TemplateResponse(
        request,
        "pages/evaluation.html",
        {
            "project": project,
            "project_id": project_id,
            "dataset": dataset,
            "class_options": class_options,
            "experiments": experiments,
            "completed_experiments": completed_experiments,
            "selected_experiment": selected_experiment,
            "selected_experiment_id": selected_experiment_id,
            "available_devices": available_devices,
            "checkpoints": checkpoints,
            "subset_options": subset_options,
            "evaluation": evaluation,
            "evaluation_config": evaluation_config,
            "aggregate": aggregate,
            "results_query": results_query,
            "results_page": results_page,
            "active_page": "evaluation",
        },
    )


@router.get("/projects/{project_id}/export", include_in_schema=False)
async def export_page(
    request: Request,
    project_id: str,
    project_service: ProjectServiceDep,
    dataset_service: DatasetServiceDep,
    evaluation_service: EvaluationServiceDep,
    export_service: ExportServiceDep,
    training_service: TrainingServiceDep,
):
    """Render the model export page for an existing project."""
    templates: Jinja2Templates = request.app.state.templates
    project = project_service.get_project(project_id)
    selected_experiment_id = request.query_params.get("experiment_id")
    selected_export_id = request.query_params.get("export_id")

    context = build_export_page_context(
        project_id=project_id,
        dataset_service=dataset_service,
        evaluation_service=evaluation_service,
        export_service=export_service,
        training_service=training_service,
        selected_experiment_id=selected_experiment_id,
        selected_export_id=selected_export_id,
    )

    return templates.TemplateResponse(
        request,
        "pages/export.html",
        {
            "project": project,
            "active_page": "export",
            **context,
        },
    )
