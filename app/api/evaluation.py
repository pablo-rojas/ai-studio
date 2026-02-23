from __future__ import annotations

from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import (
    get_dataset_service,
    get_evaluation_service,
    get_split_service,
    get_training_service,
)
from app.api.responses import is_hx_request, ok_response
from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.exceptions import NotFoundError
from app.core.split_service import SplitService
from app.core.training_service import TrainingService
from app.schemas.evaluation import EvaluationConfig, EvaluationResultsQuery

router = APIRouter()
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
SplitServiceDep = Annotated[SplitService, Depends(get_split_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
ModelT = TypeVar("ModelT")
_EVALUATION_SUBSET_ORDER: tuple[str, str, str] = ("test", "val", "train")


async def _parse_request_model(request: Request, model_type: type[ModelT]) -> ModelT:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            raw_payload: Any = await request.json()
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
        raw_payload = {}
        for raw_key in form.keys():
            values = form.getlist(raw_key)
            is_list = raw_key.endswith("[]")
            key = raw_key[:-2] if is_list else raw_key
            if is_list:
                raw_payload[key] = [value for value in values if value != ""]
            else:
                raw_payload[key] = values[-1] if values else ""

    try:
        return model_type.model_validate(raw_payload)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _parse_results_query(request: Request) -> EvaluationResultsQuery:
    raw_params = request.query_params
    query_payload: dict[str, str] = {}
    for field in (
        "page",
        "page_size",
        "sort_by",
        "sort_order",
        "filter_correct",
        "filter_class",
        "filter_subset",
    ):
        value = raw_params.get(field)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned == "":
            continue
        query_payload[field] = cleaned

    try:
        return EvaluationResultsQuery.model_validate(query_payload)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


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


def _build_empty_results_page(query: EvaluationResultsQuery) -> dict[str, object]:
    return {
        "page": query.page,
        "page_size": query.page_size,
        "total_items": 0,
        "total_pages": 0,
        "items": [],
    }


def _extract_primary_class_name(annotations: list[dict[str, Any]]) -> str | None:
    for annotation in annotations:
        if annotation.get("type") == "label":
            class_name = annotation.get("class_name")
            if isinstance(class_name, str) and class_name:
                return class_name

    for annotation in annotations:
        if annotation.get("type") == "bbox":
            class_name = annotation.get("class_name")
            if isinstance(class_name, str) and class_name:
                return class_name

    for annotation in annotations:
        if annotation.get("type") == "anomaly":
            return "anomalous" if bool(annotation.get("is_anomalous")) else "normal"

    return None


def _build_split_assignments(
    *,
    split_names: list[str],
    split_values: list[str],
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for index, split_name in enumerate(split_names):
        if index < len(split_values):
            assignments.append({"name": split_name, "value": split_values[index]})
    return assignments


def _build_workspace_context(
    *,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationService,
    dataset_service: DatasetService,
    split_service: SplitService,
    training_service: TrainingService,
    query: EvaluationResultsQuery | None = None,
    results_page_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    experiment_model = training_service.get_experiment(project_id, experiment_id)
    experiment = experiment_model.model_dump(mode="json")
    available_devices = training_service.list_available_devices()
    default_device = _resolve_default_device_id(available_devices)
    checkpoints = evaluation_service.list_checkpoints(project_id, experiment_id)
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
    query_model = query or EvaluationResultsQuery()
    results_query = query_model.model_dump(mode="json")
    results_page = results_page_payload or _build_empty_results_page(query_model)
    evaluation: dict[str, object] | None = None
    aggregate: dict[str, object] | None = None
    class_options: list[str] = []

    try:
        dataset = dataset_service.get_dataset(project_id)
        class_options = list(dataset.classes)
    except NotFoundError:
        class_options = []

    try:
        record = evaluation_service.get_evaluation(project_id, experiment_id)
        evaluation = record.model_dump(mode="json")
        evaluation_config = EvaluationConfig(
            checkpoint=record.checkpoint,
            split_subsets=list(record.split_subsets),
            batch_size=record.batch_size,
            device=record.device,
        ).model_dump(mode="json")
        aggregate_model = evaluation_service.get_aggregate_metrics(project_id, experiment_id)
        aggregate = aggregate_model.model_dump(mode="json") if aggregate_model is not None else None
        if results_page_payload is None:
            results_page = evaluation_service.get_results(
                project_id,
                experiment_id,
                query_model,
            ).model_dump(mode="json")
    except NotFoundError:
        evaluation = None

    return {
        "project_id": project_id,
        "experiment": experiment,
        "available_devices": available_devices,
        "checkpoints": checkpoints,
        "subset_options": subset_options,
        "evaluation": evaluation,
        "evaluation_config": evaluation_config,
        "aggregate": aggregate,
        "results_query": results_query,
        "results_page": results_page,
        "class_options": class_options,
    }


def _render_workspace_fragment(
    request: Request,
    *,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationService,
    dataset_service: DatasetService,
    split_service: SplitService,
    training_service: TrainingService,
    query: EvaluationResultsQuery | None = None,
    results_page_payload: dict[str, object] | None = None,
):
    templates: Jinja2Templates = request.app.state.templates
    context = _build_workspace_context(
        project_id=project_id,
        experiment_id=experiment_id,
        evaluation_service=evaluation_service,
        dataset_service=dataset_service,
        split_service=split_service,
        training_service=training_service,
        query=query,
        results_page_payload=results_page_payload,
    )
    return templates.TemplateResponse(request, "fragments/evaluation_detail.html", context)


def _render_results_fragment(
    request: Request,
    *,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationService,
    dataset_service: DatasetService,
    split_service: SplitService,
    training_service: TrainingService,
    query: EvaluationResultsQuery,
    results_page_payload: dict[str, object],
):
    templates: Jinja2Templates = request.app.state.templates
    context = _build_workspace_context(
        project_id=project_id,
        experiment_id=experiment_id,
        evaluation_service=evaluation_service,
        dataset_service=dataset_service,
        split_service=split_service,
        training_service=training_service,
        query=query,
        results_page_payload=results_page_payload,
    )
    return templates.TemplateResponse(request, "fragments/evaluation_results_grid.html", context)


def _render_result_detail_fragment(
    request: Request,
    context: dict[str, object],
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "fragments/evaluation_result_detail.html", context)


@router.post("/{project_id}/{experiment_id}")
async def start_evaluation(
    request: Request,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Start evaluation for an experiment."""
    payload = await _parse_request_model(request, EvaluationConfig)
    record = evaluation_service.start_evaluation(project_id, experiment_id, payload)
    if is_hx_request(request):
        return _render_workspace_fragment(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            evaluation_service=evaluation_service,
            dataset_service=dataset_service,
            split_service=split_service,
            training_service=training_service,
        )
    return ok_response({"status": record.status})


@router.get("/{project_id}/{experiment_id}")
async def get_evaluation(
    request: Request,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Return evaluation metadata and aggregate metrics for an experiment."""
    if is_hx_request(request):
        return _render_workspace_fragment(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            evaluation_service=evaluation_service,
            dataset_service=dataset_service,
            split_service=split_service,
            training_service=training_service,
        )
    record = evaluation_service.get_evaluation(project_id, experiment_id)
    aggregate = evaluation_service.get_aggregate_metrics(project_id, experiment_id)
    return ok_response(
        {
            "evaluation": record.model_dump(mode="json"),
            "aggregate": aggregate.model_dump(mode="json") if aggregate is not None else None,
        }
    )


@router.delete("/{project_id}/{experiment_id}")
async def reset_evaluation(
    request: Request,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Reset evaluation by deleting the experiment evaluation folder."""
    evaluation_service.reset_evaluation(project_id, experiment_id)
    if is_hx_request(request):
        return _render_workspace_fragment(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            evaluation_service=evaluation_service,
            dataset_service=dataset_service,
            split_service=split_service,
            training_service=training_service,
        )
    return ok_response(
        {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "deleted": True,
        }
    )


@router.get("/{project_id}/{experiment_id}/results")
async def get_evaluation_results(
    request: Request,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
    dataset_service: DatasetServiceDep,
    split_service: SplitServiceDep,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Return paginated per-image evaluation results."""
    query = _parse_results_query(request)
    page = evaluation_service.get_results(project_id, experiment_id, query)
    payload = page.model_dump(mode="json")
    if is_hx_request(request):
        return _render_results_fragment(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            evaluation_service=evaluation_service,
            dataset_service=dataset_service,
            split_service=split_service,
            training_service=training_service,
            query=query,
            results_page_payload=payload,
        )
    return ok_response(payload)


@router.get("/{project_id}/{experiment_id}/results/{filename}/info")
async def get_evaluation_result_info(
    request: Request,
    project_id: str,
    experiment_id: str,
    filename: str,
    evaluation_service: EvaluationServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Return one per-image evaluation result with dataset metadata."""
    query = _parse_results_query(request)
    result = evaluation_service.get_result(project_id, experiment_id, filename)
    image = dataset_service.get_image_info(project_id, result.filename)
    dataset = dataset_service.get_dataset(project_id)
    page = evaluation_service.get_results(project_id, experiment_id, query)

    previous_filename: str | None = None
    next_filename: str | None = None
    for index, item in enumerate(page.items):
        if item.filename != result.filename:
            continue
        if index > 0:
            previous_filename = page.items[index - 1].filename
        if index + 1 < len(page.items):
            next_filename = page.items[index + 1].filename
        break

    result_payload = result.model_dump(mode="json")
    image_payload = image.model_dump(mode="json")
    split_assignments = _build_split_assignments(
        split_names=list(dataset.split_names),
        split_values=list(image.split),
    )
    dataset_class_name = _extract_primary_class_name(image_payload["annotations"])
    context = {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "result": result_payload,
        "image": image_payload,
        "split_assignments": split_assignments,
        "dataset_class_name": dataset_class_name,
        "results_query": query.model_dump(mode="json"),
        "current_page": page.page,
        "previous_filename": previous_filename,
        "next_filename": next_filename,
    }

    if is_hx_request(request):
        return _render_result_detail_fragment(request, context)

    return ok_response(
        {
            "result": result_payload,
            "image": image_payload,
            "dataset": {
                "class_name": dataset_class_name,
                "annotation_count": len(image_payload["annotations"]),
                "split_assignments": split_assignments,
            },
            "navigation": {
                "page": page.page,
                "previous_filename": previous_filename,
                "next_filename": next_filename,
            },
        }
    )


@router.get("/{project_id}/{experiment_id}/checkpoints")
async def list_evaluation_checkpoints(
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """List physically available checkpoint files for one experiment."""
    checkpoints = evaluation_service.list_checkpoints(project_id, experiment_id)
    return ok_response({"checkpoints": checkpoints})
