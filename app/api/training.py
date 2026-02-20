from __future__ import annotations

import asyncio
import json
from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_dataset_service, get_training_service
from app.api.responses import is_hx_request, ok_response
from app.core.dataset_service import DatasetService
from app.core.exceptions import NotFoundError
from app.core.training_service import TrainingService
from app.schemas.training import ExperimentCreate, ExperimentUpdate

router = APIRouter()
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
ModelT = TypeVar("ModelT")
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_STREAM_FINAL_STATUSES = _TERMINAL_STATUSES | {"created"}
_CLASSIFICATION_BACKBONES: tuple[tuple[str, str], ...] = (
    ("resnet18", "ResNet-18"),
    ("resnet34", "ResNet-34"),
    ("resnet50", "ResNet-50"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("efficientnet_b3", "EfficientNet-B3"),
    ("mobilenet_v3_small", "MobileNetV3-Small"),
    ("mobilenet_v3_large", "MobileNetV3-Large"),
)


def _render_experiment_list_fragment(
    request: Request,
    *,
    project_id: str,
    experiments,
    selected_experiment_id: str | None = None,
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/training_experiment_list.html",
        {
            "project_id": project_id,
            "experiments": experiments,
            "selected_experiment_id": selected_experiment_id,
        },
    )


def _render_experiment_detail_fragment(
    request: Request,
    *,
    project_id: str,
    experiment,
    metrics,
    split_names: list[str],
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/training_experiment_detail.html",
        {
            "project_id": project_id,
            "experiment": experiment,
            "metrics": metrics,
            "split_names": split_names,
            "backbone_options": _CLASSIFICATION_BACKBONES,
        },
    )


def _assign_nested_value(payload: dict[str, Any], key: str, value: Any) -> None:
    cursor = payload
    parts = [part for part in key.split(".") if part]
    if not parts:
        return

    for part in parts[:-1]:
        existing = cursor.get(part)
        if isinstance(existing, dict):
            cursor = existing
            continue
        nested: dict[str, Any] = {}
        cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


def _coerce_form_scalar(raw_value: str) -> Any:
    stripped = raw_value.strip()
    if not stripped:
        return raw_value
    if stripped[0] in {"{", "["}:
        try:
            return json.loads(stripped)
        except JSONDecodeError:
            return raw_value
    return raw_value


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
        raw_payload: dict[str, Any] = {}

        for raw_key in form.keys():
            values = form.getlist(raw_key)
            is_list = raw_key.endswith("[]")
            key = raw_key[:-2] if is_list else raw_key
            if is_list:
                parsed_value = [_coerce_form_scalar(value) for value in values if value != ""]
            else:
                parsed_value = _coerce_form_scalar(values[-1]) if values else ""
            _assign_nested_value(raw_payload, key, parsed_value)

    try:
        return model_type.model_validate(raw_payload)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_split_names(
    *,
    dataset_service: DatasetService,
    project_id: str,
    fallback_split_name: str,
) -> list[str]:
    try:
        dataset = dataset_service.get_dataset(project_id)
    except NotFoundError:
        return [fallback_split_name]

    split_names = list(dataset.split_names)
    if fallback_split_name not in split_names:
        split_names.append(fallback_split_name)
    return split_names


def _render_workspace_for_hx(
    request: Request,
    *,
    project_id: str,
    experiment_id: str,
    training_service: TrainingService,
    dataset_service: DatasetService,
):
    experiment = training_service.get_experiment(project_id, experiment_id)
    metrics = training_service.get_metrics(project_id, experiment_id)
    split_names = _get_split_names(
        dataset_service=dataset_service,
        project_id=project_id,
        fallback_split_name=experiment.split_name,
    )
    return _render_experiment_detail_fragment(
        request,
        project_id=project_id,
        experiment=experiment.model_dump(mode="json"),
        metrics=metrics.model_dump(mode="json"),
        split_names=split_names,
    )


def _sse_event(event: str, payload: dict[str, object]) -> str:
    body = json.dumps(payload, ensure_ascii=True)
    return f"event: {event}\ndata: {body}\n\n"


@router.get("/{project_id}/experiments")
async def list_experiments(
    request: Request,
    project_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """List experiments for a project."""
    experiments = training_service.list_experiments(project_id)
    payload = [experiment.model_dump(mode="json") for experiment in experiments]
    if is_hx_request(request):
        return _render_experiment_list_fragment(
            request,
            project_id=project_id,
            experiments=payload,
            selected_experiment_id=request.query_params.get("selected_experiment_id"),
        )
    return ok_response({"experiments": payload})


@router.post("/{project_id}/experiments")
async def create_experiment(
    request: Request,
    project_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Create a new experiment."""
    payload = await _parse_request_model(request, ExperimentCreate)
    experiment = training_service.create_experiment(project_id, payload)
    serialized = experiment.model_dump(mode="json")
    if is_hx_request(request):
        experiments = training_service.list_experiments(project_id)
        list_payload = [item.model_dump(mode="json") for item in experiments]
        return _render_experiment_list_fragment(
            request,
            project_id=project_id,
            experiments=list_payload,
            selected_experiment_id=serialized["id"],
        )
    return ok_response(serialized)


@router.get("/{project_id}/experiments/{experiment_id}")
async def get_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Return one experiment configuration."""
    experiment = training_service.get_experiment(project_id, experiment_id)
    payload = experiment.model_dump(mode="json")
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response(payload)


@router.patch("/{project_id}/experiments/{experiment_id}")
async def update_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Patch editable experiment configuration fields."""
    payload = await _parse_request_model(request, ExperimentUpdate)
    experiment = training_service.update_experiment(project_id, experiment_id, payload)
    serialized = experiment.model_dump(mode="json")
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response(serialized)


@router.delete("/{project_id}/experiments/{experiment_id}")
async def delete_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Delete an experiment and all generated artifacts."""
    training_service.delete_experiment(project_id, experiment_id)
    if is_hx_request(request):
        experiments = training_service.list_experiments(project_id)
        payload = [item.model_dump(mode="json") for item in experiments]
        return _render_experiment_list_fragment(
            request,
            project_id=project_id,
            experiments=payload,
            selected_experiment_id=request.query_params.get("selected_experiment_id"),
        )
    return ok_response({"project_id": project_id, "experiment_id": experiment_id, "deleted": True})


@router.post("/{project_id}/experiments/{experiment_id}/train")
async def start_training(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Start experiment training in a subprocess."""
    experiment = training_service.start_training(project_id, experiment_id)
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/stop")
async def stop_training(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Request graceful training cancellation."""
    experiment = training_service.stop_training(project_id, experiment_id)
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/resume")
async def resume_training(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Resume training from last checkpoint."""
    experiment = training_service.resume_training(project_id, experiment_id)
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/restart")
async def restart_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Reset experiment artifacts and status back to `created`."""
    experiment = training_service.restart_experiment(project_id, experiment_id)
    if is_hx_request(request):
        return _render_workspace_for_hx(
            request,
            project_id=project_id,
            experiment_id=experiment_id,
            training_service=training_service,
            dataset_service=dataset_service,
        )
    return ok_response(experiment.model_dump(mode="json"))


@router.get("/{project_id}/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Return per-epoch metrics for one experiment."""
    metrics = training_service.get_metrics(project_id, experiment_id)
    return ok_response(metrics.model_dump(mode="json"))


@router.get("/{project_id}/experiments/{experiment_id}/stream")
async def training_stream(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> StreamingResponse:
    """Stream experiment status and epoch metrics as SSE events."""

    async def event_generator():
        last_epoch_count = 0
        last_status: str | None = None
        try:
            while True:
                experiment = training_service.get_experiment(project_id, experiment_id)
                metrics = training_service.get_metrics(project_id, experiment_id)

                if experiment.status != last_status:
                    last_status = experiment.status
                    yield _sse_event(
                        "status",
                        {
                            "project_id": project_id,
                            "experiment_id": experiment_id,
                            "status": experiment.status,
                        },
                    )

                epochs = metrics.epochs
                while last_epoch_count < len(epochs):
                    epoch_payload = dict(epochs[last_epoch_count])
                    last_epoch_count += 1
                    yield _sse_event("epoch_end", epoch_payload)

                if experiment.status in _STREAM_FINAL_STATUSES and last_epoch_count >= len(epochs):
                    yield _sse_event(
                        "complete",
                        {
                            "project_id": project_id,
                            "experiment_id": experiment_id,
                            "status": experiment.status,
                        },
                    )
                    break

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
