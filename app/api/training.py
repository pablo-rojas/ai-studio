from __future__ import annotations

import asyncio
import json
from json import JSONDecodeError
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_training_service
from app.api.responses import is_hx_request, ok_response
from app.core.training_service import TrainingService
from app.schemas.training import ExperimentCreate, ExperimentUpdate

router = APIRouter()
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
ModelT = TypeVar("ModelT")
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_STREAM_FINAL_STATUSES = _TERMINAL_STATUSES | {"created"}


def _render_experiment_list_fragment(request: Request, *, project_id: str, experiments):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/training_experiment_list.html",
        {"project_id": project_id, "experiments": experiments},
    )


def _render_experiment_detail_fragment(request: Request, *, experiment):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/training_experiment_detail.html",
        {"experiment": experiment},
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
        return _render_experiment_list_fragment(request, project_id=project_id, experiments=payload)
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
        )
    return ok_response(serialized)


@router.get("/{project_id}/experiments/{experiment_id}")
async def get_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Return one experiment configuration."""
    experiment = training_service.get_experiment(project_id, experiment_id)
    payload = experiment.model_dump(mode="json")
    if is_hx_request(request):
        return _render_experiment_detail_fragment(request, experiment=payload)
    return ok_response(payload)


@router.patch("/{project_id}/experiments/{experiment_id}")
async def update_experiment(
    request: Request,
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Patch editable experiment configuration fields."""
    payload = await _parse_request_model(request, ExperimentUpdate)
    experiment = training_service.update_experiment(project_id, experiment_id, payload)
    serialized = experiment.model_dump(mode="json")
    if is_hx_request(request):
        return _render_experiment_detail_fragment(request, experiment=serialized)
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
        return _render_experiment_list_fragment(request, project_id=project_id, experiments=payload)
    return ok_response({"project_id": project_id, "experiment_id": experiment_id, "deleted": True})


@router.post("/{project_id}/experiments/{experiment_id}/train")
async def start_training(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Start experiment training in a subprocess."""
    experiment = training_service.start_training(project_id, experiment_id)
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/stop")
async def stop_training(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Request graceful training cancellation."""
    experiment = training_service.stop_training(project_id, experiment_id)
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/resume")
async def resume_training(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Resume training from last checkpoint."""
    experiment = training_service.resume_training(project_id, experiment_id)
    return ok_response({"experiment_id": experiment.id, "status": experiment.status})


@router.post("/{project_id}/experiments/{experiment_id}/restart")
async def restart_experiment(
    project_id: str,
    experiment_id: str,
    training_service: TrainingServiceDep,
) -> dict[str, object]:
    """Reset experiment artifacts and status back to `created`."""
    experiment = training_service.restart_experiment(project_id, experiment_id)
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
