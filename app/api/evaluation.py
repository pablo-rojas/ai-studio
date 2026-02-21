from __future__ import annotations

from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_evaluation_service
from app.api.responses import ok_response
from app.core.evaluation_service import EvaluationService
from app.schemas.evaluation import EvaluationConfig, EvaluationResultsQuery

router = APIRouter()
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
ModelT = TypeVar("ModelT")


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


@router.post("/{project_id}/{experiment_id}")
async def start_evaluation(
    request: Request,
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Start evaluation for an experiment."""
    payload = await _parse_request_model(request, EvaluationConfig)
    record = evaluation_service.start_evaluation(project_id, experiment_id, payload)
    return ok_response({"status": record.status})


@router.get("/{project_id}/{experiment_id}")
async def get_evaluation(
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Return evaluation metadata and aggregate metrics for an experiment."""
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
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Reset evaluation by deleting the experiment evaluation folder."""
    evaluation_service.reset_evaluation(project_id, experiment_id)
    return ok_response(
        {
            "project_id": project_id,
            "experiment_id": experiment_id,
            "deleted": True,
        }
    )


@router.get("/{project_id}/{experiment_id}/results")
async def get_evaluation_results(
    project_id: str,
    experiment_id: str,
    query: Annotated[EvaluationResultsQuery, Depends()],
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Return paginated per-image evaluation results."""
    page = evaluation_service.get_results(project_id, experiment_id, query)
    return ok_response(page.model_dump(mode="json"))


@router.get("/{project_id}/{experiment_id}/checkpoints")
async def list_evaluation_checkpoints(
    project_id: str,
    experiment_id: str,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """List physically available checkpoint files for one experiment."""
    checkpoints = evaluation_service.list_checkpoints(project_id, experiment_id)
    return ok_response({"checkpoints": checkpoints})
