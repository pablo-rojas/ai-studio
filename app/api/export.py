from __future__ import annotations

import json
from collections.abc import Mapping
from json import JSONDecodeError
from typing import Annotated, Any, TypeVar
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import (
    get_dataset_service,
    get_evaluation_service,
    get_export_service,
    get_training_service,
)
from app.api.export_page_context import build_export_page_context
from app.api.responses import is_hx_request, ok_response
from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.export_service import ExportService
from app.core.training_service import TrainingService
from app.schemas.export import ExportCreate

router = APIRouter()
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
ModelT = TypeVar("ModelT")


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


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def _normalize_export_form_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_payload)
    options = payload.get("options")
    if not isinstance(options, dict):
        options = {}
        payload["options"] = options

    raw_height = payload.pop("input_height", None)
    raw_width = payload.pop("input_width", None)
    if raw_height is not None or raw_width is not None:
        height = raw_height if raw_height is not None else 224
        width = raw_width if raw_width is not None else 224
        options["input_shape"] = [1, 3, height, width]

    dynamic_batch = _coerce_bool(payload.pop("dynamic_batch", None), default=True)
    options["dynamic_axes"] = (
        {"input": {"0": "batch_size"}, "output": {"0": "batch_size"}} if dynamic_batch else None
    )

    if not payload.get("format"):
        payload["format"] = "onnx"
    return payload


def _build_export_push_url(
    project_id: str,
    *,
    selected_experiment_id: str | None,
    selected_export_id: str | None,
) -> str:
    query: dict[str, str] = {}
    if selected_experiment_id:
        query["experiment_id"] = selected_experiment_id
    if selected_export_id:
        query["export_id"] = selected_export_id
    encoded = urlencode(query)
    if not encoded:
        return f"/projects/{project_id}/export"
    return f"/projects/{project_id}/export?{encoded}"


def _render_export_page_fragment(
    request: Request,
    *,
    context: Mapping[str, Any],
):
    templates: Jinja2Templates = request.app.state.templates
    response = templates.TemplateResponse(
        request,
        "fragments/export_page_root.html",
        dict(context),
    )
    response.headers["HX-Push-Url"] = _build_export_push_url(
        str(context["project_id"]),
        selected_experiment_id=(
            str(context["selected_experiment_id"]) if context["selected_experiment_id"] else None
        ),
        selected_export_id=(
            str(context["selected_export_id"]) if context["selected_export_id"] else None
        ),
    )
    return response


def _render_export_page_for_hx(
    request: Request,
    *,
    project_id: str,
    selected_experiment_id: str | None,
    selected_export_id: str | None,
    dataset_service: DatasetService,
    evaluation_service: EvaluationService,
    export_service: ExportService,
    training_service: TrainingService,
):
    context = build_export_page_context(
        project_id=project_id,
        dataset_service=dataset_service,
        evaluation_service=evaluation_service,
        export_service=export_service,
        training_service=training_service,
        selected_experiment_id=selected_experiment_id,
        selected_export_id=selected_export_id,
    )
    return _render_export_page_fragment(request, context=context)


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
                parsed_value = [_coerce_form_scalar(value) for value in values if value != ""]
            else:
                parsed_value = _coerce_form_scalar(values[-1]) if values else ""
            _assign_nested_value(raw_payload, key, parsed_value)
        raw_payload = _normalize_export_form_payload(raw_payload)

    try:
        return model_type.model_validate(raw_payload)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.get("/formats")
async def list_export_formats(
    export_service: ExportServiceDep,
) -> dict[str, object]:
    """List available export formats and availability status."""
    formats = export_service.list_formats()
    return ok_response({"formats": [item.model_dump(mode="json") for item in formats]})


@router.get("/{project_id}")
async def list_exports(
    request: Request,
    project_id: str,
    export_service: ExportServiceDep,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """List exports for one project."""
    if is_hx_request(request):
        return _render_export_page_for_hx(
            request,
            project_id=project_id,
            selected_experiment_id=request.query_params.get("experiment_id"),
            selected_export_id=request.query_params.get("export_id"),
            dataset_service=dataset_service,
            evaluation_service=evaluation_service,
            export_service=export_service,
            training_service=training_service,
        )
    exports = export_service.list_exports(project_id)
    return ok_response({"exports": [item.model_dump(mode="json") for item in exports]})


@router.post("/{project_id}")
async def create_export(
    request: Request,
    project_id: str,
    export_service: ExportServiceDep,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Create and run one export."""
    payload = await _parse_request_model(request, ExportCreate)
    export_record = export_service.create_export(project_id, payload)
    if is_hx_request(request):
        return _render_export_page_for_hx(
            request,
            project_id=project_id,
            selected_experiment_id=export_record.experiment_id,
            selected_export_id=export_record.id,
            dataset_service=dataset_service,
            evaluation_service=evaluation_service,
            export_service=export_service,
            training_service=training_service,
        )
    return ok_response(export_record.model_dump(mode="json"))


@router.get("/{project_id}/{export_id}")
async def get_export(
    request: Request,
    project_id: str,
    export_id: str,
    export_service: ExportServiceDep,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Get one export record by ID."""
    export_record = export_service.get_export(project_id, export_id)
    if is_hx_request(request):
        return _render_export_page_for_hx(
            request,
            project_id=project_id,
            selected_experiment_id=export_record.experiment_id,
            selected_export_id=export_record.id,
            dataset_service=dataset_service,
            evaluation_service=evaluation_service,
            export_service=export_service,
            training_service=training_service,
        )
    return ok_response(export_record.model_dump(mode="json"))


@router.delete("/{project_id}/{export_id}")
async def delete_export(
    request: Request,
    project_id: str,
    export_id: str,
    export_service: ExportServiceDep,
    training_service: TrainingServiceDep,
    dataset_service: DatasetServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> dict[str, object]:
    """Delete one export and its generated files."""
    existing = export_service.get_export(project_id, export_id)
    export_service.delete_export(project_id, export_id)
    if is_hx_request(request):
        return _render_export_page_for_hx(
            request,
            project_id=project_id,
            selected_experiment_id=existing.experiment_id,
            selected_export_id=None,
            dataset_service=dataset_service,
            evaluation_service=evaluation_service,
            export_service=export_service,
            training_service=training_service,
        )
    return ok_response(
        {
            "project_id": project_id,
            "export_id": export_id,
            "deleted": True,
        }
    )


@router.get("/{project_id}/{export_id}/download")
async def download_export(
    project_id: str,
    export_id: str,
    export_service: ExportServiceDep,
) -> FileResponse:
    """Download the exported model artifact."""
    output_path = export_service.resolve_output_file(project_id, export_id)
    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="application/octet-stream",
    )
