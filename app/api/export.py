from __future__ import annotations

from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_export_service
from app.api.responses import ok_response
from app.core.export_service import ExportService
from app.schemas.export import ExportCreate

router = APIRouter()
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
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
        raw_payload = dict(form)

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
    project_id: str,
    export_service: ExportServiceDep,
) -> dict[str, object]:
    """List exports for one project."""
    exports = export_service.list_exports(project_id)
    return ok_response({"exports": [item.model_dump(mode="json") for item in exports]})


@router.post("/{project_id}")
async def create_export(
    request: Request,
    project_id: str,
    export_service: ExportServiceDep,
) -> dict[str, object]:
    """Create and run one export."""
    payload = await _parse_request_model(request, ExportCreate)
    export_record = export_service.create_export(project_id, payload)
    return ok_response(export_record.model_dump(mode="json"))


@router.get("/{project_id}/{export_id}")
async def get_export(
    project_id: str,
    export_id: str,
    export_service: ExportServiceDep,
) -> dict[str, object]:
    """Get one export record by ID."""
    export_record = export_service.get_export(project_id, export_id)
    return ok_response(export_record.model_dump(mode="json"))


@router.delete("/{project_id}/{export_id}")
async def delete_export(
    project_id: str,
    export_id: str,
    export_service: ExportServiceDep,
) -> dict[str, object]:
    """Delete one export and its generated files."""
    export_service.delete_export(project_id, export_id)
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
