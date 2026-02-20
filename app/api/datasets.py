from __future__ import annotations

import zipfile
from json import JSONDecodeError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_dataset_service
from app.api.responses import is_hx_request, ok_response
from app.core.dataset_service import DatasetService
from app.core.exceptions import ValidationError
from app.schemas.dataset import (
    DatasetImageListQuery,
    DatasetImageSortBy,
    DatasetImportRequest,
    DatasetSourceFormat,
    SortOrder,
)

router = APIRouter()
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
ModelT = TypeVar("ModelT")

_CHUNK_SIZE_BYTES = 1024 * 1024


def _render_dataset_summary_fragment(request: Request, dataset: dict[str, Any]):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/dataset_summary.html",
        {"dataset": dataset},
    )


def _render_dataset_images_fragment(
    request: Request,
    *,
    project_id: str,
    listing: dict[str, Any],
    query: dict[str, Any],
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/dataset_image_list.html",
        {
            "project_id": project_id,
            "listing": listing,
            "query": query,
        },
    )


def _render_dataset_image_detail_fragment(
    request: Request,
    *,
    project_id: str,
    image: dict[str, Any],
    class_name: str | None,
    split_assignments: list[dict[str, str]],
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/dataset_image_detail.html",
        {
            "project_id": project_id,
            "image": image,
            "class_name": class_name,
            "split_assignments": split_assignments,
        },
    )


def _extract_zip_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValidationError("Uploaded ZIP contains unsafe file paths.")
    archive.extractall(destination)


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


def _extract_primary_class_name(annotations: list[dict[str, Any]]) -> str | None:
    for annotation in annotations:
        if annotation.get("type") == "label":
            class_name = annotation.get("class_name")
            if isinstance(class_name, str) and class_name:
                return class_name

    for annotation in annotations:
        if annotation.get("type") == "anomaly":
            return "anomalous" if bool(annotation.get("is_anomalous")) else "normal"

    return None


@router.get("/{project_id}")
async def get_dataset(
    request: Request,
    project_id: str,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Return dataset metadata for one project."""
    dataset = dataset_service.get_dataset(project_id)
    payload = dataset.model_dump(mode="json")
    if is_hx_request(request):
        return _render_dataset_summary_fragment(request, payload)
    return ok_response(payload)


@router.post("/{project_id}/import/local")
async def import_dataset_from_local(
    request: Request,
    project_id: str,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Import a dataset from a local folder path."""
    payload = await _parse_request_model(request, DatasetImportRequest)
    dataset = dataset_service.import_dataset(project_id, payload)
    serialized = dataset.model_dump(mode="json")
    if is_hx_request(request):
        return _render_dataset_summary_fragment(request, serialized)
    return ok_response(serialized)


@router.post("/{project_id}/import/upload")
async def import_dataset_from_upload(
    request: Request,
    project_id: str,
    uploaded_file: Annotated[UploadFile, File(...)],
    dataset_service: DatasetServiceDep,
    source_format: Annotated[DatasetSourceFormat | None, Form()] = None,
) -> dict[str, object]:
    """Import a dataset from an uploaded ZIP archive."""
    filename = uploaded_file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise ValidationError("Uploaded dataset archive must be a .zip file.")

    max_upload_size: int = request.app.state.settings.max_upload_size

    try:
        with TemporaryDirectory(prefix="ai-studio-upload-") as temp_dir:
            temp_root = Path(temp_dir)
            archive_path = temp_root / "upload.zip"
            extract_path = temp_root / "extracted"
            extract_path.mkdir(parents=True, exist_ok=True)

            size_bytes = 0
            with archive_path.open("wb") as handle:
                while True:
                    chunk = await uploaded_file.read(_CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > max_upload_size:
                        raise ValidationError(
                            f"Uploaded file exceeds maximum size of {max_upload_size} bytes."
                        )
                    handle.write(chunk)

            try:
                with zipfile.ZipFile(archive_path) as archive:
                    _extract_zip_safely(archive, extract_path)
            except zipfile.BadZipFile as exc:
                raise ValidationError("Uploaded file is not a valid ZIP archive.") from exc

            imported = dataset_service.import_dataset(
                project_id,
                DatasetImportRequest(
                    source_path=str(extract_path),
                    source_format=source_format,
                ),
            )
    finally:
        await uploaded_file.close()

    serialized = imported.model_dump(mode="json")
    if is_hx_request(request):
        return _render_dataset_summary_fragment(request, serialized)
    return ok_response(serialized)


@router.get("/{project_id}/images")
async def list_dataset_images(
    request: Request,
    project_id: str,
    dataset_service: DatasetServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    sort_by: Annotated[DatasetImageSortBy, Query()] = "filename",
    sort_order: Annotated[SortOrder, Query()] = "asc",
    filter_class: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query()] = None,
) -> dict[str, object]:
    """List dataset images with paging, sorting, and filtering."""
    query = DatasetImageListQuery(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        filter_class=filter_class,
        search=search,
    )
    listing = dataset_service.list_images(project_id, query)
    payload = listing.model_dump(mode="json")
    if is_hx_request(request):
        return _render_dataset_images_fragment(
            request,
            project_id=project_id,
            listing=payload,
            query=query.model_dump(mode="json"),
        )
    return ok_response(payload)


@router.get("/{project_id}/images/{filename}/info")
async def get_dataset_image_info(
    request: Request,
    project_id: str,
    filename: str,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Return per-image metadata and available split names."""
    image = dataset_service.get_image_info(project_id, filename)
    dataset = dataset_service.get_dataset(project_id)
    image_payload = image.model_dump(mode="json")
    split_assignments: list[dict[str, str]] = []
    for index, split_name in enumerate(dataset.split_names):
        if index < len(image.split):
            split_assignments.append({"name": split_name, "value": image.split[index]})

    if is_hx_request(request):
        return _render_dataset_image_detail_fragment(
            request,
            project_id=project_id,
            image=image_payload,
            class_name=_extract_primary_class_name(image_payload["annotations"]),
            split_assignments=split_assignments,
        )

    return ok_response({"image": image_payload, "split_names": dataset.split_names})


@router.get("/{project_id}/images/{filename}")
async def get_dataset_image(
    project_id: str,
    filename: str,
    dataset_service: DatasetServiceDep,
) -> FileResponse:
    """Serve a full-resolution dataset image."""
    image_path = dataset_service.get_image_path(project_id, filename)
    return FileResponse(path=image_path)


@router.get("/{project_id}/thumbnails/{filename}")
async def get_dataset_thumbnail(
    project_id: str,
    filename: str,
    dataset_service: DatasetServiceDep,
) -> FileResponse:
    """Serve (or generate) a cached image thumbnail."""
    thumbnail_path = dataset_service.get_thumbnail_path(project_id, filename)
    return FileResponse(path=thumbnail_path)


@router.delete("/{project_id}")
async def clear_dataset(
    request: Request,
    project_id: str,
    dataset_service: DatasetServiceDep,
) -> dict[str, object]:
    """Clear a project's imported dataset."""
    dataset_service.clear_dataset(project_id)
    if is_hx_request(request):
        return _render_dataset_summary_fragment(request, {})
    return ok_response({"project_id": project_id, "cleared": True})
