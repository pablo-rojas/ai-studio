from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError as PydanticValidationError

from app.api.dependencies import get_split_service
from app.api.responses import is_hx_request, ok_response
from app.core.exceptions import ValidationError
from app.core.split_service import SplitService
from app.schemas.split import SplitCreateRequest, SplitPreviewRequest, SplitRatios

router = APIRouter()
SplitServiceDep = Annotated[SplitService, Depends(get_split_service)]


def _render_split_list_fragment(
    request: Request,
    *,
    project_id: str,
    splits: list[dict[str, Any]],
):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/split_list.html",
        {"project_id": project_id, "splits": splits},
    )


def _render_split_preview_fragment(request: Request, preview: dict[str, Any]):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/split_preview.html",
        {"preview": preview},
    )


def _parse_ratio_query(ratios: str) -> SplitRatios:
    text = ratios.strip()
    if not text:
        raise ValidationError("Query parameter 'ratios' cannot be empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3:
            raise ValidationError(
                "Ratios must be provided as JSON or comma-separated values in train,val,test order."
            ) from None
        try:
            return SplitRatios(
                train=float(parts[0]),
                val=float(parts[1]),
                test=float(parts[2]),
            )
        except ValueError as exc:
            raise ValidationError("Ratios must be numeric values.") from exc

    if isinstance(parsed, dict):
        return SplitRatios.model_validate(parsed)
    if isinstance(parsed, list) and len(parsed) == 3:
        return SplitRatios(
            train=float(parsed[0]),
            val=float(parsed[1]),
            test=float(parsed[2]),
        )
    raise ValidationError("Ratios JSON must be an object or a three-item list.")


def _build_preview_request(
    *,
    ratios: str | None,
    train: float | None,
    val: float | None,
    test: float | None,
    seed: int | None,
) -> SplitPreviewRequest:
    if ratios is not None:
        split_ratios = _parse_ratio_query(ratios)
    else:
        if train is None or val is None or test is None:
            raise ValidationError("Provide either 'ratios' or all of 'train', 'val', and 'test'.")
        split_ratios = SplitRatios(train=train, val=val, test=test)

    return SplitPreviewRequest(ratios=split_ratios, seed=seed)


def _normalize_split_create_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_payload)
    if "ratios" not in payload:
        train = payload.pop("train", None)
        val = payload.pop("val", None)
        test = payload.pop("test", None)
        if train is not None and val is not None and test is not None:
            payload["ratios"] = {
                "train": train,
                "val": val,
                "test": test,
            }

    seed = payload.get("seed")
    if isinstance(seed, str) and not seed.strip():
        payload["seed"] = None

    return payload


async def _parse_split_create_request(request: Request) -> SplitCreateRequest:
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
        if isinstance(raw_payload, dict):
            normalized: Any = _normalize_split_create_payload(raw_payload)
        else:
            normalized = raw_payload
        return SplitCreateRequest.model_validate(normalized)
    except PydanticValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body",),
                    "msg": "Invalid split payload.",
                    "input": None,
                }
            ]
        ) from exc


@router.get("/{project_id}")
async def list_splits(
    request: Request,
    project_id: str,
    split_service: SplitServiceDep,
) -> dict[str, object]:
    """List all saved splits for a project."""
    splits = split_service.list_splits(project_id)
    payload = [item.model_dump(mode="json") for item in splits]
    if is_hx_request(request):
        return _render_split_list_fragment(request, project_id=project_id, splits=payload)
    return ok_response({"splits": payload})


@router.post("/{project_id}")
async def create_split(
    request: Request,
    project_id: str,
    split_service: SplitServiceDep,
) -> dict[str, object]:
    """Create and persist a new split."""
    payload = await _parse_split_create_request(request)
    split = split_service.create_split(project_id, payload)
    if is_hx_request(request):
        refreshed = split_service.list_splits(project_id)
        serialized = [item.model_dump(mode="json") for item in refreshed]
        return _render_split_list_fragment(request, project_id=project_id, splits=serialized)
    return ok_response(split.model_dump(mode="json"))


@router.get("/{project_id}/preview")
async def preview_split_via_query(
    request: Request,
    project_id: str,
    split_service: SplitServiceDep,
    ratios: Annotated[str | None, Query()] = None,
    train: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    val: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    test: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    seed: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, object]:
    """Preview split statistics without persisting them."""
    payload = _build_preview_request(
        ratios=ratios,
        train=train,
        val=val,
        test=test,
        seed=seed,
    )
    preview = split_service.preview_split(project_id, payload)
    serialized = preview.model_dump(mode="json")
    if is_hx_request(request):
        return _render_split_preview_fragment(request, serialized)
    return ok_response(serialized)


@router.post("/{project_id}/preview")
async def preview_split_via_body(
    request: Request,
    project_id: str,
    payload: SplitPreviewRequest,
    split_service: SplitServiceDep,
) -> dict[str, object]:
    """Preview split statistics from a JSON body payload."""
    preview = split_service.preview_split(project_id, payload)
    serialized = preview.model_dump(mode="json")
    if is_hx_request(request):
        return _render_split_preview_fragment(request, serialized)
    return ok_response(serialized)


@router.get("/{project_id}/{split_name}")
async def get_split(
    project_id: str,
    split_name: str,
    split_service: SplitServiceDep,
) -> dict[str, object]:
    """Return one split by name."""
    split = split_service.get_split(project_id, split_name)
    return ok_response(split.model_dump(mode="json"))


@router.delete("/{project_id}/{split_name}")
async def delete_split(
    request: Request,
    project_id: str,
    split_name: str,
    split_service: SplitServiceDep,
) -> dict[str, object]:
    """Delete an existing split."""
    split_service.delete_split(project_id, split_name)
    if is_hx_request(request):
        refreshed = split_service.list_splits(project_id)
        serialized = [item.model_dump(mode="json") for item in refreshed]
        return _render_split_list_fragment(request, project_id=project_id, splits=serialized)
    return ok_response({"project_id": project_id, "split_name": split_name, "deleted": True})
