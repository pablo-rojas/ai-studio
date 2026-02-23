from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.exceptions import NotFoundError
from app.core.export_service import ExportService
from app.core.training_service import TrainingService

_DEFAULT_INPUT_SHAPE = [1, 3, 224, 224]


def _extract_hw(raw_size: Any) -> tuple[int, int] | None:
    if isinstance(raw_size, (list, tuple)):
        if len(raw_size) >= 2:
            height = int(raw_size[0])
            width = int(raw_size[1])
        elif len(raw_size) == 1:
            height = int(raw_size[0])
            width = int(raw_size[0])
        else:
            return None
    else:
        try:
            side = int(raw_size)
        except (TypeError, ValueError):
            return None
        height = side
        width = side

    if height <= 0 or width <= 0:
        return None
    return height, width


def _resolve_input_shape(experiment_payload: dict[str, Any]) -> list[int]:
    augmentations = experiment_payload.get("augmentations", {})
    val_steps = augmentations.get("val", []) if isinstance(augmentations, dict) else []
    train_steps = augmentations.get("train", []) if isinstance(augmentations, dict) else []

    prioritized = (
        (val_steps, ("Resize",)),
        (val_steps, ("CenterCrop", "RandomResizedCrop", "RandomCrop")),
        (train_steps, ("RandomResizedCrop", "Resize")),
    )

    for steps, target_names in prioritized:
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("name") not in target_names:
                continue
            params = step.get("params", {})
            if not isinstance(params, dict):
                continue
            size = params.get("size")
            hw = _extract_hw(size)
            if hw is None:
                continue
            return [1, 3, hw[0], hw[1]]

    return list(_DEFAULT_INPUT_SHAPE)


def _resolve_default_checkpoint(checkpoints: list[str]) -> str:
    if "best" in checkpoints:
        return "best"
    if checkpoints:
        return checkpoints[0]
    return "best"


def _resolve_selected_export_record(
    *,
    project_id: str,
    export_service: ExportService,
    exports: list[dict[str, Any]],
    selected_export_id: str | None,
    selected_experiment_id: str | None,
) -> dict[str, Any] | None:
    candidate_export_id = selected_export_id

    if candidate_export_id is None and selected_experiment_id is not None:
        for item in exports:
            if item.get("experiment_id") == selected_experiment_id:
                candidate_export_id = str(item.get("id"))
                break

    if candidate_export_id is None and exports:
        candidate_export_id = str(exports[0].get("id"))

    if candidate_export_id is None:
        return None

    try:
        return export_service.get_export(project_id, candidate_export_id).model_dump(mode="json")
    except NotFoundError:
        return None


def _build_export_url(
    base_path: str,
    *,
    selected_experiment_id: str | None,
    selected_export_id: str | None,
    open_new_export_modal: bool = False,
) -> str:
    query: dict[str, str] = {}
    if selected_experiment_id:
        query["experiment_id"] = selected_experiment_id
    if selected_export_id:
        query["export_id"] = selected_export_id
    if open_new_export_modal:
        query["new_export"] = "1"
    encoded = urlencode(query)
    if not encoded:
        return base_path
    return f"{base_path}?{encoded}"


def build_export_page_context(
    *,
    project_id: str,
    dataset_service: DatasetService,
    evaluation_service: EvaluationService,
    export_service: ExportService,
    training_service: TrainingService,
    selected_experiment_id: str | None = None,
    selected_export_id: str | None = None,
    open_new_export_modal: bool = False,
) -> dict[str, Any]:
    """Build the template context used by the Export page and HTMX fragments."""
    try:
        dataset_model = dataset_service.get_dataset(project_id)
    except NotFoundError:
        dataset_model = None
    dataset = dataset_model.model_dump(mode="json") if dataset_model is not None else None

    experiments = [
        experiment.model_dump(mode="json")
        for experiment in training_service.list_experiments(project_id)
    ]
    completed_experiments = [
        experiment for experiment in experiments if experiment.get("status") == "completed"
    ]
    completed_ids = {str(experiment.get("id")) for experiment in completed_experiments}
    experiment_name_by_id = {
        str(experiment.get("id")): str(experiment.get("name"))
        for experiment in experiments
        if experiment.get("id") is not None
    }

    exports = [item.model_dump(mode="json") for item in export_service.list_exports(project_id)]
    selected_export = _resolve_selected_export_record(
        project_id=project_id,
        export_service=export_service,
        exports=exports,
        selected_export_id=selected_export_id,
        selected_experiment_id=selected_experiment_id,
    )

    resolved_experiment_id = selected_experiment_id
    if (
        selected_export is not None
        and str(selected_export.get("experiment_id")) in completed_ids
        and resolved_experiment_id not in completed_ids
    ):
        resolved_experiment_id = str(selected_export.get("experiment_id"))
    if resolved_experiment_id not in completed_ids:
        resolved_experiment_id = (
            str(completed_experiments[0].get("id")) if completed_experiments else None
        )

    checkpoints: list[str] = []
    if resolved_experiment_id is not None:
        checkpoints = evaluation_service.list_checkpoints(project_id, resolved_experiment_id)

    default_input_shape = list(_DEFAULT_INPUT_SHAPE)
    if resolved_experiment_id is not None:
        try:
            experiment_payload = training_service.get_experiment(
                project_id,
                resolved_experiment_id,
            ).model_dump(mode="json")
            default_input_shape = _resolve_input_shape(experiment_payload)
        except NotFoundError:
            default_input_shape = list(_DEFAULT_INPUT_SHAPE)

    form_defaults = {
        "experiment_id": resolved_experiment_id or "",
        "checkpoint": _resolve_default_checkpoint(checkpoints),
        "opset_version": 17,
        "input_height": default_input_shape[2],
        "input_width": default_input_shape[3],
        "dynamic_batch": True,
        "simplify": True,
    }

    if selected_export is not None and selected_export.get("format") == "onnx":
        options = selected_export.get("options", {})
        if isinstance(options, dict):
            input_shape = options.get("input_shape")
            if isinstance(input_shape, list) and len(input_shape) >= 4:
                try:
                    form_defaults["input_height"] = int(input_shape[2])
                    form_defaults["input_width"] = int(input_shape[3])
                except (TypeError, ValueError):
                    pass
            opset_version = options.get("opset_version")
            if opset_version is not None:
                try:
                    form_defaults["opset_version"] = int(opset_version)
                except (TypeError, ValueError):
                    pass
            form_defaults["dynamic_batch"] = options.get("dynamic_axes") is not None
            form_defaults["simplify"] = bool(options.get("simplify", True))

        if selected_export.get("experiment_id") == resolved_experiment_id:
            form_defaults["checkpoint"] = str(
                selected_export.get("checkpoint", form_defaults["checkpoint"])
            )

    if checkpoints and form_defaults["checkpoint"] not in checkpoints:
        form_defaults["checkpoint"] = _resolve_default_checkpoint(checkpoints)

    selected_experiment: dict[str, Any] | None = None
    if resolved_experiment_id is not None:
        for experiment in completed_experiments:
            if str(experiment.get("id")) == resolved_experiment_id:
                selected_experiment = experiment
                break

    selected_export_id_value = str(selected_export.get("id")) if selected_export else None

    page_base_path = f"/projects/{project_id}/export"
    fragment_base_path = f"/api/export/{project_id}"

    return {
        "project_id": project_id,
        "dataset": dataset,
        "experiments": experiments,
        "completed_experiments": completed_experiments,
        "selected_experiment": selected_experiment,
        "selected_experiment_id": resolved_experiment_id,
        "checkpoints": checkpoints,
        "exports": exports,
        "selected_export": selected_export,
        "selected_export_id": selected_export_id_value,
        "experiment_name_by_id": experiment_name_by_id,
        "form_defaults": form_defaults,
        "open_new_export_modal": open_new_export_modal,
        "page_url": _build_export_url(
            page_base_path,
            selected_experiment_id=resolved_experiment_id,
            selected_export_id=selected_export_id_value,
        ),
        "page_url_with_modal": _build_export_url(
            page_base_path,
            selected_experiment_id=resolved_experiment_id,
            selected_export_id=selected_export_id_value,
            open_new_export_modal=True,
        ),
        "fragment_url": _build_export_url(
            fragment_base_path,
            selected_experiment_id=resolved_experiment_id,
            selected_export_id=selected_export_id_value,
        ),
        "fragment_url_with_modal": _build_export_url(
            fragment_base_path,
            selected_experiment_id=resolved_experiment_id,
            selected_export_id=selected_export_id_value,
            open_new_export_modal=True,
        ),
    }
