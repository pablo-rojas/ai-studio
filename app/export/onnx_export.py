from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.export.registry import register_format
from app.export.validation import validate_onnx_export
from app.models.catalog import create_model
from app.schemas.export import ExportValidationResult, OnnxExportOptions
from app.schemas.training import ModelConfig

logger = logging.getLogger(__name__)

_INPUT_NAME = "input"
_OUTPUT_NAME = "output"


@register_format("onnx")
def export_checkpoint_to_onnx(
    *,
    task: str,
    model_config: ModelConfig,
    num_classes: int,
    checkpoint_path: Path,
    output_path: Path,
    options: OnnxExportOptions,
) -> ExportValidationResult:
    """Export one checkpoint to ONNX and run post-export validation."""
    model = _build_model_for_export(
        task=task,
        model_config=model_config,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )
    export_onnx_graph(model=model, output_path=output_path, options=options)

    input_shape = tuple(options.input_shape)
    validation = validate_onnx_export(
        model=model,
        onnx_path=output_path,
        input_shape=(
            int(input_shape[0]),
            int(input_shape[1]),
            int(input_shape[2]),
            int(input_shape[3]),
        ),
    )
    if not validation.passed:
        raise ValueError(
            "ONNX validation failed. "
            f"Max numerical difference {validation.max_diff:.6f} exceeded tolerance."
        )
    return validation


def export_onnx_graph(*, model: nn.Module, output_path: Path, options: OnnxExportOptions) -> None:
    """Export an already-instantiated PyTorch model to ONNX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(torch.device("cpu"))
    model.eval()
    sample_input = torch.randn(*options.input_shape, dtype=torch.float32)

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": options.opset_version,
        "do_constant_folding": True,
        "input_names": [_INPUT_NAME],
        "output_names": [_OUTPUT_NAME],
        "dynamic_axes": options.torch_dynamic_axes(),
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # Prefer the legacy exporter for compatibility when `onnxscript` is unavailable.
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        **export_kwargs,
    )

    if options.simplify:
        _simplify_onnx_graph(output_path)


def _build_model_for_export(
    *,
    task: str,
    model_config: ModelConfig,
    num_classes: int,
    checkpoint_path: Path,
) -> nn.Module:
    # Checkpoint weights define model parameters, so pretrained initialization must be disabled.
    resolved_config = model_config.model_copy(update={"pretrained": False})
    model = create_model(
        task=task,
        architecture=resolved_config.backbone,
        config=resolved_config,
        num_classes=num_classes,
    )

    raw_state_dict = _load_checkpoint_state_dict(checkpoint_path)
    state_dict = _extract_model_state_dict(raw_state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Checkpoint weights do not match the selected model architecture. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    model.eval()
    return model


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"Checkpoint '{checkpoint_path.name}' is invalid.")

    raw_state_dict = checkpoint_payload.get("state_dict")
    if raw_state_dict is None:
        raw_state_dict = checkpoint_payload
    if not isinstance(raw_state_dict, dict):
        raise ValueError(f"Checkpoint '{checkpoint_path.name}' does not contain a state_dict.")
    return raw_state_dict


def _extract_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    model_entries = {
        key[6:]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    if model_entries:
        return model_entries
    return state_dict


def _simplify_onnx_graph(onnx_path: Path) -> None:
    try:
        import onnx
        import onnxsim
    except ImportError:
        logger.info("Skipping ONNX simplification because onnxsim is not installed.")
        return

    model = onnx.load(str(onnx_path))
    simplified_model, check = onnxsim.simplify(model)
    if not check:
        raise ValueError("ONNX simplification failed integrity checks.")
    onnx.save(simplified_model, str(onnx_path))
