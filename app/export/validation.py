from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from app.schemas.export import ExportValidationResult

_DEFAULT_MAX_DIFF_TOLERANCE = 1e-4


def validate_onnx_model_file(onnx_path: Path) -> None:
    """Run structural ONNX validation for one exported model file."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)


def validate_onnx_export(
    *,
    model: nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    max_diff_tolerance: float = _DEFAULT_MAX_DIFF_TOLERANCE,
) -> ExportValidationResult:
    """Compare PyTorch and ONNX Runtime predictions for a random sample input."""
    if max_diff_tolerance <= 0:
        raise ValueError("max_diff_tolerance must be greater than 0.")

    validate_onnx_model_file(onnx_path)

    sample_input = torch.randn(*input_shape, dtype=torch.float32)
    pytorch_output = _run_pytorch(model=model, sample_input=sample_input)
    onnx_output = _run_onnx_runtime(onnx_path=onnx_path, sample_input=sample_input)

    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            "Output shape mismatch between PyTorch and ONNX Runtime. "
            f"pytorch={pytorch_output.shape}, onnx={onnx_output.shape}"
        )

    difference = np.abs(pytorch_output - onnx_output)
    max_diff = float(np.max(difference)) if difference.size else 0.0
    mean_diff = float(np.mean(difference)) if difference.size else 0.0
    return ExportValidationResult(
        passed=max_diff <= max_diff_tolerance,
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def _run_pytorch(*, model: nn.Module, sample_input: Tensor) -> np.ndarray:
    model = model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    return _coerce_output_to_numpy(output)


def _run_onnx_runtime(*, onnx_path: Path, sample_input: Tensor) -> np.ndarray:
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception:
        session = ort.InferenceSession(str(onnx_path))

    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("The ONNX model does not define any inputs.")

    output_values = session.run(None, {inputs[0].name: sample_input.numpy()})
    if not output_values:
        raise ValueError("The ONNX model did not produce any outputs.")
    return np.asarray(output_values[0])


def _coerce_output_to_numpy(output: Any) -> np.ndarray:
    if isinstance(output, Tensor):
        return output.detach().cpu().numpy()

    if isinstance(output, (list, tuple)) and output:
        first_item = output[0]
        if isinstance(first_item, Tensor):
            return first_item.detach().cpu().numpy()

    raise ValueError(
        "Export validation currently supports tensor outputs only. "
        f"Received type: {type(output).__name__}."
    )
