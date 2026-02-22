from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from app.schemas.export import ExportFormatInfo, ExportValidationResult, OnnxExportOptions
from app.schemas.training import ModelConfig


class ExporterFunction(Protocol):
    """Callable signature for format-specific export implementations."""

    def __call__(
        self,
        *,
        task: str,
        model_config: ModelConfig,
        num_classes: int,
        checkpoint_path: Path,
        output_path: Path,
        options: OnnxExportOptions,
    ) -> ExportValidationResult: ...


_KNOWN_FORMATS: tuple[tuple[str, str], ...] = (
    ("onnx", "ONNX"),
    ("torchscript", "TorchScript"),
    ("tensorrt", "TensorRT"),
    ("openvino", "OpenVINO"),
)
_EXPORT_REGISTRY: dict[str, ExporterFunction] = {}


def register_format(name: str) -> Callable[[ExporterFunction], ExporterFunction]:
    """Register an exporter implementation under a normalized format key."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Export format name cannot be empty.")

    def decorator(exporter: ExporterFunction) -> ExporterFunction:
        _EXPORT_REGISTRY[normalized] = exporter
        return exporter

    return decorator


def get_exporter(name: str) -> ExporterFunction:
    """Resolve a registered exporter by format name."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Export format name cannot be empty.")
    try:
        return _EXPORT_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"Export format '{name}' is not available.") from exc


def list_formats() -> list[ExportFormatInfo]:
    """List known export formats and whether each one is available now."""
    available = set(_EXPORT_REGISTRY)
    return [
        ExportFormatInfo(name=name, display_name=display_name, available=name in available)
        for name, display_name in _KNOWN_FORMATS
    ]
