from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from app.export import export_onnx_graph, list_formats, validate_onnx_export
from app.schemas.export import OnnxExportOptions


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def test_export_onnx_graph_writes_file_and_validates_numerically(tmp_path: Path) -> None:
    model = _TinyClassifier()
    output_path = tmp_path / "model.onnx"

    export_onnx_graph(
        model=model,
        output_path=output_path,
        options=OnnxExportOptions(input_shape=[1, 3, 32, 32], simplify=False),
    )

    assert output_path.exists()

    validation = validate_onnx_export(
        model=model,
        onnx_path=output_path,
        input_shape=(1, 3, 32, 32),
    )
    assert validation.passed
    assert validation.max_diff < 1e-4


def test_list_formats_marks_onnx_available_and_future_formats_unavailable() -> None:
    formats = {item.name: item for item in list_formats()}
    assert formats["onnx"].available is True
    assert formats["torchscript"].available is False
    assert formats["tensorrt"].available is False
    assert formats["openvino"].available is False
