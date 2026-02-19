# Export — Overview

This document describes the model export system: converting trained PyTorch models to deployable formats.

---

## 1. Overview

Export converts a trained model checkpoint (`.ckpt`) into a portable format that can be deployed outside of AI Studio. Initially only **ONNX** is supported, with a format registry design that allows adding TensorRT, OpenVINO, and PyTorch (TorchScript) later.

---

## 2. Export Flow

```
User navigates to Export page
         │
         ▼
   Select experiment + checkpoint
   (best or last)
         │
         ▼
   Select export format (ONNX)
         │
         ▼
   Configure export options
   (input resolution, dynamic batch, opset)
         │
         ▼
   Click "Export"
         │
         ▼
   Backend: load checkpoint → build model → export
         │
         ▼
   Validate exported model (sample inference)
         │
         ▼
   Store exported file in exports/ folder
         │
         ▼
   User can download the exported file
```

---

## 3. Export Configuration

```json
{
  "id": "export-xyz789",
  "experiment_id": "exp-a1b2c3d4",
  "checkpoint": "best",
  "format": "onnx",
  "options": {
    "opset_version": 17,
    "input_shape": [1, 3, 224, 224],
    "dynamic_axes": { "input": { 0: "batch_size" }, "output": { 0: "batch_size" } },
    "simplify": true
  },
  "created_at": "2026-02-19T15:00:00Z",
  "status": "completed",
  "output_file": "model.onnx",
  "output_size_mb": 98.5,
  "validation": {
    "passed": true,
    "max_diff": 1.2e-6
  }
}
```

---

## 4. Format Registry

An extensible registry for export formats:

```python
# app/export/registry.py

EXPORT_REGISTRY: Dict[str, ExportFunction] = {}

def register_format(name: str):
    def decorator(fn):
        EXPORT_REGISTRY[name] = fn
        return fn
    return decorator

def get_available_formats() -> list[dict]:
    return [
        {"name": "onnx", "display_name": "ONNX", "available": True},
        {"name": "torchscript", "display_name": "TorchScript", "available": False},
        {"name": "tensorrt", "display_name": "TensorRT", "available": False},
        {"name": "openvino", "display_name": "OpenVINO", "available": False},
    ]
```

Formats marked `available: False` are shown as "Coming Soon" in the GUI.

---

## 5. Export Considerations by Task

| Task | Export Notes |
|------|-------------|
| Classification | Simple: single input → logits output |
| Anomaly Detection | Single input → anomaly score |
| Object Detection | May need NMS post-processing. torchvision models include NMS by default. |
| Oriented OD | Custom NMS for rotated boxes may not export cleanly to ONNX |
| Segmentation | Large output tensor (H×W×C). Consider output resolution options. |
| Instance Segmentation | Complex multi-output (boxes + masks + labels). ONNX export may need custom handling. |
| Regression | Simple: single input → output vector `(N, num_outputs)` |

### Pre-Export Model Preparation

Before export, the model is prepared:
1. Set to eval mode.
2. Remove training-only components (dropout in eval already disabled, but verify).
3. For detection models: ensure NMS is included in the forward pass.
4. Wrap if needed to produce a clean input/output interface.

---

## 6. Post-Export Validation

After exporting, validate the exported model:

```python
# app/export/validation.py

def validate_export(pytorch_model, exported_path, format, input_shape):
    # Generate random input
    sample_input = torch.randn(input_shape)
    
    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input)
    
    # Exported model output
    exported_output = run_exported_model(exported_path, format, sample_input.numpy())
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output.numpy() - exported_output))
    passed = max_diff < 1e-4
    
    return {"passed": passed, "max_diff": float(max_diff)}
```

---

## 7. Storage Layout

```
projects/<project-id>/exports/
├── exports_index.json
└── export-xyz789/
    ├── export.json          # Config + status + validation results
    └── model.onnx           # The exported model file
```

---

## 8. Download

The exported file can be downloaded via:
- **GUI**: "Download" button on the Export page.
- **API**: `GET /api/export/{export_id}/download` → file stream.

---

## 9. Future Formats

### TorchScript
- `torch.jit.script()` or `torch.jit.trace()`.
- Produces `.pt` file.
- Runs in any PyTorch-compatible environment without Python source.

### TensorRT
- Requires NVIDIA TensorRT SDK.
- Converts ONNX → optimized TensorRT engine (`.engine`).
- Fastest inference on NVIDIA GPUs.
- Platform-specific (must match GPU architecture).

### OpenVINO
- Intel's inference toolkit.
- Converts ONNX → OpenVINO IR (`.xml` + `.bin`).
- Optimized for Intel CPUs and integrated GPUs.

---

## 10. Related Documents

- ONNX export detail → [01-onnx.md](01-onnx.md)
- Checkpoints (input) → [../06-experiment-tracking/02-checkpoints.md](../06-experiment-tracking/02-checkpoints.md)
- Export GUI page → [../10-gui/01-pages/06-export-page.md](../10-gui/01-pages/06-export-page.md)
