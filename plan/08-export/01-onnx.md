# Export — ONNX

This document details the ONNX export process.

---

## 1. Overview

ONNX (Open Neural Network Exchange) is the primary export format. It provides broad runtime compatibility (ONNX Runtime, TensorRT, OpenVINO, CoreML, etc.).

---

## 2. Export Implementation

```python
# app/export/onnx_export.py

@register_format("onnx")
def export_onnx(model: nn.Module, config: ExportConfig, output_path: Path):
    model.eval()
    
    input_shape = config.options["input_shape"]  # e.g., [1, 3, 224, 224]
    dummy_input = torch.randn(*input_shape)
    
    dynamic_axes = config.options.get("dynamic_axes", None)
    opset_version = config.options.get("opset_version", 17)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    # Optional: simplify
    if config.options.get("simplify", True):
        simplify_onnx(output_path)
    
    # Validate
    validate_onnx(output_path)
```

---

## 3. Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `opset_version` | int | 17 | ONNX opset version. Higher = more op support. |
| `input_shape` | list[int] | `[1, 3, 224, 224]` | `[batch, channels, height, width]` |
| `dynamic_axes` | dict | batch dim dynamic | Which axes can vary at runtime |
| `simplify` | bool | `true` | Run `onnx-simplifier` to optimize the graph |

### Default Dynamic Axes

```json
{
  "input":  { "0": "batch_size" },
  "output": { "0": "batch_size" }
}
```

This allows running inference with any batch size.

---

## 4. Input Resolution

The input resolution for export should match the training resolution:

- **Classification/Regression/Anomaly**: typically 224×224 (or 518×518 for DINOv3 ViT models).
- **Detection**: typically 640×640 or 800×800.
- **Segmentation**: typically 512×512.

The GUI pre-populates the input resolution from the experiment's augmentation config (the `Resize` transform size).

### DINOv3 Models (HuggingFace Transformers)

DINOv3-based models (ViT and ConvNeXt) can be exported to ONNX. Key considerations:

- **ViT attention**: The self-attention mechanism in DINOv3 ViT models exports correctly to ONNX with opset ≥ 14 (required for `Einsum` and `LayerNormalization` ops).
- **Fixed vs. dynamic resolution**: DINOv3 ViT models work best at fixed input resolutions that are multiples of the patch size (14). Dynamic spatial axes may not be supported.
- **HuggingFace export utilities**: For complex models, `optimum.exporters.onnx` from the HuggingFace `optimum` library can be used as a fallback if `torch.onnx.export` encounters issues.

---

## 5. Output Format by Task

| Task | Output Names | Output Shape | Description |
|------|-------------|-------------|-------------|
| Classification | `output` | `(N, num_classes)` | Class logits |
| Anomaly Detection | `output` | `(N, 1)` | Anomaly logit |
| Regression | `output` | `(N, num_outputs)` | Regression output vector |
| Object Detection | `boxes`, `labels`, `scores` | Variable length | Detected objects |
| Segmentation | `output` | `(N, num_classes, H, W)` | Per-pixel logits |

Detection and instance segmentation models may require special handling for ONNX export due to:
- Non-maximum suppression (NMS) operations.
- Variable-length outputs.
- ROI pooling / align operations.

### Handling Detection Models

torchvision detection models can be exported to ONNX, but may need:

```python
# Wrap the model to export only the inference path
class DetectionExportWrapper(nn.Module):
    def __init__(self, model, score_threshold=0.5, nms_threshold=0.5):
        super().__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
    
    def forward(self, images):
        predictions = self.model(images)
        # Filter and format predictions for clean ONNX export
        return self._post_process(predictions)
```

---

## 6. Validation

After export, two validation steps:

### 6.1 ONNX Model Check

```python
import onnx

def validate_onnx(path: Path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)  # Structural validation
```

### 6.2 Numerical Comparison

```python
import onnxruntime as ort

def numerical_validation(pytorch_model, onnx_path, input_shape):
    sample = torch.randn(*input_shape)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(sample).numpy()
    
    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"input": sample.numpy()})[0]
    
    max_diff = np.max(np.abs(pt_output - ort_output))
    return max_diff < 1e-4, max_diff
```

---

## 7. ONNX Simplification

Use `onnx-simplifier` to optimize the graph:

```python
import onnxsim

def simplify_onnx(path: Path):
    model = onnx.load(str(path))
    simplified, check = onnxsim.simplify(model)
    if check:
        onnx.save(simplified, str(path))
```

Benefits:
- Removes redundant operations.
- Folds constants.
- Reduces file size (slightly).
- Improves runtime compatibility.

---

## 8. Dependencies

```
onnx>=1.15
onnxruntime>=1.17
onnxsim>=0.4          # Optional: for simplification
```

---

## 9. GUI: ONNX Export Options

On the Export page, when ONNX is selected:

| Control | Type | Notes |
|---------|------|-------|
| Opset Version | Dropdown | 13, 14, 15, 16, 17 (default: 17) |
| Input Height | Number | Pre-populated from training config |
| Input Width | Number | Pre-populated from training config |
| Dynamic Batch | Toggle | Default: on |
| Simplify | Toggle | Default: on |
| **Export** | Button | Triggers export |

After export:
- **Validation result**: "Passed ✓" or "Failed ✗" with max numerical difference.
- **File size**: displayed in MB.
- **Download** button.

---

## 10. Related Documents

- Export overview → [00-export-overview.md](00-export-overview.md)
- Export GUI page → [../10-gui/01-pages/06-export-page.md](../10-gui/01-pages/06-export-page.md)
