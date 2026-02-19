# Models — Backbones

This document describes the feature-extractor backbones available in AI Studio.

---

## 1. Overview

A backbone is a convolutional neural network (from torchvision) used as a feature extractor. The final classification/regression layers are removed, and the backbone outputs feature maps that task-specific heads consume.

---

## 2. Backbone Families

### 2.1 ResNet

| Variant | Layers | Parameters | Output Channels | Notes |
|---------|--------|-----------|----------------|-------|
| ResNet-18 | 18 | 11.7M | 512 | Lightweight, fast |
| ResNet-34 | 34 | 21.8M | 512 | Mid-range |
| ResNet-50 | 50 | 25.6M | 2048 | Standard baseline, bottleneck blocks |
| ResNet-101 | 101 | 44.5M | 2048 | Higher capacity |

- **Feature map access**: `layer1` (stride 4), `layer2` (stride 8), `layer3` (stride 16), `layer4` (stride 32).
- For FPN-based models (detection, segmentation): tap `layer1` through `layer4`.
- For classification/regression: use `layer4` output + global average pooling.

### 2.2 EfficientNet

| Variant | Parameters | Output Channels | Notes |
|---------|-----------|----------------|-------|
| EfficientNet-B0 | 5.3M | 1280 | Compound scaling baseline |
| EfficientNet-B1 | 7.8M | 1280 | Slightly larger |
| EfficientNet-B2 | 9.1M | 1408 | Mid-range |
| EfficientNet-B3 | 12.2M | 1536 | Good accuracy/size trade-off |
| EfficientNet-B4 | 19.3M | 1792 | Higher accuracy |

- **Feature map access**: `features` sequential returns feature maps at multiple scales.
- Global average pooling after `features` → `classifier`.

### 2.3 MobileNetV3

| Variant | Parameters | Output Channels | Notes |
|---------|-----------|----------------|-------|
| MobileNetV3-Small | 2.5M | 576 | Minimal footprint |
| MobileNetV3-Large | 5.5M | 960 | Better accuracy, still lightweight |

- Designed for mobile/edge deployment.
- Feature extraction via `features` sequential.

---

## 3. Feature Extractor Interface

All backbones are wrapped to expose a consistent interface:

```python
class BackboneWrapper:
    """Unified interface for accessing backbone features."""
    
    def __init__(self, model: nn.Module, backbone_name: str):
        self.model = model
        self.out_channels = self._get_out_channels(backbone_name)
    
    def forward(self, x: Tensor) -> Tensor:
        """Returns the final feature map (before classification head)."""
        return self.model(x)
    
    def forward_multiscale(self, x: Tensor) -> Dict[str, Tensor]:
        """Returns feature maps at multiple scales (for FPN)."""
        # Returns {"feat1": ..., "feat2": ..., "feat3": ..., "feat4": ...}
        ...
```

For detection and segmentation tasks, torchvision models already include FPN and backbone integration. The wrapper is mainly useful for classification/regression/anomaly tasks where we compose backbone + head manually.

---

## 4. Backbone Selection Rules

| Task | Backbone Usage |
|------|---------------|
| Classification | Any backbone → GAP → FC head |
| Anomaly Detection | Any backbone → GAP → FC head (binary) |
| Regression | Any backbone → GAP → FC head (scalar) |
| Object Detection | ResNet-50 + FPN (built into torchvision detection models) |
| Oriented OD | ResNet-50 + FPN (custom head) |
| Segmentation | ResNet-50/101 or MobileNetV3 (built into torchvision segmentation models) |
| Instance Segmentation | ResNet-50 + FPN (built into Mask R-CNN) |

For detection and segmentation, the backbone is part of the full architecture (e.g., `fasterrcnn_resnet50_fpn` includes both backbone and head). The user selects the full architecture rather than backbone + head separately.

For classification, anomaly detection, and regression, the user selects a backbone and the head is auto-determined by the task type.

---

## 5. Input Specifications

All backbones expect:
- **Input shape**: `(N, 3, H, W)` — batch of 3-channel RGB images.
- **Value range**: `[0, 1]` (after `ToTensor`), then normalized with ImageNet stats.
- **Minimum input size**: typically 32×32 (but recommended ≥ 224×224 for pretrained weights).

---

## 6. Related Documents

- Architecture catalog → [00-architecture-catalog.md](00-architecture-catalog.md)
- Task-specific heads → [02-heads.md](02-heads.md)
- Pretrained weights → [03-pretrained-weights.md](03-pretrained-weights.md)
