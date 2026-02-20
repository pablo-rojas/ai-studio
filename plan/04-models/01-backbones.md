# Models — Backbones

This document describes the feature-extractor backbones available in AI Studio.

---

## 1. Overview

A backbone is a neural network used as a feature extractor. Backbones come from two sources: **torchvision** (CNNs with ImageNet pretraining) and **HuggingFace Transformers** (DINOv3 models with self-supervised pretraining). The final classification/regression layers are removed, and the backbone outputs feature maps that task-specific heads consume.

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

### 2.4 DINOv3 ViT (HuggingFace Transformers)

| Variant | Parameters | Output Channels | Patch Size | Notes |
|---------|-----------|----------------|------------|-------|
| DINOv3 ViT-S/14 | 22M | 384 | 14×14 | Small, fast |
| DINOv3 ViT-B/14 | 86M | 768 | 14×14 | Standard baseline |
| DINOv3 ViT-L/14 | 304M | 1024 | 14×14 | High capacity |

- **Self-supervised pretraining**: DINOv3 models are pretrained with a self-supervised distillation objective — no ImageNet labels required during pretraining. This produces highly transferable features.
- **Feature map access**: ViT outputs a sequence of patch tokens `(N, num_patches, D)`. For classification, the `[CLS]` token `(N, D)` is used. For dense prediction tasks (detection, segmentation), patch tokens are reshaped into a spatial feature map `(N, D, H', W')` where `H' = H/patch_size`, `W' = W/patch_size`.
- **Multi-scale features**: ViT is not natively hierarchical. For FPN-based tasks, intermediate transformer block outputs at layers 1/4, 1/2, 3/4, and 1/1 of depth are used to simulate multi-scale features.
- **Source**: `transformers.Dinov2Model` (or equivalent DINOv3 class from HuggingFace).

### 2.5 DINOv3 ConvNeXt (HuggingFace Transformers)

| Variant | Parameters | Output Channels | Notes |
|---------|-----------|----------------|-------|
| DINOv3 ConvNeXt-S | 50M | 768 | Efficient, hierarchical |
| DINOv3 ConvNeXt-B | 89M | 1024 | Standard baseline |
| DINOv3 ConvNeXt-L | 198M | 1536 | High capacity |

- **Self-supervised pretraining**: Same DINOv3 self-supervised objective applied to ConvNeXt architecture.
- **Feature map access**: ConvNeXt is a hierarchical CNN producing feature maps at 4 stages with strides 4, 8, 16, 32 — natively compatible with FPN.
- **Source**: `transformers.Dinov2Model` with ConvNeXt backbone (or equivalent DINOv3 ConvNeXt class from HuggingFace).
- **Multi-scale features**: `stage1` through `stage4`, directly usable by FPN for detection and segmentation.

---

## 3. Feature Extractor Interface

All backbones are wrapped to expose a consistent interface:

```python
class BackboneWrapper:
    """Unified interface for accessing backbone features.
    Supports both torchvision and HuggingFace Transformers backbones."""
    
    def __init__(self, model: nn.Module, backbone_name: str, source: str = "torchvision"):
        self.model = model
        self.source = source  # "torchvision" or "huggingface"
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
| Classification | Any backbone → GAP → FC head (torchvision); [CLS] token → FC head (DINOv3 ViT); GAP → FC head (DINOv3 ConvNeXt) |
| Anomaly Detection | ResNet-18 (frozen) as feature extractor for teacher distillation — no FC/binary head (see note below) |
| Regression | Any backbone → GAP → FC head (`num_outputs`) |
| Object Detection | ResNet-50 + FPN (torchvision); DINOv3 ConvNeXt + FPN or DINOv3 ViT + FPN (HuggingFace) |
| Oriented OD | ResNet-50 + FPN (custom head) |
| Segmentation | ResNet-50/101 or MobileNetV3 (torchvision); DINOv3 ViT/ConvNeXt + decoder (HuggingFace) |
| Instance Segmentation | ResNet-50 + FPN (built into Mask R-CNN) |

For detection and segmentation, the backbone is part of the full architecture (e.g., `fasterrcnn_resnet50_fpn` includes both backbone and head). The user selects the full architecture rather than backbone + head separately.

For classification and regression, the user selects a backbone and the head is auto-determined by the task type.

**DINOv3 note**: DINOv3 backbones (both ViT and ConvNeXt) are loaded via HuggingFace Transformers. They use different preprocessing (DINOv3-specific normalization stats) and produce outputs in a different format. The `BackboneWrapper` normalizes these differences so that downstream heads receive features in the standard `(N, C, H, W)` format.

**Anomaly Detection note**: The backbone (ResNet-18, pretrained) is used purely as a **frozen feature extractor** to train the teacher network via knowledge distillation. There is no classification head — anomaly scores are derived from the discrepancy between teacher and student descriptors. This is a custom pipeline (not torchvision-based); see [02-heads.md §2.2](02-heads.md) and [02-anomaly-detection.md](../03-tasks/02-anomaly-detection.md) for details.

---

## 5. Input Specifications

All backbones expect:
- **Input shape**: `(N, 3, H, W)` — batch of 3-channel RGB images.
- **Value range**: `[0, 1]` (after `ToImage` + `ToDtype`), then normalized.
- **Normalization**: ImageNet stats for torchvision backbones (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`); DINOv3-specific stats for HuggingFace backbones (determined by the HuggingFace preprocessor config).
- **Minimum input size**: typically 32×32 (but recommended ≥ 224×224 for pretrained weights). DINOv3 ViT models work best at multiples of the patch size (e.g., 224, 518).

---

## 6. Related Documents

- Architecture catalog → [00-architecture-catalog.md](00-architecture-catalog.md)
- Task-specific heads → [02-heads.md](02-heads.md)
- Pretrained weights → [03-pretrained-weights.md](03-pretrained-weights.md)
