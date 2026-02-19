# Models — Task-Specific Heads

This document describes the head modules that sit on top of backbones to produce task-specific outputs.

---

## 1. Overview

A **head** transforms backbone feature maps into task-specific predictions. For classification/regression/anomaly tasks, the head is a simple module we compose with the backbone. For detection and segmentation, heads are integrated into the torchvision model architectures.

---

## 2. Head Definitions

### 2.1 Classification Head

**File**: `app/models/heads/classification.py`

```python
class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W) from backbone
        x = self.pool(x)        # (N, C, 1, 1)
        x = self.flatten(x)     # (N, C)
        x = self.dropout(x)     # (N, C)
        x = self.fc(x)          # (N, num_classes)
        return x  # logits
```

**Output**: `(N, num_classes)` logits. Apply `softmax` for probabilities at inference.

### 2.2 Anomaly Detection Head

**File**: `app/models/heads/anomaly.py`

```python
class AnomalyHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # logit (scalar per image)
```

**Output**: `(N, 1)` logit. Apply `sigmoid` for anomaly probability.

### 2.3 Regression Head

**File**: `app/models/heads/regression.py`

```python
class RegressionHead(nn.Module):
    def __init__(self, in_features: int, num_outputs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_outputs)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw scalar prediction(s)
```

**Output**: `(N, num_outputs)` scalar values. No activation — raw predictions. `num_outputs` is determined by the dataset (`len(values)` per image).

### 2.4 Detection Heads

**File**: `app/models/heads/detection.py`

Detection heads are not standalone modules — they are part of the torchvision detection models:

| Model | Head Components |
|-------|----------------|
| **Faster R-CNN** | RPN (Region Proposal Network) + RoI Head (classification + bbox regression) |
| **RetinaNet** | Classification subnet + Regression subnet (on each FPN level) |
| **FCOS** | Classification + Centerness + Bbox regression (per-pixel on FPN levels) |
| **SSD/SSDLite** | Multi-scale classification + regression heads |

These are instantiated via `torchvision.models.detection.*` factories with `num_classes` parameter.

**Customization**: To change `num_classes` on a pretrained model:
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

### 2.5 Oriented Detection Head

**File**: `app/models/heads/oriented_detection.py`

Extends a standard detection head with an angle regression branch:

```python
class OrientedDetectionHead(nn.Module):
    """Additional angle prediction head on top of FPN features."""
    
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        # Per-anchor angle regression
        self.angle_pred = nn.Conv2d(in_channels, num_anchors, kernel_size=3, padding=1)
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        angles = [self.angle_pred(f) for f in features]
        return angles  # One angle prediction per anchor per FPN level
```

Integrated with a modified FCOS or RetinaNet that includes the angle output in its loss.

### 2.6 Segmentation Heads

**File**: `app/models/heads/segmentation.py`

Like detection, segmentation heads are integrated in torchvision models:

| Model | Decoder |
|-------|---------|
| **FCN** | 1×1 convolutions on backbone features, bilinear upsampling |
| **DeepLabV3** | ASPP (Atrous Spatial Pyramid Pooling) module |
| **DeepLabV3+** | ASPP + low-level feature fusion + decoder |
| **LRASPP** | Lite R-ASPP for mobile |

**Customization**: Change `num_classes` via the `classifier` submodule:
```python
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
```

### 2.7 Instance Segmentation Head

**File**: `app/models/heads/instance_segmentation.py`

Part of Mask R-CNN:
- **Box predictor**: same as Faster R-CNN.
- **Mask predictor**: per-RoI FCN that predicts a binary mask for each class.

**Customization**:
```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
```

---

## 3. Head Selection Logic

The head is automatically determined by the task — the user doesn't choose it directly:

```python
def get_head_for_task(task: TaskType) -> str:
    return {
        TaskType.CLASSIFICATION: "classification",
        TaskType.ANOMALY_DETECTION: "anomaly",
        TaskType.REGRESSION: "regression",
        # Detection and segmentation heads are embedded in full architectures
    }[task]
```

For detection/segmentation tasks, the "head" is part of the full model — there's no separate head selection.

---

## 4. Related Documents

- Architecture catalog → [00-architecture-catalog.md](00-architecture-catalog.md)
- Backbones → [01-backbones.md](01-backbones.md)
- Task definitions → [../03-tasks/](../03-tasks/)
