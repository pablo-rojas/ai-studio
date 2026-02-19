# Models — Architecture Catalog

This document describes how model architectures are registered, configured, and instantiated in AI Studio.

---

## 1. Overview

The architecture catalog is a registry that maps `(task, architecture_name)` to a factory function that returns a ready-to-use `nn.Module`. Users select from this catalog on the Training page.

---

## 2. Registry Structure

```python
# app/models/catalog.py

from typing import Callable, Dict, Tuple
from torch import nn
from app.schemas.training import ModelConfig

# Registry: (task, architecture_name) → factory function
ModelFactory = Callable[[ModelConfig, int], nn.Module]
# Parameters: (config, num_classes) → nn.Module

ARCHITECTURE_CATALOG: Dict[Tuple[str, str], ModelFactory] = {}

def register(task: str, name: str):
    """Decorator to register a model factory."""
    def decorator(fn: ModelFactory):
        ARCHITECTURE_CATALOG[(task, name)] = fn
        return fn
    return decorator

def create_model(task: str, architecture: str, config: ModelConfig, num_classes: int) -> nn.Module:
    """Instantiate a model from the catalog."""
    key = (task, architecture)
    if key not in ARCHITECTURE_CATALOG:
        raise ValueError(f"Unknown architecture: {architecture} for task: {task}")
    return ARCHITECTURE_CATALOG[key](config, num_classes)

def list_architectures(task: str) -> list[str]:
    """List available architecture names for a task."""
    return [name for (t, name) in ARCHITECTURE_CATALOG if t == task]
```

---

## 3. Catalog by Task

### Classification

| Architecture | Key | torchvision Model | Notes |
|-------------|-----|----------|-------|
| ResNet-18 | `resnet18` | `torchvision.models.resnet18` | Lightweight |
| ResNet-34 | `resnet34` | `torchvision.models.resnet34` | Mid-range |
| ResNet-50 | `resnet50` | `torchvision.models.resnet50` | Standard baseline |
| EfficientNet-B0 | `efficientnet_b0` | `torchvision.models.efficientnet_b0` | Parameter-efficient |
| EfficientNet-B3 | `efficientnet_b3` | `torchvision.models.efficientnet_b3` | Higher accuracy |
| MobileNetV3-Small | `mobilenet_v3_small` | `torchvision.models.mobilenet_v3_small` | Edge-optimized |
| MobileNetV3-Large | `mobilenet_v3_large` | `torchvision.models.mobilenet_v3_large` | Edge, better accuracy |

### Anomaly Detection

Anomaly detection uses a custom student–teacher pipeline (Uninformed Students, Bergmann et al. 2020) rather than torchvision models. The architecture is not yet fully finalised — it will be implemented from scratch and may evolve during development (Phase 22).

| Architecture | Key | Source | Notes |
|-------------|-----|--------|-------|
| Uninformed Students (ResNet-18) | `uninformed_students_resnet18` | Custom | Frozen ResNet-18 backbone → teacher distillation → student ensemble. See [02-anomaly-detection.md](../03-tasks/02-anomaly-detection.md). |

### Object Detection

| Architecture | Key | torchvision Model |
|-------------|-----|----------|
| Faster R-CNN (ResNet-50) | `fasterrcnn_resnet50` | `torchvision.models.detection.fasterrcnn_resnet50_fpn` |
| FCOS (ResNet-50) | `fcos_resnet50` | `torchvision.models.detection.fcos_resnet50_fpn` |
| RetinaNet (ResNet-50) | `retinanet_resnet50` | `torchvision.models.detection.retinanet_resnet50_fpn` |
| SSDLite (MobileNetV3) | `ssdlite_mobilenet_v3` | `torchvision.models.detection.ssdlite320_mobilenet_v3_large` |

### Oriented Object Detection

To be defined.

### Segmentation

| Architecture | Key | torchvision Model |
|-------------|-----|----------|
| FCN (ResNet-50) | `fcn_resnet50` | `torchvision.models.segmentation.fcn_resnet50` |
| FCN (ResNet-101) | `fcn_resnet101` | `torchvision.models.segmentation.fcn_resnet101` |
| DeepLabV3 (ResNet-50) | `deeplabv3_resnet50` | `torchvision.models.segmentation.deeplabv3_resnet50` |
| DeepLabV3 (ResNet-101) | `deeplabv3_resnet101` | `torchvision.models.segmentation.deeplabv3_resnet101` |
| DeepLabV3 (MobileNetV3) | `deeplabv3_mobilenet` | `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` |
| LRASPP (MobileNetV3) | `lraspp_mobilenet` | `torchvision.models.segmentation.lraspp_mobilenet_v3_large` |

### Instance Segmentation

| Architecture | Key | torchvision Model |
|-------------|-----|----------|
| Mask R-CNN (ResNet-50) | `maskrcnn_resnet50` | `torchvision.models.detection.maskrcnn_resnet50_fpn` |

### Regression

Same architectures as classification, but with a regression head (`num_outputs` neurons) instead of FC classifier.

---

## 4. Model Configuration Schema

```json
{
  "backbone": "resnet50",
  "head": "classification",
  "pretrained": true,
  "freeze_backbone": false,
  "dropout": 0.2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `backbone` | string | Backbone architecture key |
| `head` | string | Task type (determines which head to use) |
| `pretrained` | bool | Whether to use pretrained ImageNet weights |
| `freeze_backbone` | bool | Whether to freeze backbone parameters during training |
| `dropout` | float | Dropout rate before final layer (classification/regression only) |

---

## 5. Factory Example (Classification)

```python
@register("classification", "resnet50")
def create_resnet50_classifier(config: ModelConfig, num_classes: int) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if config.pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    
    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(in_features, num_classes)
    )
    
    return model
```

---

## 6. GUI: Architecture Selection

On the Training page (center column), the architecture selector shows:

1. **Dropdown** filtered by the project's task type.
2. Each option shows: name, parameter count, brief note (e.g., "Lightweight", "Standard baseline").
3. On selection, the default hyperparameters for that architecture are populated.
4. **Backbone freeze toggle** checkbox.
5. **Pretrained toggle** checkbox.

---

## 7. Related Documents

- Backbones → [01-backbones.md](01-backbones.md)
- Task-specific heads → [02-heads.md](02-heads.md)
- Pretrained weights → [03-pretrained-weights.md](03-pretrained-weights.md)
- Task definitions → [../03-tasks/](../03-tasks/)
