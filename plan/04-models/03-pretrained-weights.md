# Models — Pretrained Weights

This document describes how pretrained weights are managed in AI Studio.

---

## 1. Overview

Most models are initialized with **pretrained ImageNet weights** from torchvision. This provides faster convergence and better accuracy, especially with small datasets (transfer learning).

---

## 2. Weight Sources

| Source | Models | Mechanism |
|--------|--------|-----------|
| **torchvision weights API** | All torchvision models | `weights=ModelName_Weights.DEFAULT` |
| **torch hub cache** | Downloaded automatically | Stored in `~/.cache/torch/hub/checkpoints/` || **HuggingFace Transformers** | DINOv3 ViT, DINOv3 ConvNeXt | `AutoModel.from_pretrained("facebook/dinov3-...")` |
### torchvision Weights API (v2)

torchvision >= 0.13 uses a typed weights enum:

```python
from torchvision.models import ResNet50_Weights, resnet50

# Use latest pretrained weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Or explicitly
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# No pretrained weights
model = resnet50(weights=None)
```

### HuggingFace Weights

DINOv3 models are loaded via HuggingFace Transformers:

```python
from transformers import AutoModel

# DINOv3 ViT-Base
model = AutoModel.from_pretrained("facebook/dinov3-vitb14")

# DINOv3 ConvNeXt-Base
model = AutoModel.from_pretrained("facebook/dinov3-convnext-base")
```

Weights are cached at `~/.cache/huggingface/hub/` and shared across projects.

---

## 3. Cache Management

- torchvision downloads weights on first use and caches them at `~/.cache/torch/hub/checkpoints/`.
- HuggingFace downloads weights on first use and caches them at `~/.cache/huggingface/hub/`.
- These caches are **shared** across all projects and persist across restarts.
- No explicit cache cleanup is needed — the total size of common weights is ~1-2 GB for torchvision, ~2-5 GB for HuggingFace DINOv3 models.
- If internet access is unavailable, models must have been previously cached. A clear error message is shown if weights cannot be downloaded.

### Optional: Local Weight Mirror

For air-gapped environments (future feature):
- Allow configuring a local path containing weight files.
- The catalog would check the local path before attempting download.

---

## 4. Fine-Tuning vs. Freezing

Two modes of using pretrained weights:

### 4.1 Fine-Tuning (default)

- All model parameters are trainable.
- Backbone starts from pretrained weights but updates during training.
- The head is randomly initialized and trained from scratch.
- **When to use**: most cases. Works well with moderate-to-large datasets.

### 4.2 Backbone Freezing

- Backbone parameters have `requires_grad = False`.
- Only the head (classification/regression FC layer, or detection predictor) is trained.
- **When to use**: very small datasets (< 100 images), or when fine-tuning overfits.

### Implementation

```python
def apply_weight_config(model: nn.Module, config: ModelConfig):
    if config.freeze_backbone:
        # Freeze all backbone parameters
        for name, param in model.named_parameters():
            if not is_head_parameter(name):
                param.requires_grad = False
```

The `is_head_parameter()` function checks if a parameter name belongs to the head (e.g., `fc.`, `classifier.`, `roi_heads.`, `mask_predictor.`).

For DINOv3 models, backbone freezing is especially effective because the self-supervised pretraining produces highly transferable features. Freezing the DINOv3 backbone and training only the head is recommended for small datasets.

---

## 5. Configuration in Experiment JSON

```json
{
  "model": {
    "backbone": "resnet50",
    "pretrained": true,
    "freeze_backbone": false
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `pretrained` | `true` | Use pretrained weights |
| `freeze_backbone` | `false` | Freeze backbone layers |

---

## 6. Weight Compatibility

When loading pretrained weights for a model with a different number of classes:

1. Load the full pretrained model.
2. Replace the final layer(s) with the correct output size.
3. The new layer(s) are randomly initialized.

This is handled automatically by the model factory functions in the [architecture catalog](00-architecture-catalog.md).

---

## 7. GUI: Pretrained Controls

On the Training page (center column, model configuration section):

- **"Use pretrained weights"** toggle (default: on).
  - When on: tooltip explains "Model initialized with ImageNet weights for faster convergence."
  - When off: tooltip explains "Model initialized with random weights. Requires more training data."
- **"Freeze backbone"** toggle (default: off, only shown when pretrained is on).
  - When on: tooltip explains "Only the final layers will be trained. Use for very small datasets."
  - When off: tooltip explains "All layers will be fine-tuned."

---

## 8. Related Documents

- Architecture catalog → [00-architecture-catalog.md](00-architecture-catalog.md)
- Backbones → [01-backbones.md](01-backbones.md)
- Training pipeline → [../05-training/00-training-pipeline.md](../05-training/00-training-pipeline.md)
