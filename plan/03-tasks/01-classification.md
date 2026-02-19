# Tasks — Classification

**Phase**: 1 (first task to implement)

---

## 1. Task Description

Assign a single class label to an entire image. The model receives an image and outputs a probability distribution over the set of known classes.

---

## 2. Label Schema

```json
{
  "type": "label",
  "class_id": 0,
  "class_name": "cat"
}
```

- Each image has exactly one annotation of type `"label"`.
- Classes are stored in `dataset.json` → `classes` (ordered list, index = `class_id`).

---

## 3. Compatible Dataset Formats
These formats will be converted to the required `dataset.json` structure:

| Format | How Classes Are Defined |
|--------|------------------------|
| **COCO JSON** | `categories[].name` — one annotation per image |
| **CSV** | `label` column |
| **Folder structure** | Subfolders = class names (e.g., `images/cat/img01.jpg`) |

---

## 4. Compatible Architectures

Any backbone + FC classification head:

| Backbone | Parameters | Notes |
|----------|-----------|-------|
| ResNet-18 | 11.7M | Lightweight, fast training |
| ResNet-34 | 21.8M | Good balance |
| ResNet-50 | 25.6M | Standard baseline |
| EfficientNet-B0 | 5.3M | Parameter-efficient |
| EfficientNet-B3 | 12.2M | Higher accuracy |
| MobileNetV3-Small | 2.5M | Edge-optimized |
| MobileNetV3-Large | 5.5M | Edge-optimized, better accuracy |

### Head: FC Classifier

```
backbone features (N, C) → Dropout(p) → Linear(C, num_classes) → logits (N, num_classes)
```

- Global Average Pooling is applied to spatial features before the FC layer.
- Dropout rate is a hyperparameter (default `0.2`).

---

## 5. Loss Functions

| Loss | When to Use |
|------|------------|
| **CrossEntropyLoss** (default) | Standard multi-class classification |
| **FocalLoss** | Imbalanced classes — down-weights easy examples |
| **LabelSmoothingCrossEntropy** | Regularization — softens one-hot targets |

---

## 6. Metrics

| Metric | Role | Implementation |
|--------|------|----------------|
| **Accuracy** | Primary | `torchmetrics.Accuracy(task="multiclass")` |
| **F1 (macro)** | Secondary | `torchmetrics.F1Score(task="multiclass", average="macro")` |
| **Precision (macro)** | Secondary | `torchmetrics.Precision(task="multiclass", average="macro")` |
| **Recall (macro)** | Secondary | `torchmetrics.Recall(task="multiclass", average="macro")` |
| **Confusion Matrix** | Secondary | `torchmetrics.ConfusionMatrix(task="multiclass")` |
| **Per-class Accuracy** | Detail | Derived from confusion matrix |

- **Best model selection**: highest `val_accuracy`.
- **Early stopping**: monitor `val_loss` (patience configurable, default 10).

---

## 7. Default Augmentations

### Training

```json
[
  { "name": "RandomResizedCrop", "params": { "size": [224, 224], "scale": [0.8, 1.0] } },
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomRotation", "params": { "degrees": 15 } },
  { "name": "ColorJitter", "params": { "brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.05 } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

### Validation / Test

```json
[
  { "name": "Resize", "params": { "size": [256, 256] } },
  { "name": "CenterCrop", "params": { "size": [224, 224] } },
  { "name": "ToTensor", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

---

## 8. Default Hyperparameters

```json
{
  "optimizer": "adam",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "scheduler": "cosine",
  "warmup_epochs": 5,
  "batch_size": 32,
  "max_epochs": 50,
  "early_stopping_patience": 10,
  "loss": "cross_entropy",
  "dropout": 0.2,
  "pretrained": true,
  "freeze_backbone": false
}
```

---

## 9. Visualization

### Dataset Page — Annotation Overlay

- **Class label badge**: semi-transparent colored rectangle in the top-left corner of the thumbnail showing the class name.
- **Color**: each class is assigned a consistent color from a palette.

### Evaluation Page — Prediction Overlay

- **Predicted class badge**: top-left corner with predicted class + confidence percentage.
- **Color coding**: green border if correct, red border if incorrect.
- **Confidence bar**: small horizontal bar showing confidence value.

---

## 10. Evaluation Specifics

Per-image result JSON:

```json
{
  "filename": "img_001.png",
  "ground_truth": { "class_id": 0, "class_name": "cat" },
  "prediction": { "class_id": 0, "class_name": "cat", "confidence": 0.94 },
  "correct": true,
  "probabilities": { "cat": 0.94, "dog": 0.04, "bird": 0.02 }
}
```

Aggregate metrics:

```json
{
  "accuracy": 0.956,
  "f1_macro": 0.951,
  "precision_macro": 0.948,
  "recall_macro": 0.953,
  "confusion_matrix": [[38, 1, 1], [0, 40, 1], [2, 0, 37]],
  "per_class": {
    "cat": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 40 },
    "dog": { "precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 41 },
    "bird": { "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 39 }
  }
}
```

---

## 11. Related Documents

- Task registry → [00-task-registry.md](00-task-registry.md)
- Architecture catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
- Backbones → [../04-models/01-backbones.md](../04-models/01-backbones.md)
- Training pipeline → [../05-training/00-training-pipeline.md](../05-training/00-training-pipeline.md)
- Augmentations → [../05-training/02-augmentation.md](../05-training/02-augmentation.md)
