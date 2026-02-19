# Training — Hyperparameters

This document defines the hyperparameter configuration schema, defaults per task, and how users configure them in the GUI.

---

## 1. Hyperparameter Schema

All hyperparameters for an experiment are stored in `experiment.json` under the `hyperparameters` key.

```json
{
  "hyperparameters": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "step_size": 10,
    "gamma": 0.1,
    "poly_power": 0.9,
    "batch_size": 32,
    "max_epochs": 50,
    "early_stopping_patience": 10,
    "loss": "cross_entropy",
    "dropout": 0.2
  }
}
```

---

## 2. Parameter Definitions

### 2.1 Optimizer

| Value | Class | Additional Params |
|-------|-------|------------------|
| `"adam"` | `torch.optim.Adam` | `learning_rate`, `weight_decay` |
| `"adamw"` | `torch.optim.AdamW` | `learning_rate`, `weight_decay` |
| `"sgd"` | `torch.optim.SGD` | `learning_rate`, `weight_decay`, `momentum` |

### 2.2 Learning Rate Scheduler

| Value | Class | Additional Params |
|-------|-------|------------------|
| `"cosine"` | `CosineAnnealingLR` | `T_max` = `max_epochs` |
| `"step"` | `StepLR` | `step_size`, `gamma` |
| `"multistep"` | `MultiStepLR` | `milestones` (list), `gamma` |
| `"poly"` | `PolynomialLR` | `poly_power` |
| `"none"` | No scheduler | Constant LR |

Optional **warmup**: if `warmup_epochs > 0`, use `LinearWarmup` scheduler for the first N epochs, then switch to the selected scheduler.

### 2.3 Loss Functions

See individual task documents for available loss functions per task:
- [Classification](../03-tasks/01-classification.md)
- [Anomaly Detection](../03-tasks/02-anomaly-detection.md)
- [Object Detection](../03-tasks/03-object-detection.md)
- [Segmentation](../03-tasks/05-segmentation.md)
- [Regression](../03-tasks/07-regression.md)

### 2.4 General Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `batch_size` | int | 1–256 | Mini-batch size (constrained by GPU memory) |
| `max_epochs` | int | 1–1000 | Maximum training epochs |
| `early_stopping_patience` | int | 0–100 | Epochs without improvement before stopping (0 = disabled) |
| `dropout` | float | 0.0–0.9 | Dropout rate in the head |

---

## 3. Defaults Per Task

### Classification / Anomaly Detection

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
  "dropout": 0.2
}
```

### Object Detection / Oriented OD / Instance Segmentation

```json
{
  "optimizer": "sgd",
  "learning_rate": 0.005,
  "weight_decay": 0.0005,
  "momentum": 0.9,
  "scheduler": "step",
  "step_size": 10,
  "gamma": 0.1,
  "batch_size": 8,
  "max_epochs": 50,
  "early_stopping_patience": 15,
  "loss": "default"
}
```

### Segmentation

```json
{
  "optimizer": "sgd",
  "learning_rate": 0.01,
  "weight_decay": 0.0001,
  "momentum": 0.9,
  "scheduler": "poly",
  "poly_power": 0.9,
  "batch_size": 8,
  "max_epochs": 60,
  "early_stopping_patience": 15,
  "loss": "ce_dice"
}
```

### Regression

```json
{
  "optimizer": "adam",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "scheduler": "cosine",
  "batch_size": 32,
  "max_epochs": 50,
  "early_stopping_patience": 10,
  "loss": "mse",
  "dropout": 0.2
}
```

---

## 4. Optimizer Builder

```python
def build_optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    name = config["optimizer"]
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=wd,
                               momentum=config.get("momentum", 0.9))
```

---

## 5. GUI: Hyperparameter Form

In the Training page center column, the hyperparameter form includes:

| Control | Type | Notes |
|---------|------|-------|
| Optimizer | Dropdown | adam / adamw / sgd |
| Learning Rate | Number input | Scientific notation support (e.g., 1e-3) |
| Weight Decay | Number input | |
| Momentum | Number input | Only shown when optimizer = sgd |
| Scheduler | Dropdown | cosine / step / poly / none |
| Warmup Epochs | Number input | |
| Step Size / Gamma | Number inputs | Only shown when scheduler = step |
| Batch Size | Number input | Powers of 2 recommended |
| Max Epochs | Number input | |
| Early Stopping | Number input | 0 to disable |
| Loss Function | Dropdown | Filtered by task |
| Dropout | Slider | 0.0 – 0.9 |

When the user selects a different architecture, defaults are repopulated (overwriting custom values unless the user has explicitly modified them).

---

## 6. Related Documents

- Training pipeline → [00-training-pipeline.md](00-training-pipeline.md)
- Augmentations → [02-augmentation.md](02-augmentation.md)
- Task-specific losses and metrics → [../03-tasks/](../03-tasks/)
- Training GUI page → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
