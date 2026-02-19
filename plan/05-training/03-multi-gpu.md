# Training — Multi-GPU

This document describes how AI Studio supports multi-GPU training via PyTorch Lightning.

---

## 1. Overview

PyTorch Lightning abstracts multi-GPU training via **strategies**. AI Studio exposes a unified device selector in the experiment settings — the strategy is determined automatically based on the number of selected GPUs.

---

## 2. Supported Configurations

The configuration is derived automatically from the user's device selection:

| User Selection | Resolved Config | Use Case |
|----------------|-----------------|----------|
| **GPU 0 only** | `accelerator="gpu"`, `devices=[0]`, `strategy="auto"` | Default — one GPU |
| **GPU 0 + GPU 1** | `accelerator="gpu"`, `devices=[0,1]`, `strategy="ddp"` | Data-parallel across GPUs |
| **CPU only** | `accelerator="cpu"`, `devices="auto"`, `strategy="auto"` | Fallback when no GPU available |
| **GPU 0 + CPU** | `accelerator="gpu"`, `devices=[0]`, `strategy="auto"` | CPU is silently ignored |

---

## 3. Hardware Configuration Schema

```json
{
  "hardware": {
    "selected_devices": ["gpu:0"],
    "precision": "32"
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `selected_devices` | list[string] | `["gpu:0"]` | Devices selected by the user. Values: `"cpu"`, `"gpu:0"`, `"gpu:1"`, etc. If GPU 0 is available, default is `["gpu:0"]`; otherwise `["cpu"]`. |
| `precision` | string | `"32"` | Training precision: `"32"`, `"16-mixed"`, `"bf16-mixed"` |

The `accelerator`, `devices`, and `strategy` fields passed to Lightning's `Trainer` are **resolved at runtime** from `selected_devices` (see §8).

---

## 4. GPU Detection

At application startup, detect available GPUs:

```python
import torch

def get_available_gpus() -> list[dict]:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_mem / 1e9, 1),
            })
    return gpus
```

This information is displayed in the GUI and used to validate device selection.

---

## 5. DDP Considerations

### 5.1 Effective Batch Size

With DDP, the effective batch size = `batch_size × num_gpus`. The user should be aware:
- If `batch_size=32` and `devices=[0, 1]`, effective batch size is 64.
- Learning rate may need to be scaled accordingly (linear scaling rule).

The GUI shows a note: *"Effective batch size: 64 (32 × 2 GPUs)"*.

### 5.2 Data Loading

- Each DDP process gets a shard of the data via `DistributedSampler` (handled automatically by Lightning).
- `num_workers` is per-process — total data loading threads = `num_workers × num_gpus`.

### 5.3 Logging

- Only rank 0 writes to `metrics.json` and `run.json`.
- The `JSONMetricLogger` callback checks `self.trainer.is_global_zero` before writing.

### 5.4 Checkpointing

- Checkpoints are saved by rank 0 only (Lightning default).
- All processes synchronize before checkpoint saving.

---

## 6. Automatic Strategy Resolution

| Selected Devices | Resolved Strategy | Notes |
|------------------|-------------------|-------|
| 1 GPU | `"auto"` | Lightning selects the best single-GPU strategy |
| 2+ GPUs | `"ddp"` | Standard DDP, one process per GPU |
| CPU only | `"auto"` | Single-process CPU training |
| GPU(s) + CPU | Same as GPU-only | CPU entry is silently dropped |

The user never selects a strategy directly — it is inferred from the device selection.

---

## 7. GUI: Hardware Configuration

In the Training page center column, a "Hardware" section:

| Control | Type | Notes |
|---------|------|-------|
| Device selector | Multi-select dropdown with checkboxes | Lists **all** detected devices: CPU and each GPU (name + memory). Default: GPU 0 checked if available, otherwise CPU checked. |
| Precision | Dropdown | `32` / `16-mixed` / `bf16-mixed` |
| Effective batch size | Read-only label | Shown when 2+ GPUs selected. Computed: `batch_size × num_selected_gpus`. |

**Selection rules:**
- If both GPU(s) and CPU are checked, CPU is silently ignored (a subtle note informs the user).
- If 2+ GPUs are checked, a note appears: *"Multi-GPU (DDP) will be used automatically."*
- If no device is checked, the "Run" button is disabled.

If no GPU is detected, the GPU entries are absent and only CPU is listed (pre-selected).

---

## 8. Trainer Integration

```python
def resolve_hardware(selected_devices: list[str]) -> dict:
    """Resolve user device selection into Lightning Trainer kwargs."""
    gpu_indices = [int(d.split(":")[1]) for d in selected_devices if d.startswith("gpu:")]
    has_cpu = "cpu" in selected_devices
    
    # If any GPU is selected, ignore CPU
    if gpu_indices:
        accelerator = "gpu"
        devices = gpu_indices
        strategy = "ddp" if len(gpu_indices) > 1 else "auto"
    elif has_cpu:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"
    else:
        raise ValueError("No device selected")
    
    return {"accelerator": accelerator, "devices": devices, "strategy": strategy}


def build_trainer(run_dir, config):
    hw = config["hardware"]
    resolved = resolve_hardware(hw["selected_devices"])
    precision = hw.get("precision", "32")
    
    return pl.Trainer(
        accelerator=resolved["accelerator"],
        devices=resolved["devices"],
        strategy=resolved["strategy"],
        precision=precision,
        # ... other settings
    )
```

---

## 9. Related Documents

- Training pipeline → [00-training-pipeline.md](00-training-pipeline.md)
- Hyperparameters (batch size) → [01-hyperparameters.md](01-hyperparameters.md)
- Training GUI page → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
