# Training — Training Pipeline

This document describes the end-to-end training pipeline, built on PyTorch Lightning.

---

## 1. Pipeline Overview

```
Experiment Config (experiment.json)
           │
           ▼
   ┌───────────────┐
   │ Trainer Factory│  → builds pl.Trainer from config
   └───────┬───────┘
           │
   ┌───────┴───────┐
   │                │
   ▼                ▼
LightningModule   LightningDataModule
   │                │
   │ (model +       │ (dataset +
   │  loss +        │  dataloader +
   │  optimizer +   │  augmentations)
   │  metrics)      │
   └───────┬───────┘
           │
           ▼
     trainer.fit(module, datamodule)
           │
           ▼
   Per-epoch: metrics → JSON logger → metrics.json
   Checkpoints: best.ckpt, last.ckpt
   Completion: run.json updated with final status
```

---

## 2. LightningModule (`app/training/lightning_module.py`)

A single generic `LightningModule` handles all tasks:

```python
class AIStudioModule(pl.LightningModule):
    def __init__(self, model, loss_fn, metrics, optimizer_config, scheduler_config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.train_metrics.update(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.val_metrics.update(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
    
    def configure_optimizers(self):
        optimizer = build_optimizer(self.parameters(), self.optimizer_config)
        scheduler = build_scheduler(optimizer, self.scheduler_config)
        return {"optimizer": optimizer, "lr_schedulers": scheduler}
```

### Detection/Instance Segmentation Variation

torchvision detection models compute losses internally. The training step differs:

```python
def training_step(self, batch, batch_idx):
    images, targets = batch
    loss_dict = self.model(images, targets)  # Returns dict of losses
    loss = sum(loss_dict.values())
    self.log("train_loss", loss, prog_bar=True)
    return loss

def validation_step(self, batch, batch_idx):
    images, targets = batch
    self.model.eval()
    predictions = self.model(images)  # Returns predictions in eval mode
    # Compute metrics on predictions vs targets
    self.val_metrics.update(predictions, targets)
```

---

## 3. LightningDataModule (`app/datasets/base.py`)

```python
class AIStudioDataModule(pl.LightningDataModule):
    def __init__(self, project_path, split_index, task, augmentation_config, batch_size, num_workers=4):
        super().__init__()
        self.project_path = project_path
        self.split_index = split_index
        self.task = task
        self.augmentation_config = augmentation_config
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        dataset_json = load_dataset(self.project_path)
        images = dataset_json["images"]
        
        train_images = [img for img in images if img["split"][self.split_index] == "train"]
        val_images = [img for img in images if img["split"][self.split_index] == "val"]
        test_images = [img for img in images if img["split"][self.split_index] == "test"]
        
        train_transform = build_augmentation_pipeline(self.augmentation_config["train"])
        val_transform = build_augmentation_pipeline(self.augmentation_config["val"])
        
        dataset_cls = get_dataset_class(self.task)
        self.train_dataset = dataset_cls(train_images, train_transform)
        self.val_dataset = dataset_cls(val_images, val_transform)
        self.test_dataset = dataset_cls(test_images, val_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers,
                         collate_fn=get_collate_fn(self.task))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         collate_fn=get_collate_fn(self.task))
```

---

## 4. Trainer Factory (`app/training/trainer_factory.py`)

```python
def build_trainer(run_dir: Path, experiment_config: dict) -> pl.Trainer:
    hw = experiment_config["hardware"]
    hp = experiment_config["hyperparameters"]
    
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=hp["early_stopping_patience"],
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        JSONMetricLogger(output_path=run_dir / "metrics.json"),
    ]
    
    # Resolve device selection into Lightning args
    resolved = resolve_hardware(hw["selected_devices"])
    precision = hw.get("precision", "32")
    
    return pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=hp["max_epochs"],
        accelerator=resolved["accelerator"],
        devices=resolved["devices"],
        strategy=resolved["strategy"],
        precision=precision,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
```

---

## 5. Training Run Lifecycle

```
                    ┌─────────┐
                    │ pending │  ← experiment created, not started
                    └────┬────┘
                         │  user clicks "Run"
                         ▼
                    ┌─────────┐
                    │ running │  ← trainer.fit() in background task
                    └────┬────┘
                    ┌────┴────┐
                    │         │
                    ▼         ▼
             ┌───────────┐ ┌────────┐
             │ completed │ │ failed │  ← exception during training
             └───────────┘ └────────┘
                              │
                              ▼ (user can retry)
                         ┌─────────┐
                         │ pending │
                         └─────────┘
```

Status transitions are written to `run.json` immediately.

---

## 6. Background Execution

Training is long-running. It runs as a background task:

```python
# In the API router
@router.post("/api/training/{experiment_id}/run")
async def start_training(experiment_id: str, background_tasks: BackgroundTasks):
    run_id = create_run(experiment_id)
    background_tasks.add_task(execute_training, experiment_id, run_id)
    return {"run_id": run_id, "status": "pending"}
```

### Progress Reporting

The GUI polls for updates or uses SSE (Server-Sent Events) to receive live updates:

1. The `JSONMetricLogger` callback writes to `metrics.json` after each epoch.
2. An SSE endpoint `/api/training/{run_id}/stream` tails `metrics.json` and emits events.
3. The Training page's right column subscribes to the SSE stream and updates charts in real-time.

---

## 7. Concurrency

- Only **one training run** should be active at a time per project (GPU resource management).
- If a training run is already active, the "Run" button is disabled with a message.
- A global lock or semaphore prevents concurrent training starts.

---

## 8. Artifacts

Each completed run produces:

| Artifact | Location | Purpose |
|----------|----------|---------|
| `run.json` | `runs/<run-id>/` | Run metadata, status, final metrics |
| `config.json` | `runs/<run-id>/` | Frozen copy of experiment config |
| `metrics.json` | `runs/<run-id>/` | Per-epoch metrics array |
| `best.ckpt` | `runs/<run-id>/checkpoints/` | Best model checkpoint (lowest val_loss) |
| `last.ckpt` | `runs/<run-id>/checkpoints/` | Last epoch checkpoint |
| `training.log` | `runs/<run-id>/logs/` | Detailed training log |

---

## 9. Related Documents

- Hyperparameters → [01-hyperparameters.md](01-hyperparameters.md)
- Augmentations → [02-augmentation.md](02-augmentation.md)
- Multi-GPU → [03-multi-gpu.md](03-multi-gpu.md)
- Callbacks → [04-callbacks-logging.md](04-callbacks-logging.md)
- Experiment tracking → [../06-experiment-tracking/00-run-management.md](../06-experiment-tracking/00-run-management.md)
- Training GUI page → [../10-gui/01-pages/04-training-page.md](../10-gui/01-pages/04-training-page.md)
