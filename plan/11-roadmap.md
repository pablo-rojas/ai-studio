# Roadmap — Phased Rollout

This document describes the phased development plan for AI Studio.
Each phase is small and self-contained — designed for iterative vibe coding.
Phases are grouped so that **backend → API → GUI** for each feature area are
consecutive, letting you view and test each feature end-to-end before moving on.

---

## Phase Overview

| Phase | Focus | Target |
|-------|-------|--------|
| **Phase 1** | Core Platform & Storage | Project CRUD, JSON persistence, workspace folder structure |
| **Phase 2** | Dataset Management | Import, thumbnails, browsing logic |
| **Phase 3** | Splits | Stratified split creation |
| **Phase 4** | API — Projects, Datasets & Splits | REST endpoints for Phases 1–3 |
| **Phase 5** | GUI — Project Page | Project list, create/delete/rename, task selection |
| **Phase 6** | GUI — Dataset Page | Image grid, pagination, filtering, sorting, detail view |
| **Phase 7** | GUI — Split Page | Split creation form, preview, immutability display |
| **Phase 8** | Classification Task | Task definition, model catalog, loss, metrics, augmentations |
| **Phase 9** | Training Pipeline | LightningModule, DataModule, Trainer, single-GPU, SSE progress |
| **Phase 10** | Experiment Tracking | Experiment CRUD, training management, metrics logging, checkpoints |
| **Phase 11** | API — Training & Experiments | REST endpoints for Phases 8–10 |
| **Phase 12** | GUI — Training Page | Experiment config, start training, live loss curves |
| **Phase 13** | Evaluation Pipeline | Test-set evaluation, per-image results, aggregate metrics |
| **Phase 14** | API — Evaluation | REST endpoints for Phase 13 |
| **Phase 15** | GUI — Evaluation Page | Metrics display, per-image results, confusion matrix |
| **Phase 16** | ONNX Export | Export with validation |
| **Phase 17** | API — Export | REST endpoints for Phase 16 |
| **Phase 18** | GUI — Export Page | Export form, download link, validation status |
| **Phase 19** | Object Detection — Core | Task definition, models, COCO/YOLO dataset adapter, metrics, eval pipeline |
| **Phase 20** | Object Detection — GUI | Dataset page bbox overlays, training page hparam fields, eval page bbox+AP |
| **Phase 21** | Semantic Segmentation — Core | Task definition, models, mask handling, loss, metrics, eval pipeline |
| **Phase 22** | Semantic Segmentation — GUI | Mask overlays on dataset page, predicted/GT masks + error map on eval page |
| **Phase 23** | Instance Segmentation | Task definition, Mask R-CNN, COCO instance format, per-instance overlays |
| **Phase 24** | DINOv3 Models | DINOv3-ConvNeXt & DINOv3-ViT backbones via HuggingFace Transformers |
| **Phase 25** | Anomaly Detection | Task definition, metrics, evaluation adaptations |
| **Phase 26** | Regression | Task definition, target normalization, scatter plots |
| **Phase 27** | Oriented Object Detection | Angle regression, rotated IoU, DOTA/YOLO-OBB import |
| **Phase 28** | Polish & Extras | Multi-GPU, extra exports, advanced features |

---

<!-- ────────────────────────────────────────────────────────────────
     GROUP A — Data Foundation  (backend → API → GUI)
     ──────────────────────────────────────────────────────────── -->

## Phase 1 — Core Platform & Storage

**Goal**: Establish the project skeleton and persistence layer.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Workspace structure** | Create workspace root with standard folder layout |
| **JSON persistence** | Read/write helpers for all JSON metadata files |
| **Project CRUD** | Create, delete, rename projects; store task type per project |
| **Config schema** | Define all JSON schemas (project, dataset, experiment, etc.) |

### Acceptance Criteria

- [ ] Workspace folder is created on first launch
- [ ] Projects can be created, listed, renamed, and deleted
- [ ] All data is persisted as JSON and survives restarts

---

## Phase 2 — Dataset Management

**Goal**: Import datasets and manage image metadata.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Import — folder structure** | Read class folders, copy images, generate metadata |
| **Import — COCO JSON** | Parse COCO JSON and map to internal format |
| **Import — CSV** | Parse CSV with image paths and labels |
| **Thumbnails** | Generate thumbnails on import |
| **Image metadata** | Store per-image metadata (path, size, label, hash) |

### Acceptance Criteria

- [ ] A folder-structured dataset (e.g., cats vs dogs) can be imported
- [ ] Images are copied into the workspace and thumbnails are generated
- [ ] Dataset metadata JSON is created and queryable

---

## Phase 3 — Splits

**Goal**: Create and manage train/val/test splits.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Stratified split** | Partition images by ratio preserving class distribution |
| **Split metadata** | Persist split assignments, mark splits as immutable |
| **Preview** | Return class distribution per split before confirming |

### Acceptance Criteria

- [ ] An 80/10/10 stratified split can be created
- [ ] Split preview shows class distribution before saving
- [ ] Saved splits are immutable

---

## Phase 4 — API: Projects, Datasets & Splits

**Goal**: Expose project, dataset, and split functionality via REST endpoints so the first GUI pages can be built on top.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Project endpoints** | CRUD for projects |
| **Dataset endpoints** | Import, list images, get image detail, thumbnails |
| **Split endpoints** | Create split, preview, get split info |
| **Error handling** | Consistent error response format |

### Acceptance Criteria

- [ ] Projects can be created, listed, renamed, and deleted via API
- [ ] A dataset can be imported and browsed via API
- [ ] Splits can be created and queried via API
- [ ] Error responses follow a consistent JSON schema
- [ ] API can be exercised end-to-end with curl/httpie

---

## Phase 5 — GUI: Project Page

**Goal**: Build the project management page.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Project list** | Display all projects with task type badges |
| **Create project modal** | Name input + task type selector |
| **Delete / rename** | Inline actions on project cards |
| **Navigation** | Clicking a project navigates to its dataset page |

### Acceptance Criteria

- [ ] Projects are listed on the home page
- [ ] A new project can be created from the modal
- [ ] Project can be renamed and deleted from the UI

---

## Phase 6 — GUI: Dataset Page

**Goal**: Build the dataset browsing page.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Image grid** | Thumbnail grid with lazy loading |
| **Pagination** | Page controls for large datasets |
| **Filtering & sorting** | Filter by class, sort by name/date |
| **Image detail view** | Click-to-expand with full metadata |
| **Import trigger** | Button to start dataset import flow |

### Acceptance Criteria

- [ ] Imported images appear in a paginated grid
- [ ] Images can be filtered by class label
- [ ] Clicking a thumbnail shows the detail view

---

## Phase 7 — GUI: Split Page

**Goal**: Build the split creation and display page.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Split form** | Ratio sliders, seed input |
| **Preview panel** | Show class distribution per split before confirming |
| **Split summary** | Display existing split with sample counts |
| **Immutability indicator** | Visual badge showing split is locked |

### Acceptance Criteria

- [ ] User can configure and preview a split
- [ ] After creation, the split summary is displayed
- [ ] Locked splits show an immutability indicator

---

<!-- ────────────────────────────────────────────────────────────────
     GROUP B — Training & Experiments  (backend → API → GUI)
     ──────────────────────────────────────────────────────────── -->

## Phase 8 — Classification Task

**Goal**: Define the classification task with models, losses, metrics, and augmentations.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Task definition** | `ClassificationTask` registered in the task registry |
| **Model catalog** | ResNet-18/34/50, EfficientNet-B0/B3, MobileNetV3-Small/Large |
| **Loss functions** | CrossEntropyLoss (with label smoothing option) |
| **Metrics** | Accuracy, precision, recall, F1, confusion matrix |
| **Augmentation defaults** | Classification-specific transform pipeline |
| **Hyperparameter defaults** | Default optimizer, scheduler, LR, batch size for classification |

### Acceptance Criteria

- [ ] `ClassificationTask` is registered and selectable
- [ ] All 6 backbone architectures are available in the model catalog
- [ ] Default hyperparameters produce a valid training config

---

## Phase 9 — Training Pipeline

**Goal**: Build the training engine that runs experiments.

### Deliverables

| Component | Scope |
|-----------|-------|
| **LightningModule** | Generic training module parameterized by task |
| **LightningDataModule** | Loads dataset + split, applies augmentations |
| **Trainer factory** | Builds PyTorch Lightning Trainer from config |
| **Background training** | Run training in a subprocess, non-blocking |
| **SSE progress** | Stream epoch/loss/metric updates to clients via SSE |
| **Hyperparameter config** | Full optimizer/scheduler/loss configuration |

### Acceptance Criteria

- [ ] Training can be started and runs in the background
- [ ] Live loss and metric values are streamed via SSE
- [ ] Training can be stopped gracefully

---

## Phase 10 — Experiment Tracking

**Goal**: Manage experiments, training execution, metrics, and checkpoints.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Experiment CRUD** | Create, list, delete experiments |
| **Training management** | Each experiment has a single training execution |
| **Metrics logging** | Per-epoch metrics saved to JSON |
| **Checkpoints** | Save best + last checkpoints, track paths |

### Acceptance Criteria

- [ ] Experiments are listed with their status
- [ ] Per-epoch metrics are persisted and retrievable
- [ ] Best checkpoint path is recorded per experiment

---

## Phase 11 — API: Training & Experiments

**Goal**: Expose training and experiment management via REST endpoints.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Training endpoints** | Start/stop training, SSE stream |
| **Experiment endpoints** | CRUD experiments, start/stop training, get metrics |

### Acceptance Criteria

- [ ] Training can be started, monitored (SSE), and stopped via API
- [ ] Experiments can be listed with metrics via API
- [ ] API can be exercised end-to-end with curl/httpie

---

## Phase 12 — GUI: Training Page

**Goal**: Build the experiment configuration and training page.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Experiment config form** | Model selector, hyperparameter inputs, augmentation toggles |
| **Start/stop training** | Button to launch training, button to stop |
| **Live loss curves** | Chart updated via SSE (Chart.js or similar) |
| **Experiment list** | List experiments with status and final metrics |

### Acceptance Criteria

- [ ] User can configure and start training an experiment
- [ ] Loss curves update in real time
- [ ] Experiments are listed with status and metrics

---

<!-- ────────────────────────────────────────────────────────────────
     GROUP C — Evaluation  (backend → API → GUI)
     ──────────────────────────────────────────────────────────── -->

## Phase 13 — Evaluation Pipeline

**Goal**: Evaluate trained models on test/validation sets. Evaluation is 1:1 with an experiment — data lives inside the experiment folder.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Evaluator** | Load checkpoint, build combined dataloader from selected split subsets, run inference |
| **Per-image results** | Store prediction + confidence + subset tag per image in `results.json` |
| **Aggregate metrics** | Compute task-specific metrics over the combined pool, store in `aggregate.json` |
| **Confusion matrix** | Generate confusion matrix data (classification) |
| **Evaluation service** | CRUD: start evaluation, get status, get results, reset (delete), list checkpoints |
| **Storage paths** | New path helpers for `experiments/<exp-id>/evaluation/` subfolder |
| **Pydantic schemas** | `EvaluationConfig`, `EvaluationRecord`, per-image result models |

### Acceptance Criteria

- [ ] Best checkpoint can be evaluated on selected split subsets
- [ ] Per-image predictions with confidence scores and subset tags are stored
- [ ] Aggregate metrics (including confusion matrix) are computed and stored
- [ ] Evaluation data lives inside the experiment folder (`evaluation/` subfolder)
- [ ] Reset deletes the evaluation subfolder immediately
- [ ] Checkpoint discovery only lists existing `.ckpt` files

---

## Phase 14 — API: Evaluation

**Goal**: Expose evaluation functionality via REST endpoints. Evaluation is scoped to an experiment (no separate evaluation ID).

### Deliverables

| Component | Scope |
|-----------|-------|
| **Start evaluation** | `POST /api/evaluation/{project_id}/{experiment_id}` |
| **Get evaluation** | `GET /api/evaluation/{project_id}/{experiment_id}` (status + aggregate) |
| **Reset evaluation** | `DELETE /api/evaluation/{project_id}/{experiment_id}` (immediate delete) |
| **Per-image results** | `GET /api/evaluation/{project_id}/{experiment_id}/results` (paginated, filterable by subset) |
| **List checkpoints** | `GET /api/evaluation/{project_id}/{experiment_id}/checkpoints` |

### Acceptance Criteria

- [ ] Evaluation can be triggered and results retrieved via API
- [ ] Reset endpoint immediately deletes evaluation data
- [ ] Results endpoint supports filtering by subset, correctness, and class
- [ ] Checkpoints endpoint only returns existing checkpoint files
- [ ] API can be exercised end-to-end with curl/httpie

---

## Phase 15 — GUI: Evaluation Page

**Goal**: Build the evaluation page with 2-column layout (experiment list + top non-collapsible cards + collapsible per-image section).

### Deliverables

| Component | Scope |
|-----------|-------|
| **Left panel** | Experiment list (completed only), identical style to Training page |
| **Top Card 1: Hardware + Config** | Checkpoint dropdown (existing only), split subset checklist, batch size, device, Evaluate/Reset buttons |
| **Top Card 2: Metrics** | Aggregate metrics table, confusion matrix heatmap (Chart.js), per-class bar chart |
| **Bottom Section: Per-image** | Thumbnail grid with ✓/✗ borders, filters (correct/incorrect, class, subset), sort, pagination, detail view |
| **Collapsible UI** | Alpine.js collapse only for the per-image section |

### Acceptance Criteria

- [ ] Left panel only shows completed experiments
- [ ] Evaluation can be triggered from the UI with configurable subsets
- [ ] Confusion matrix heatmap is displayed
- [ ] Per-image results show predictions with confidence scores
- [ ] Reset immediately clears evaluation and returns to config state
- [ ] Checkpoint selector only shows existing checkpoint files

---

<!-- ────────────────────────────────────────────────────────────────
     GROUP D — Export  (backend → API → GUI)
     ──────────────────────────────────────────────────────────── -->

## Phase 16 — ONNX Export

**Goal**: Export trained models to ONNX format.

### Deliverables

| Component | Scope |
|-----------|-------|
| **ONNX export** | `torch.onnx.export` from checkpoint |
| **Validation** | Run ONNX Runtime inference and compare outputs |
| **Metadata** | Store export config (opset, input shape, etc.) |

### Acceptance Criteria

- [ ] A trained model can be exported to ONNX
- [ ] Exported model passes ONNX Runtime validation
- [ ] Export metadata is saved alongside the file

---

## Phase 17 — API: Export

**Goal**: Expose export functionality via REST endpoints.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Export endpoints** | Trigger export, download file |

### Acceptance Criteria

- [ ] Export can be triggered and file downloaded via API
- [ ] API can be exercised end-to-end with curl/httpie

---

## Phase 18 — GUI: Export Page

**Goal**: Build the export page.

### Deliverables

| Component | Scope |
|-----------|-------|
| **Export form** | Select experiment/checkpoint, choose format (ONNX) |
| **Export status** | Show progress and validation result |
| **Download link** | Direct download of the exported file |

### Acceptance Criteria

- [ ] User can select a checkpoint and trigger ONNX export
- [ ] Validation result is displayed
- [ ] Exported file can be downloaded

---

<!-- ────────────────────────────────────────────────────────────────
     GROUP E — Additional Tasks  (each includes backend + GUI adaptations)
     ──────────────────────────────────────────────────────────── -->

## Phase 19 — Object Detection: Core

**Goal**: Add the object detection task backend — models, data handling, training, and evaluation logic.

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `ObjectDetectionTask` registered in task registry |
| Models | Faster R-CNN, FCOS, RetinaNet, SSD, SSDLite (from torchvision) |
| Dataset adapter | COCO bbox format + YOLO format parser |
| Collate function | Custom collate returning `(images_list, targets_list)` for variable-length targets |
| Augmentations | Joint image+bbox transforms (torchvision.transforms.v2) |
| Loss | Built into torchvision model forward pass |
| Metrics | mAP@0.5, mAP@0.5:0.95 via torchmetrics |
| Evaluation pipeline | Per-image: TP/FP/FN boxes. Aggregate: per-class AP. |

### Acceptance Criteria

- [ ] Object detection project can be created and trained end-to-end (CLI/API)
- [ ] COCO-format dataset can be imported
- [ ] YOLO-format dataset can be imported
- [ ] mAP metrics are computed correctly

---

## Phase 20 — Object Detection: GUI

**Goal**: Add UI adaptations for the object detection task across all relevant pages.

### Deliverables

| Component | Change |
|-----------|--------|
| Dataset page | Bbox overlay on thumbnails and detail view (canvas-overlay.js) |
| Training page | Detection-specific hyperparameter fields (NMS threshold, score threshold) |
| Evaluation page | Bbox overlays with TP/FP/FN coloring, per-class AP bar chart |

### Acceptance Criteria

- [ ] Bounding boxes are drawn on images in the dataset page
- [ ] Detection-specific hyperparameters are configurable in the training UI
- [ ] Evaluation page shows box overlays color-coded by TP/FP/FN
- [ ] Full end-to-end: create project → import → train → evaluate visible in UI

---

## Phase 21 — Semantic Segmentation: Core

**Goal**: Add the semantic segmentation task backend — models, mask data handling, training, and evaluation logic.

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `SegmentationTask` registered in task registry |
| Models | FCN, DeepLabV3, DeepLabV3+, LRASPP (from torchvision) |
| Dataset adapter | Pixel mask loading, class mapping, COCO polygon/RLE rasterization |
| Augmentations | Joint image+mask transforms — same spatial ops applied identically to both |
| Loss | CrossEntropyLoss + DiceLoss |
| Metrics | mIoU, pixel accuracy, per-class IoU |
| Evaluation pipeline | Per-image pixel accuracy + per-class IoU. Aggregate mIoU. |

### Acceptance Criteria

- [ ] Segmentation project can be created and trained end-to-end (CLI/API)
- [ ] COCO polygon/RLE dataset can be imported
- [ ] mIoU metric is computed correctly

---

## Phase 22 — Semantic Segmentation: GUI

**Goal**: Add UI adaptations for the semantic segmentation task across all relevant pages.

### Deliverables

| Component | Change |
|-----------|--------|
| Dataset page | Semi-transparent colored mask overlay (one color per class) |
| Evaluation page | Predicted mask overlay, ground-truth overlay, error map (misclassified pixels), per-class IoU bar chart |

### Acceptance Criteria

- [ ] Mask overlays render on images in the dataset page
- [ ] Evaluation page shows predicted mask, GT mask, and per-pixel error map
- [ ] Per-class IoU bar chart is displayed
- [ ] Full end-to-end: create project → import → train → evaluate visible in UI

---

## Phase 23 — Instance Segmentation

**Goal**: Add instance segmentation, extending segmentation + detection (builds on phases 19–22).

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `InstanceSegmentationTask` registered in task registry |
| Models | Mask R-CNN (from torchvision, extends Faster R-CNN with mask head) |
| Dataset adapter | COCO instance format (polygons + bbox + category_id) |
| Collate function | Reuse OD collate from Phase 19 |
| Metrics | mAP (mask IoU), mAP (box IoU) via torchmetrics `iou_type="segm"` |
| Evaluation pipeline | Per-instance predictions with class, bbox, mask, confidence |
| Dataset page | Per-instance colored mask overlays (unique color per instance, class-based hue) |
| Evaluation page | Predicted instance masks (solid) vs ground-truth (dashed outlines), confidence per instance |

### Acceptance Criteria

- [ ] Instance segmentation project can be created
- [ ] COCO instance-format dataset can be imported
- [ ] Per-instance masks render correctly in dataset and evaluation pages
- [ ] mAP (mask IoU) is computed and displayed

---

## Phase 24 — DINOv3 Models

**Goal**: Integrate the DINOv3 model family (ConvNeXt and ViT variants) as additional backbone options for classification, detection, and segmentation tasks, sourced from HuggingFace Transformers.

### Deliverables

| Component | Scope |
|-----------|-------|
| **HuggingFace integration** | Add `transformers` dependency; implement a HuggingFace backbone wrapper that exposes the same `BackboneWrapper` interface as torchvision backbones |
| **DINOv3-ViT backbones** | `dinov3_vit_small`, `dinov3_vit_base`, `dinov3_vit_large` — Vision Transformer backbones pretrained with DINOv3 self-supervised method |
| **DINOv3-ConvNeXt backbones** | `dinov3_convnext_small`, `dinov3_convnext_base`, `dinov3_convnext_large` — ConvNeXt backbones pretrained with DINOv3 |
| **Classification models** | Register DINOv3 backbones + FC classification head in the architecture catalog |
| **Detection models** | Register DINOv3 backbones as feature extractors for detection heads (Faster R-CNN, RetinaNet, FCOS with DINOv3 backbone + FPN) |
| **Segmentation models** | Register DINOv3 backbones for semantic segmentation (DINOv3 backbone + segmentation decoder) |
| **Weight management** | Download and cache HuggingFace pretrained weights; integrate with existing pretrained weight system |
| **Preprocessing adapter** | Handle DINOv3-specific image preprocessing (different normalization stats and input resolutions) |
| **ONNX export support** | Ensure DINOv3-based models export cleanly to ONNX |

### Architecture Additions

| Architecture | Key | Source | Tasks |
|-------------|-----|--------|-------|
| DINOv3 ViT-S/14 | `dinov3_vit_small` | `transformers` | Classification, Detection, Segmentation |
| DINOv3 ViT-B/14 | `dinov3_vit_base` | `transformers` | Classification, Detection, Segmentation |
| DINOv3 ViT-L/14 | `dinov3_vit_large` | `transformers` | Classification, Detection, Segmentation |
| DINOv3 ConvNeXt-S | `dinov3_convnext_small` | `transformers` | Classification, Detection, Segmentation |
| DINOv3 ConvNeXt-B | `dinov3_convnext_base` | `transformers` | Classification, Detection, Segmentation |
| DINOv3 ConvNeXt-L | `dinov3_convnext_large` | `transformers` | Classification, Detection, Segmentation |

### Integration Notes

- HuggingFace models return dict/dataclass outputs — the `BackboneWrapper` must extract the relevant feature tensors and reshape them to match the `(N, C, H, W)` convention expected by existing heads.
- ViT backbones produce patch tokens — for classification the `[CLS]` token is used; for dense tasks (detection, segmentation) patch tokens are reshaped into a spatial feature map.
- ConvNeXt backbones produce standard hierarchical feature maps compatible with FPN.
- DINOv3 models use different normalization constants than ImageNet-pretrained torchvision models — the augmentation pipeline must adapt automatically based on the selected backbone source.
- Frozen backbone mode is especially effective with DINOv3 due to strong self-supervised pretraining.

### Acceptance Criteria

- [ ] All 6 DINOv3 backbone variants are available in the architecture catalog
- [ ] A classification experiment can be trained with a DINOv3-ViT backbone
- [ ] A classification experiment can be trained with a DINOv3-ConvNeXt backbone
- [ ] DINOv3-based models can be exported to ONNX and pass validation
- [ ] HuggingFace weights are downloaded and cached correctly
- [ ] DINOv3 backbones can be used for detection and segmentation tasks (when those tasks are active)

---

## Phase 25 — Anomaly Detection

**Goal**: Add the anomaly detection task.

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `AnomalyDetectionTask` (binary classification variant) |
| Loss function | `BCEWithLogitsLoss` |
| Metrics | AUROC, F1 (binary threshold) |
| Augmentations | Lighter defaults (less geometric augmentation) |
| Evaluation | Anomaly score display, ROC curve, optimal threshold |

### GUI Adaptations

- Enable "Anomaly Detection" in project creation modal.
- Evaluation page: add ROC curve visualization.
- Per-image results: show anomaly score instead of class probabilities.

### Acceptance Criteria

- [ ] Anomaly detection project can be created
- [ ] Training completes with AUROC metric tracked
- [ ] ROC curve is displayed on the evaluation page

---

## Phase 26 — Regression

**Goal**: Add the regression task.

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `RegressionTask` |
| Head | Configurable `num_outputs` neurons (no activation) |
| Loss | MSE, L1, Huber |
| Metrics | MAE, MSE, RMSE, R² |
| Target normalization | Z-score normalization with stored mean/std |
| Import | CSV format with numeric target values |
| Evaluation | Scatter plot (predicted vs actual), error histogram |

### Acceptance Criteria

- [ ] Regression project can be created
- [ ] CSV dataset with numeric targets can be imported
- [ ] Scatter plot and error histogram display on evaluation page

---

## Phase 27 — Oriented Object Detection

**Goal**: Add oriented object detection, extending the detection foundation.

### Deliverables

| Component | Change |
|-----------|--------|
| Task definition | `OrientedObjectDetectionTask` |
| Model adaptation | Extend FCOS/RetinaNet with angle regression head |
| Dataset format | YOLO-OBB / DOTA import support |
| Rotated IoU | Shapely-based rotated bounding box IoU |

### GUI Adaptations

- Rotated rectangle drawing on canvas overlay.

### Acceptance Criteria

- [ ] Oriented OD project can be created
- [ ] DOTA-format dataset can be imported
- [ ] Rotated bounding boxes render correctly

---

## Phase 28 — Polish & Extras

**Goal**: Improve usability, add deferred features.

### Planned Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Multi-GPU (DDP)** | Enable multi-GPU training via DDP | High |
| **Mixed precision** | Enable fp16/bf16 training options | High |
| **TorchScript export** | Add `torch.jit.trace` export | Medium |
| **TensorRT export** | ONNX → TensorRT engine conversion | Medium |
| **OpenVINO export** | ONNX → OpenVINO IR conversion | Medium |
| **Experiment comparison** | Overlay loss/metric curves from multiple experiments | Medium |
| **REST API docs** | Swagger UI with examples | Medium |
| **Annotation editor** | Basic annotation tools (bbox, label) in the browser | Low |
| **Dataset versioning** | Track dataset changes over time | Low |
| **Hyperparameter search** | Grid/random search over hyperparameters | Low |
| **Model pruning/quantization** | Post-training optimization | Low |
| **Dark mode** | Toggle dark/light theme | Low |

---

## Dependency Graph

```
GROUP A — Data Foundation
  Phase 1  (Core Platform)
    └── Phase 2  (Datasets)
          └── Phase 3  (Splits)
                └── Phase 4  (API: Projects, Datasets & Splits)
                      ├── Phase 5  (GUI: Project Page)
                      ├── Phase 6  (GUI: Dataset Page)
                      └── Phase 7  (GUI: Split Page)

GROUP B — Training & Experiments
  Phase 8  (Classification Task)  ── depends on Phase 3
    └── Phase 9  (Training Pipeline)
          └── Phase 10 (Experiment Tracking)
                └── Phase 11 (API: Training & Experiments)
                      └── Phase 12 (GUI: Training Page)

GROUP C — Evaluation
  Phase 13 (Evaluation Pipeline)  ── depends on Phase 10
    └── Phase 14 (API: Evaluation)
          └── Phase 15 (GUI: Evaluation Page)

GROUP D — Export
  Phase 16 (ONNX Export)  ── depends on Phase 10
    └── Phase 17 (API: Export)
          └── Phase 18 (GUI: Export Page)

GROUP E — Additional Tasks  ── depend on Groups A–D
  ├── Phase 19 (OD Core) ── introduces bbox handling
  │     └── Phase 20 (OD GUI) ── UI overlays for OD
  ├── Phase 21 (Segmentation Core) ── introduces mask handling
  │     └── Phase 22 (Segmentation GUI) ── UI overlays for segmentation
  ├── Phase 23 (Instance Seg) ── extends Phases 19–22; core + GUI combined
  ├── Phase 24 (DINOv3 Models) ── extends backbone catalog; depends on Phase 8+
  ├── Phase 25 (Anomaly Detection) ── independent
  ├── Phase 26 (Regression) ── independent
  ├── Phase 27 (Oriented OD) ── extends Phase 19
  └── Phase 28 (Polish) ── can run in parallel with Phases 19–27
```

---

## Development Principles

1. **Each phase is self-contained** — complete it fully before moving on.
2. **Backend → API → GUI** — implement logic, expose it, then build the interface. Test end-to-end after each GUI phase.
3. **Task registry is the extension point** — each new task plugs in via the registry.
4. **Test each phase independently** — each phase should be release-ready.
5. **Don't over-engineer early** — build what's needed now; refactor when patterns emerge.
6. **Keep the GUI simple** — HTMX + Alpine.js. Avoid adding a JS framework.

---

## Related Documents

- Overview → [00-overview.md](00-overview.md)
- Task registry → [03-tasks/00-task-registry.md](03-tasks/00-task-registry.md)
- All task definitions → [03-tasks/](03-tasks/)
