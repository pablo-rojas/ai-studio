# Tasks — Anomaly Detection

**Phase**: 23

---

## 1. Task Description

Determine whether an image is **normal** or **anomalous** and localize the anomaly by producing a pixel-level anomaly heatmap.

The approach is based on **Uninformed Students** (Bergmann et al., CVPR 2020) — a student–teacher framework for unsupervised anomaly detection and pixel-precise anomaly segmentation. Only **normal (anomaly-free) images** are required for training; no anomaly labels are needed during training.

Two output modes:
- **Image-level**: anomaly score (scalar) derived from the anomaly heatmap (e.g., max value). A threshold converts this to a binary normal/anomalous decision.
- **Pixel-level**: anomaly heatmap — a per-pixel score indicating how anomalous each region is.

---

## 2. Method Overview — Uninformed Students

> **Paper**: *Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings* — Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. CVPR 2020. [arXiv:1911.02357](https://arxiv.org/abs/1911.02357)

### 2.1 Core Idea

A **pretrained teacher network** produces dense feature descriptors for image patches. An **ensemble of student networks** is trained to regress the teacher's output using only anomaly-free images. At inference time, anomalies are detected wherever the students **fail to reproduce** the teacher's descriptors — because the students have never seen anomalous patterns during training.

### 2.2 Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1 — Backbone (pretrained)                              │
│  ResNet-18 pretrained on ImageNet. Used as a frozen feature  │
│  extractor. Optionally fine-tuned on domain images.          │
└───────────────────────────┬──────────────────────────────────┘
                            │ knowledge distillation
┌───────────────────────────▼──────────────────────────────────┐
│  STEP 2 — Teacher Network                                    │
│  Smaller CNN that distils the backbone's knowledge.          │
│  Trained with distillation loss + compactness loss.          │
│  Produces a d-dimensional descriptor per image patch.        │
└───────────────────────────┬──────────────────────────────────┘
                            │ teacher descriptors (normalized)
┌───────────────────────────▼──────────────────────────────────┐
│  STEP 3 — Student Ensemble (M students, same arch as teacher)│
│  Each student is trained on anomaly-free images only to      │
│  regress the teacher's normalized output (MSE loss).         │
└───────────────────────────┬──────────────────────────────────┘
                            │ compare teacher vs students
┌───────────────────────────▼──────────────────────────────────┐
│  STEP 4 — Anomaly Scoring                                    │
│  score(x,y) = prediction_error(x,y) + predictive_uncertainty │
│  prediction_error  = ‖μ_S(x,y) − T(x,y)‖²                  │
│  predictive_uncert = Var across student outputs at (x,y)     │
│  → pixel-level anomaly heatmap → image-level score (max)     │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Multi-Scale Detection

Different **patch sizes** capture anomalies of different spatial scales:

| Patch Size | Target Anomaly Size |
|------------|---------------------|
| 17 × 17 | Small defects |
| 33 × 33 | Medium defects |
| 65 × 65 | Large defects |

The final anomaly heatmap can combine scores from multiple patch sizes for robust multi-scale detection.

---

## 3. Label Schema

```json
{
  "type": "anomaly",
  "is_anomalous": true
}
```

- `is_anomalous`: `true` for defective/anomalous images, `false` for normal.
- Normal images have `"is_anomalous": false`.
- **Training uses only normal images** — anomalous images are excluded from the training split entirely and routed to val/test only.
- **Pixel-level masks** (`mask_path`) are optional and used only for evaluation of pixel-level metrics.

---

## 4. Compatible Dataset Formats

| Format | Notes |
|--------|-------|
| **Folder-based** | `images/good/` for normal images + `images/defect/` for test anomalies + optional `masks/` |
| **CSV** | `filename, is_anomalous` (0 or 1) |
| **MVTec-style** | Folder structure: `train/good/`, `test/good/`, `test/defect_type_N/` + optional `ground_truth/` masks |

---

## 5. Architecture

| Component | Architecture | Details |
|-----------|-------------|---------|
| **Backbone** | ResNet-18 (pretrained on ImageNet) | Frozen feature extractor. Produces 512-d descriptor vectors. Optionally fine-tuned on domain data. |
| **Teacher** | Small CNN (patch-based) | Distilled from the backbone. Same receptive field as the chosen patch size. Outputs d-dimensional descriptors (d = 512). |
| **Student × M** | Same architecture as Teacher | Ensemble of M identical (but independently initialized) networks. Trained to regress teacher's normalized output on anomaly-free data. |

Default ensemble size: **M = 3** students.

All three components (backbone, teacher, students) use **convolutional** architectures that produce **dense** (per-patch) descriptors via Fast Dense Feature Extraction (FDFE), enabling efficient pixel-level anomaly maps without sliding-window inference.

---

## 6. Loss Functions

### Teacher Training (knowledge distillation from backbone)

| Loss | Formula | Purpose |
|------|---------|---------|
| **Distillation loss** | $\mathcal{L}_D = \frac{1}{N}\sum_i \|\| T(x_i) - B(x_i) \|\|^2$ | MSE between teacher output and backbone output |
| **Compactness loss** | $\mathcal{L}_C = \sum_{i<j} \|Corr(T_i, T_j)\|$ | Decorrelates descriptor dimensions to maximize information |
| **Total** | $\mathcal{L}_{teacher} = \mathcal{L}_D + \mathcal{L}_C$ | |

### Student Training (regression on teacher descriptors)

| Loss | Formula | Purpose |
|------|---------|---------|
| **Regression loss (MSE)** | $\mathcal{L}_S = \frac{1}{N}\sum_i \|\| S_k(x_i) - \hat{T}(x_i) \|\|^2$ | MSE between each student's output and the teacher's normalized output |

Where $\hat{T}(x_i)$ denotes the teacher output normalized to zero mean and unit variance (computed over the training set).

---

## 7. Anomaly Scoring

At inference, for each spatial position (x, y):

| Component | Formula | Meaning |
|-----------|---------|---------|
| **Prediction error** | $e(x,y) = \|\| \mu_S(x,y) - T(x,y) \|\|^2$ | Squared distance between mean student prediction and teacher output |
| **Predictive uncertainty** | $v(x,y) = \frac{1}{M}\sum_{k=1}^{M} \|\| S_k(x,y) - \mu_S(x,y) \|\|^2$ | Variance among the M student predictions |
| **Anomaly score (pixel)** | $a(x,y) = e(x,y) + v(x,y)$ | Combined pixel-level anomaly score |
| **Anomaly score (image)** | $A = \max_{x,y} a(x,y)$ | Image-level score: maximum over the anomaly heatmap |

A threshold on $A$ yields the binary normal/anomalous classification.

---

## 8. Metrics

| Metric | Role | Notes |
|--------|------|-------|
| **Image-AUROC** | Primary | Area under ROC curve on image-level anomaly scores |
| **Pixel-AUROC** | Primary (pixel-level) | Area under ROC curve on pixel-level scores — requires mask annotations |
| **F1** | Secondary | At optimal threshold |
| **Precision** | Secondary | At optimal threshold |
| **Recall** | Secondary | At optimal threshold |

---

## 9. Default Augmentations

### Teacher Training (patch-level)

```json
[
  { "name": "Resize", "params": { "size": [256, 256] } },
  { "name": "RandomCrop", "params": { "size": "patch_size" } },
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomVerticalFlip", "params": { "p": 0.5 } },
  { "name": "RandomRotation", "params": { "degrees": 180 } },
  { "name": "ToImage", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

### Student Training (image-level)

```json
[
  { "name": "Resize", "params": { "size": [256, 256] } },
  { "name": "RandomHorizontalFlip", "params": { "p": 0.5 } },
  { "name": "RandomVerticalFlip", "params": { "p": 0.5 } },
  { "name": "ToImage", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

Light augmentation is used to preserve the structure of normal patterns. No color jitter or aggressive transforms.

### Validation / Test

```json
[
  { "name": "Resize", "params": { "size": [256, 256] } },
  { "name": "ToImage", "params": {} },
  { "name": "Normalize", "params": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } }
]
```

---

## 10. Default Hyperparameters

### Teacher Distillation

```json
{
  "optimizer": "adam",
  "learning_rate": 0.0002,
  "weight_decay": 0.00001,
  "batch_size": 64,
  "max_epochs": 200,
  "descriptor_dim": 512,
  "backbone": "resnet18",
  "pretrained": true
}
```

### Student Training

```json
{
  "optimizer": "adam",
  "learning_rate": 0.0001,
  "weight_decay": 0.00001,
  "batch_size": 1,
  "max_epochs": 25,
  "n_students": 3,
  "patch_size": 33,
  "image_size": 256
}
```

---

## 11. Visualization

### Dataset Page
- **Normal images**: green "Normal" badge.
- **Anomalous images**: red "Anomalous" badge + mask overlay (semi-transparent red) if mask exists.

### Evaluation Page
- **Anomaly heatmap overlay**: semi-transparent heatmap (blue → red gradient) showing per-pixel anomaly score on top of the original image.
- **Anomaly score** displayed as a colored badge (gradient from green → red based on image-level score).
- **Correct/incorrect** border coloring.
- **Uncertainty map** (optional): visualization of the predictive uncertainty $v(x,y)$ across the student ensemble.

---

## 12. Evaluation Specifics

Per-image result:

```json
{
  "filename": "img_001.png",
  "ground_truth": { "is_anomalous": true },
  "prediction": {
    "is_anomalous": true,
    "anomaly_score": 0.87,
    "heatmap_path": "heatmaps/img_001.png"
  },
  "correct": true
}
```

---

## 13. Split Constraints (Semi-Supervised)

Anomaly detection is **semi-supervised**: only normal (good) images are used for training, so the split algorithm must enforce this constraint.

### Automatic Split Behaviour

When the user provides train/val/test percentages (e.g., **70-15-15**), the split engine operates in two stages:

1. **Normal images** — the specified train percentage is applied to **normal images only**. The remaining normal images are distributed between val and test according to the original val/test ratio.
2. **Anomalous images** — all anomalous images are excluded from train. They are distributed between **val and test only**, proportionally to the val/test ratio derived from the user's percentages.

**Ratio derivation**: given user ratios `train%-val%-test%`, the anomalous split ratio is `val% / (val% + test%)` for val and `test% / (val% + test%)` for test.

| User Percentages | Normal Images | Anomalous Images |
|------------------|---------------|------------------|
| 70-15-15 | 70% train, 15% val, 15% test | 50% val, 50% test |
| 70-20-10 | 70% train, 20% val, 10% test | 67% val, 33% test |
| 80-10-10 | 80% train, 10% val, 10% test | 50% val, 50% test |
| 60-30-10 | 60% train, 30% val, 10% test | 75% val, 25% test |

See also: [../02-data-layer/03-splits.md](../02-data-layer/03-splits.md) — Section 3.1.

---

## 14. Training Workflow Summary

1. **Prepare dataset** — only normal images go into the training split; anomalous images are routed to val/test (see Section 13).
2. **Load pretrained backbone** — ResNet-18 from torchvision (ImageNet weights), frozen.
3. **Train teacher** — distil backbone knowledge into a smaller patch-CNN using distillation + compactness loss.
4. **Normalize teacher outputs** — compute mean and variance of teacher descriptors over the training set.
5. **Train student ensemble** — each of M students independently regresses the normalized teacher output on anomaly-free images.
6. **Inference** — for each test image, compute teacher descriptors and student descriptors, derive the anomaly heatmap, and threshold the image-level score.

---

## 15. Related Documents

- Task registry → [00-task-registry.md](00-task-registry.md)
- Architecture catalog → [../04-models/00-architecture-catalog.md](../04-models/00-architecture-catalog.md)
- Split logic (anomaly-aware) → [../02-data-layer/03-splits.md](../02-data-layer/03-splits.md)
- Original paper → [arXiv:1911.02357](https://arxiv.org/abs/1911.02357)
