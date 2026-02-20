from __future__ import annotations

import pytest
import torch

from app.evaluation.metrics import build_classification_metrics, compute_classification_metrics


def test_build_classification_metrics_validates_num_classes() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_classification_metrics(0)


def test_compute_classification_metrics_for_perfect_predictions() -> None:
    logits = torch.tensor(
        [
            [9.0, 0.5, 0.1],
            [0.2, 8.2, 0.1],
            [0.1, 0.3, 7.7],
            [7.5, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

    metrics = compute_classification_metrics(logits, targets, num_classes=3)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
