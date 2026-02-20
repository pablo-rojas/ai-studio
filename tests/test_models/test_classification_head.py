from __future__ import annotations

import pytest
import torch

from app.models.heads.classification import ClassificationHead


def test_classification_head_supports_spatial_feature_maps() -> None:
    head = ClassificationHead(in_features=16, num_classes=4, dropout=0.0)
    features = torch.randn(3, 16, 8, 8)

    logits = head(features)

    assert logits.shape == (3, 4)


def test_classification_head_supports_flat_feature_vectors() -> None:
    head = ClassificationHead(in_features=32, num_classes=5, dropout=0.0)
    features = torch.randn(2, 32)

    logits = head(features)

    assert logits.shape == (2, 5)


def test_classification_head_rejects_invalid_input_rank() -> None:
    head = ClassificationHead(in_features=32, num_classes=5)
    features = torch.randn(2, 32, 4)

    with pytest.raises(ValueError, match="2D or 4D"):
        head(features)
