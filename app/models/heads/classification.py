from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Task head for image classification logits."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be at least 1.")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Return class logits for 2D or 4D backbone features."""
        if features.ndim == 4:
            x = self.pool(features)
            x = self.flatten(x)
        elif features.ndim == 2:
            x = features
        else:
            raise ValueError("ClassificationHead expects a 2D or 4D tensor.")
        x = self.dropout(x)
        return self.fc(x)
