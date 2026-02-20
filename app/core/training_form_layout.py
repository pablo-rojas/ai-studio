from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingSection:
    """Declarative metadata for one collapsible training config section."""

    id: str
    title: str
    description: str
    default_open: bool


_CLASSIFICATION_SECTIONS: tuple[TrainingSection, ...] = (
    TrainingSection(
        id="setup",
        title="Experiment Setup",
        description="Name and dataset split selection.",
        default_open=True,
    ),
    TrainingSection(
        id="architecture",
        title="Architecture",
        description="Model backbone and head behavior.",
        default_open=True,
    ),
    TrainingSection(
        id="optimization",
        title="Optimization",
        description="Optimizer and learning-rate schedule.",
        default_open=True,
    ),
    TrainingSection(
        id="objective",
        title="Objective (Loss)",
        description="Loss function and loss-specific parameters.",
        default_open=False,
    ),
    TrainingSection(
        id="training_settings",
        title="Training Settings",
        description="Epoch, batch, and stopping controls.",
        default_open=True,
    ),
    TrainingSection(
        id="hardware",
        title="Hardware",
        description="Device selection and precision settings.",
        default_open=False,
    ),
    TrainingSection(
        id="augmentation",
        title="Data Augmentation",
        description="Enable or disable training transforms.",
        default_open=False,
    ),
)


def get_training_sections(task: str) -> tuple[TrainingSection, ...]:
    """Return the training form sections for a task.

    For now all implemented tasks reuse the classification section layout.
    """
    del task
    return _CLASSIFICATION_SECTIONS
