from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from app.schemas.dataset import DatasetImage, SplitValue
from app.schemas.split import SplitRatios

_DEFAULT_SEED = 0
_NORMAL_LABEL = "normal"
_ANOMALOUS_LABEL = "anomalous"


@dataclass(frozen=True, slots=True)
class SplitComputation:
    """Computed split assignments and aggregate statistics."""

    assignments: list[SplitValue]
    subset_counts: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    warnings: list[str]


def compute_split_assignments(
    *,
    images: Sequence[DatasetImage],
    task: str,
    ratios: SplitRatios,
    seed: int | None = None,
    class_order: Sequence[str] | None = None,
) -> SplitComputation:
    """Compute train/val/test assignments for dataset images.

    Args:
        images: Dataset image entries from `dataset.json`.
        task: Project task type.
        ratios: Target split ratios.
        seed: Optional random seed used for reproducibility.
        class_order: Optional preferred class order for distribution output.

    Returns:
        SplitComputation containing per-image assignments and statistics.
    """
    if not images:
        raise ValueError("Cannot generate a split for an empty dataset.")

    resolved_seed = _DEFAULT_SEED if seed is None else seed
    labels = [_extract_image_label(image, task=task) for image in images]
    assignments: list[SplitValue] = ["none"] * len(images)
    warnings: list[str] = []

    if len(images) < 3:
        warnings.append("Dataset has fewer than 3 images; one or more subsets may be empty.")

    if task == "anomaly_detection":
        warnings.extend(
            _assign_anomaly_detection_subsets(
                labels=labels,
                assignments=assignments,
                ratios=ratios,
                seed=resolved_seed,
            )
        )
    else:
        warnings.extend(
            _assign_standard_subsets(
                labels=labels,
                assignments=assignments,
                ratios=ratios,
                seed=resolved_seed,
            )
        )

    subset_counts = _count_subsets(assignments)
    class_distribution = _build_class_distribution(
        labels=labels,
        assignments=assignments,
        class_order=class_order,
    )
    return SplitComputation(
        assignments=assignments,
        subset_counts=subset_counts,
        class_distribution=class_distribution,
        warnings=warnings,
    )


def _assign_standard_subsets(
    *,
    labels: Sequence[str],
    assignments: list[SplitValue],
    ratios: SplitRatios,
    seed: int,
) -> list[str]:
    warnings: list[str] = []
    label_counts = Counter(labels)
    singleton_indices = [index for index, label in enumerate(labels) if label_counts[label] == 1]
    if singleton_indices:
        warnings.append("Classes with a single image were assigned to the train subset.")
        for index in singleton_indices:
            assignments[index] = "train"

    remaining_indices = [index for index, subset in enumerate(assignments) if subset == "none"]
    if not remaining_indices:
        return warnings

    remaining_labels = [labels[index] for index in remaining_indices]
    if len(set(remaining_labels)) == 1:
        warnings.append(
            "All images belong to one class; split used random shuffling without stratification."
        )

    train_indices, val_indices, test_indices, split_warnings = _split_three_way(
        indices=remaining_indices,
        labels=remaining_labels,
        train_ratio=ratios.train,
        val_ratio=ratios.val,
        test_ratio=ratios.test,
        seed=seed,
        context="dataset",
    )
    warnings.extend(split_warnings)
    _apply_subset(assignments=assignments, indices=train_indices, subset="train")
    _apply_subset(assignments=assignments, indices=val_indices, subset="val")
    _apply_subset(assignments=assignments, indices=test_indices, subset="test")
    return warnings


def _assign_anomaly_detection_subsets(
    *,
    labels: Sequence[str],
    assignments: list[SplitValue],
    ratios: SplitRatios,
    seed: int,
) -> list[str]:
    warnings: list[str] = []
    normal_indices = [index for index, label in enumerate(labels) if label == _NORMAL_LABEL]
    anomalous_indices = [index for index, label in enumerate(labels) if label == _ANOMALOUS_LABEL]

    if not anomalous_indices:
        warnings.append(
            "Dataset has no anomalous images; evaluation subsets will not include anomaly samples."
        )
    if not normal_indices:
        warnings.append("Dataset has no normal images; train subset will be empty.")

    normal_labels = [labels[index] for index in normal_indices]
    train_indices, normal_val_indices, normal_test_indices, normal_warnings = _split_three_way(
        indices=normal_indices,
        labels=normal_labels,
        train_ratio=ratios.train,
        val_ratio=ratios.val,
        test_ratio=ratios.test,
        seed=seed,
        context="normal pool",
    )
    warnings.extend(normal_warnings)
    _apply_subset(assignments=assignments, indices=train_indices, subset="train")
    _apply_subset(assignments=assignments, indices=normal_val_indices, subset="val")
    _apply_subset(assignments=assignments, indices=normal_test_indices, subset="test")

    if not anomalous_indices:
        return warnings

    holdout_ratio = ratios.val + ratios.test
    if holdout_ratio <= 0.0:
        raise ValueError(
            "Anomaly splits require a non-zero val or test ratio so anomalous samples avoid train."
        )

    anomaly_val_ratio = ratios.val / holdout_ratio
    anomaly_test_ratio = ratios.test / holdout_ratio
    _, anomalous_val_indices, anomalous_test_indices, anomaly_warnings = _split_three_way(
        indices=anomalous_indices,
        labels=[labels[index] for index in anomalous_indices],
        train_ratio=0.0,
        val_ratio=anomaly_val_ratio,
        test_ratio=anomaly_test_ratio,
        seed=seed + 1,
        context="anomalous pool",
    )
    warnings.extend(anomaly_warnings)
    _apply_subset(assignments=assignments, indices=anomalous_val_indices, subset="val")
    _apply_subset(assignments=assignments, indices=anomalous_test_indices, subset="test")
    return warnings


def _split_three_way(
    *,
    indices: Sequence[int],
    labels: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    context: str,
) -> tuple[list[int], list[int], list[int], list[str]]:
    warnings: list[str] = []
    if not indices:
        return [], [], [], warnings

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0.0:
        return list(indices), [], [], warnings

    if train_ratio <= 0.0:
        val_indices, test_indices, split_warnings = _split_two_way(
            indices=indices,
            labels=labels,
            test_ratio=test_ratio / holdout_ratio,
            seed=seed,
            context=context,
        )
        warnings.extend(split_warnings)
        return [], val_indices, test_indices, warnings

    train_indices, holdout_indices, first_warnings = _split_two_way(
        indices=indices,
        labels=labels,
        test_ratio=holdout_ratio,
        seed=seed,
        context=context,
    )
    warnings.extend(first_warnings)
    if not holdout_indices:
        return train_indices, [], [], warnings

    if val_ratio <= 0.0:
        return train_indices, [], holdout_indices, warnings
    if test_ratio <= 0.0:
        return train_indices, holdout_indices, [], warnings

    holdout_labels = _slice_labels(labels=labels, indices=indices, subset_indices=holdout_indices)
    val_indices, test_indices, second_warnings = _split_two_way(
        indices=holdout_indices,
        labels=holdout_labels,
        test_ratio=test_ratio / holdout_ratio,
        seed=seed + 1,
        context=f"{context} holdout",
    )
    warnings.extend(second_warnings)
    return train_indices, val_indices, test_indices, warnings


def _split_two_way(
    *,
    indices: Sequence[int],
    labels: Sequence[str],
    test_ratio: float,
    seed: int,
    context: str,
) -> tuple[list[int], list[int], list[str]]:
    warnings: list[str] = []
    if not indices:
        return [], [], warnings
    if test_ratio <= 0.0:
        return list(indices), [], warnings
    if test_ratio >= 1.0:
        return [], list(indices), warnings
    if len(indices) == 1:
        if test_ratio >= 0.5:
            return [], list(indices), warnings
        return list(indices), [], warnings

    stratify_labels = _build_stratify_labels(labels)
    try:
        train_indices, test_indices = train_test_split(
            list(indices),
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        warnings.append(
            f"{context.capitalize()} split could not preserve stratification; used random shuffle."
        )
        random_train, random_test = _random_split(indices=indices, test_ratio=test_ratio, seed=seed)
        return random_train, random_test, warnings

    return list(train_indices), list(test_indices), warnings


def _random_split(
    *,
    indices: Sequence[int],
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    test_count = _compute_test_count(size=len(shuffled), test_ratio=test_ratio)
    test_indices = shuffled[:test_count]
    train_indices = shuffled[test_count:]
    return train_indices, test_indices


def _compute_test_count(*, size: int, test_ratio: float) -> int:
    if size <= 1:
        return 0 if test_ratio < 1.0 else size
    if test_ratio <= 0.0:
        return 0
    if test_ratio >= 1.0:
        return size

    candidate = round(size * test_ratio)
    if candidate <= 0:
        return 1
    if candidate >= size:
        return size - 1
    return int(candidate)


def _build_stratify_labels(labels: Sequence[str]) -> list[str] | None:
    if len(set(labels)) <= 1:
        return None
    counts = Counter(labels)
    if min(counts.values()) < 2:
        return None
    return list(labels)


def _slice_labels(
    *,
    labels: Sequence[str],
    indices: Sequence[int],
    subset_indices: Sequence[int],
) -> list[str]:
    label_by_index = {index: labels[position] for position, index in enumerate(indices)}
    return [label_by_index[index] for index in subset_indices]


def _apply_subset(
    *,
    assignments: list[SplitValue],
    indices: Sequence[int],
    subset: SplitValue,
) -> None:
    for index in indices:
        assignments[index] = subset


def _extract_image_label(image: DatasetImage, *, task: str) -> str:
    if task == "anomaly_detection":
        for annotation in image.annotations:
            if annotation.type == "anomaly":
                return _ANOMALOUS_LABEL if annotation.is_anomalous else _NORMAL_LABEL
        raise ValueError("Anomaly dataset images must include an anomaly annotation.")

    for annotation in image.annotations:
        if annotation.type == "label":
            return annotation.class_name
    raise ValueError("Classification dataset images must include a label annotation.")


def _count_subsets(assignments: Sequence[SplitValue]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "none": 0}
    for subset in assignments:
        counts[subset] += 1
    return counts


def _build_class_distribution(
    *,
    labels: Sequence[str],
    assignments: Sequence[SplitValue],
    class_order: Sequence[str] | None,
) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    ordered_classes = list(class_order or [])
    for class_name in ordered_classes:
        distribution[class_name] = {"train": 0, "val": 0, "test": 0, "none": 0}

    for class_name in sorted(set(labels)):
        distribution.setdefault(class_name, {"train": 0, "val": 0, "test": 0, "none": 0})

    for index, subset in enumerate(assignments):
        class_name = labels[index]
        distribution[class_name][subset] += 1
    return distribution
