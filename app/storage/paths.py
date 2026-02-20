from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_identifier(identifier: str, *, field_name: str) -> str:
    """Validate identifiers that are used as directory names."""
    if not _ID_PATTERN.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, numbers, underscores, and hyphens."
        )
    return identifier


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Centralized path resolution for all workspace entities."""

    root: Path

    @classmethod
    def from_settings(cls) -> WorkspacePaths:
        """Create a path resolver from current runtime settings."""
        return cls(root=get_settings().workspace_root)

    def ensure_workspace_layout(self) -> None:
        """Create the top-level workspace directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects_root().mkdir(parents=True, exist_ok=True)

    def workspace_metadata_file(self) -> Path:
        """Return the path to `workspace.json`."""
        return self.root / "workspace.json"

    def workspace_metadata_exists(self) -> bool:
        """Check whether `workspace.json` already exists."""
        return self.workspace_metadata_file().exists()

    def projects_root(self) -> Path:
        """Return the root directory for all projects."""
        return self.root / "projects"

    def project_dir(self, project_id: str) -> Path:
        """Resolve a project directory path."""
        safe_project_id = _validate_identifier(project_id, field_name="project_id")
        return self.projects_root() / safe_project_id

    def project_metadata_file(self, project_id: str) -> Path:
        """Return the path to a project's `project.json`."""
        return self.project_dir(project_id) / "project.json"

    def dataset_dir(self, project_id: str) -> Path:
        """Return the path to a project's dataset folder."""
        return self.project_dir(project_id) / "dataset"

    def dataset_metadata_file(self, project_id: str) -> Path:
        """Return the path to a project's `dataset.json`."""
        return self.dataset_dir(project_id) / "dataset.json"

    def dataset_images_dir(self, project_id: str) -> Path:
        """Return the path to a project's dataset image folder."""
        return self.dataset_dir(project_id) / "images"

    def dataset_masks_dir(self, project_id: str) -> Path:
        """Return the path to a project's dataset mask folder."""
        return self.dataset_dir(project_id) / "masks"

    def dataset_thumbnails_dir(self, project_id: str) -> Path:
        """Return the path to cached dataset thumbnails."""
        return self.dataset_dir(project_id) / ".thumbs"

    def experiments_dir(self, project_id: str) -> Path:
        """Return the path to a project's experiments folder."""
        return self.project_dir(project_id) / "experiments"

    def experiments_index_file(self, project_id: str) -> Path:
        """Return the path to `experiments_index.json`."""
        return self.experiments_dir(project_id) / "experiments_index.json"

    def experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        """Return the path to an experiment folder."""
        safe_experiment_id = _validate_identifier(experiment_id, field_name="experiment_id")
        return self.experiments_dir(project_id) / safe_experiment_id

    def experiment_metadata_file(self, project_id: str, experiment_id: str) -> Path:
        """Return the path to an experiment's `experiment.json`."""
        return self.experiment_dir(project_id, experiment_id) / "experiment.json"

    def experiment_metrics_file(self, project_id: str, experiment_id: str) -> Path:
        """Return the path to an experiment's `metrics.json`."""
        return self.experiment_dir(project_id, experiment_id) / "metrics.json"

    def experiment_checkpoints_dir(self, project_id: str, experiment_id: str) -> Path:
        """Return the path to an experiment's checkpoints folder."""
        return self.experiment_dir(project_id, experiment_id) / "checkpoints"

    def experiment_logs_dir(self, project_id: str, experiment_id: str) -> Path:
        """Return the path to an experiment's logs folder."""
        return self.experiment_dir(project_id, experiment_id) / "logs"

    def evaluations_dir(self, project_id: str) -> Path:
        """Return the path to a project's evaluations folder."""
        return self.project_dir(project_id) / "evaluations"

    def evaluations_index_file(self, project_id: str) -> Path:
        """Return the path to `evaluations_index.json`."""
        return self.evaluations_dir(project_id) / "evaluations_index.json"

    def evaluation_dir(self, project_id: str, evaluation_id: str) -> Path:
        """Return the path to an evaluation folder."""
        safe_evaluation_id = _validate_identifier(evaluation_id, field_name="evaluation_id")
        return self.evaluations_dir(project_id) / safe_evaluation_id

    def evaluation_metadata_file(self, project_id: str, evaluation_id: str) -> Path:
        """Return the path to an evaluation's metadata JSON."""
        return self.evaluation_dir(project_id, evaluation_id) / "evaluation.json"

    def evaluation_aggregate_file(self, project_id: str, evaluation_id: str) -> Path:
        """Return the path to an evaluation's aggregate metrics JSON."""
        return self.evaluation_dir(project_id, evaluation_id) / "aggregate.json"

    def evaluation_results_file(self, project_id: str, evaluation_id: str) -> Path:
        """Return the path to an evaluation's per-image results JSON."""
        return self.evaluation_dir(project_id, evaluation_id) / "results.json"

    def exports_dir(self, project_id: str) -> Path:
        """Return the path to a project's exports folder."""
        return self.project_dir(project_id) / "exports"

    def exports_index_file(self, project_id: str) -> Path:
        """Return the path to `exports_index.json`."""
        return self.exports_dir(project_id) / "exports_index.json"

    def export_dir(self, project_id: str, export_id: str) -> Path:
        """Return the path to an export folder."""
        safe_export_id = _validate_identifier(export_id, field_name="export_id")
        return self.exports_dir(project_id) / safe_export_id

    def export_metadata_file(self, project_id: str, export_id: str) -> Path:
        """Return the path to an export's metadata JSON."""
        return self.export_dir(project_id, export_id) / "export.json"

    def ensure_project_layout(self, project_id: str) -> None:
        """Create the standard folder structure for a new project."""
        directories = (
            self.project_dir(project_id),
            self.dataset_dir(project_id),
            self.dataset_images_dir(project_id),
            self.dataset_masks_dir(project_id),
            self.dataset_thumbnails_dir(project_id),
            self.experiments_dir(project_id),
            self.evaluations_dir(project_id),
            self.exports_dir(project_id),
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def project_exists(self, project_id: str) -> bool:
        """Check whether a project directory exists."""
        return self.project_dir(project_id).exists()

    def remove_project(self, project_id: str) -> None:
        """Delete a project directory and all of its contents."""
        project_directory = self.project_dir(project_id)
        if project_directory.exists():
            shutil.rmtree(project_directory)
