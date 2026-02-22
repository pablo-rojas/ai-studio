from __future__ import annotations

from fastapi import Request

from app.core.dataset_service import DatasetService
from app.core.evaluation_service import EvaluationService
from app.core.export_service import ExportService
from app.core.project_service import ProjectService
from app.core.split_service import SplitService
from app.core.training_service import TrainingService


def get_project_service(request: Request) -> ProjectService:
    """Resolve the shared project service instance from app state."""
    return request.app.state.project_service


def get_dataset_service(request: Request) -> DatasetService:
    """Resolve the shared dataset service instance from app state."""
    return request.app.state.dataset_service


def get_split_service(request: Request) -> SplitService:
    """Resolve the shared split service instance from app state."""
    return request.app.state.split_service


def get_training_service(request: Request) -> TrainingService:
    """Resolve the shared training service instance from app state."""
    return request.app.state.training_service


def get_evaluation_service(request: Request) -> EvaluationService:
    """Resolve the shared evaluation service instance from app state."""
    return request.app.state.evaluation_service


def get_export_service(request: Request) -> ExportService:
    """Resolve the shared export service instance from app state."""
    return request.app.state.export_service
