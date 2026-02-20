from __future__ import annotations

from fastapi import Request

from app.core.dataset_service import DatasetService
from app.core.project_service import ProjectService
from app.core.split_service import SplitService


def get_project_service(request: Request) -> ProjectService:
    """Resolve the shared project service instance from app state."""
    return request.app.state.project_service


def get_dataset_service(request: Request) -> DatasetService:
    """Resolve the shared dataset service instance from app state."""
    return request.app.state.dataset_service


def get_split_service(request: Request) -> SplitService:
    """Resolve the shared split service instance from app state."""
    return request.app.state.split_service
