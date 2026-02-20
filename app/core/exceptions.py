from __future__ import annotations


class AIStudioError(Exception):
    """Base exception for all application-level errors."""

    status_code = 500
    error_code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    @property
    def code(self) -> str:
        """Backward-compatible alias for `error_code`."""
        return self.error_code


class ValidationError(AIStudioError):
    """Raised when input data is invalid."""

    error_code = "VALIDATION_ERROR"
    status_code = 422


class NotFoundError(AIStudioError):
    """Raised when an entity is missing."""

    error_code = "NOT_FOUND"
    status_code = 404


class ConflictError(AIStudioError):
    """Raised when an operation cannot be completed due to existing state."""

    error_code = "CONFLICT"
    status_code = 409


class DatasetNotImportedError(ValidationError):
    """Raised when a project has no imported dataset yet."""

    error_code = "DATASET_NOT_IMPORTED"

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project {project_id} has no imported dataset.")


class SplitNotFoundError(NotFoundError):
    """Raised when a split name cannot be resolved."""

    error_code = "SPLIT_NOT_FOUND"
