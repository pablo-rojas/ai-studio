from __future__ import annotations


class AIStudioError(Exception):
    """Base exception for all application-level errors."""

    code = "APP_ERROR"
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(AIStudioError):
    """Raised when input data is invalid."""

    code = "VALIDATION_ERROR"
    status_code = 422


class NotFoundError(AIStudioError):
    """Raised when an entity is missing."""

    code = "NOT_FOUND"
    status_code = 404


class ConflictError(AIStudioError):
    """Raised when an operation cannot be completed due to existing state."""

    code = "CONFLICT"
    status_code = 409
