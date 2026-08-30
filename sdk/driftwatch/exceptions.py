class DriftWatchError(Exception):
    """Base exception for DriftWatch SDK errors."""


class ModelNotFoundError(DriftWatchError):
    """Raised when a model ID does not exist."""


class BaselineNotFoundError(DriftWatchError):
    """Raised when no baseline exists for a model."""


class APIError(DriftWatchError):
    """Raised when the API returns an unexpected error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")
