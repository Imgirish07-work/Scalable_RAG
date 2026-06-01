"""Outermost exception hierarchy. Inner-layer actionable errors (LLMAuthError, LLMRateLimitError) propagate as-is."""


class PipelineError(Exception):
    """Base exception for all pipeline-layer errors."""

    def __init__(self, message: str, details: dict = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class PipelineInitError(PipelineError):
    """Raised when pipeline initialization fails."""


class PipelineValidationError(PipelineError):
    """Raised when input validation fails before execution."""


class PipelineIngestionError(PipelineError):
    """Raised when document ingestion fails."""


class PipelineFallbackExhaustedError(PipelineError):
    """Raised when all fallback strategies have been exhausted."""
