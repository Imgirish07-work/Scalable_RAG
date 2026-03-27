"""Pipeline-layer exceptions.

These are the outermost exceptions in the system. They catch and wrap
errors from inner layers (RAG, LLM, cache) only when the pipeline
needs to communicate a pipeline-specific failure (e.g., initialization,
validation, fallback exhaustion). Inner-layer exceptions that are
actionable by the caller (LLMAuthError, LLMRateLimitError) still
propagate as-is — the pipeline does NOT blanket-wrap everything.
"""

class PipelineError(Exception):
    """Base exception for all pipeline-layer errors.

    Attributes:
        message: Human-readable error description.
        details: Structured dict for logging and debugging.
    """

    def __init__(self, message: str, details: dict = None) -> None:
        """Initialize PipelineError.

        Args:
            message: Human-readable error description.
            details: Optional structured context for logging.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class PipelineInitError(PipelineError):
    """Raised when pipeline initialization fails.

    Covers subsystem boot failures — Qdrant unreachable, Redis down,
    embedding model load failure, etc.
    """


class PipelineValidationError(PipelineError):
    """Raised when input validation fails before execution.

    Covers empty queries, unknown collections, invalid config
    combinations. Caught early, before any LLM or retrieval calls.
    """


class PipelineIngestionError(PipelineError):
    """Raised when document ingestion fails.

    Covers file load errors, chunking failures, and vector store
    write failures during the ingest path.
    """


class PipelineFallbackExhaustedError(PipelineError):
    """Raised when all fallback strategies have been exhausted.

    This is the last-resort error — primary variant failed, fallback
    variant failed, cached response unavailable. The pipeline has
    no more options.
    """