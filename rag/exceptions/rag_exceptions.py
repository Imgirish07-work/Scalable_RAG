"""Exception hierarchy for RAG; LLM-layer errors propagate as-is."""


class RAGError(Exception):
    """Base exception for all RAG errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class RAGConfigError(RAGError):
    """Invalid RAG configuration or request parameters."""


class RAGRetrievalError(RAGError):
    """Retrieval step failed."""


class RAGContextError(RAGError):
    """Context assembly or ranking failed."""


class RAGGenerationError(RAGError):
    """LLM generation step failed within the RAG pipeline."""


class RAGQualityError(RAGError):
    """Answer quality check failed."""
