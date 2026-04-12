"""
Custom exception hierarchy for RAG-specific failure modes.

Design:
    Every RAG component translates internal failures into this hierarchy.
    Pipeline and agent layers catch only these exceptions — never raw
    Qdrant, embedding, or LLM errors directly. RAGError is the base:
    catching it catches all RAG failures uniformly. LLM-specific errors
    (auth, rate limit, timeout) are NOT re-wrapped here; they propagate
    as-is from the LLM layer.

Chain of Responsibility:
    Raised by DenseRetriever / HybridRetriever, ContextAssembler,
    ContextRanker, BaseRAG variants → caught by the pipeline layer
    (RAGPipeline) or the API layer for structured error responses.

Dependencies:
    None (stdlib only).

Hierarchy:
    RAGError
    ├── RAGConfigError        (invalid request config or settings)
    ├── RAGRetrievalError     (retrieval failed — vector DB, embedding, etc.)
    ├── RAGContextError       (context assembly failed — token budget, empty docs)
    ├── RAGGenerationError    (LLM generation failed within RAG pipeline)
    └── RAGQualityError       (answer quality check failed — low confidence, etc.)
"""


class RAGError(Exception):
    """Base exception for all RAG errors.

    All RAG-specific exceptions are subclasses of this. Pipeline code
    catches RAGError to handle any RAG failure uniformly without importing
    every individual subclass.

    Attributes:
        message: Human-readable error description.
        details: Optional dict with structured context for logging/debugging.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize RAGError.

        Args:
            message: Human-readable error description.
            details: Optional structured context for logging and debugging.
                Example: {"query": "...", "collection": "...", "top_k": 5}
        """
        super().__init__(message)
        self.details = details or {}


class RAGConfigError(RAGError):
    """Invalid RAG configuration or request parameters.

    Raised when:
        - RAGRequest validation fails (empty query, invalid collection name).
        - RAGConfig has conflicting or unsupported options.
        - Unknown RAG variant is requested from the factory.
        - Settings contain invalid values for RAG parameters.
    """


class RAGRetrievalError(RAGError):
    """Retrieval step failed.

    Raised when:
        - Vector store query fails (Qdrant connection error, collection not found).
        - Embedding generation fails for the query.
        - Retriever returns no results and empty results are not acceptable.
        - Hybrid retrieval sparse component fails.

    Note:
        Transient failures (network timeouts to Qdrant) should be retried at
        the retriever level before raising this. This exception means retrieval
        genuinely failed after all retry attempts.
    """


class RAGContextError(RAGError):
    """Context assembly or ranking failed.

    Raised when:
        - No documents survive relevance filtering (all below threshold).
        - Token budget is exhausted and cannot be resolved by truncation.
        - Context assembler receives an empty document list.
        - Ranker encounters an incompatible document format.
    """


class RAGGenerationError(RAGError):
    """LLM generation step failed within the RAG pipeline.

    Raised when:
        - LLM returns empty text after prompt assembly.
        - Generation produces output that fails RAG-specific validation.
        - Generation produces output after exhausting all retry attempts.

    Note:
        LLM-layer errors (LLMAuthError, LLMRateLimitError, etc.) are NOT
        re-wrapped in this exception. They propagate as-is so the pipeline
        can apply provider-specific logic (retry, fallback, etc.).
        This exception covers RAG-specific generation failures only.
    """


class RAGQualityError(RAGError):
    """Answer quality check failed.

    Raised when:
        - Relevance evaluation falls below the minimum threshold
          after all retry attempts are exhausted.
        - Self-refinement (future RLM integration) detects persistent
          grounding failures.
        - Confidence score is below the acceptable floor and the variant
          is configured to reject low-confidence answers rather than flag them.

    Note:
        Most quality issues are surfaced as flags on RAGResponse
        (low_confidence=True, confidence_score < threshold) rather than
        exceptions. This exception is raised only when the variant is
        configured to treat quality failures as hard errors.
    """
