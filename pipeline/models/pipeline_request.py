"""
Simplified external-facing request and response models for the pipeline.

Design:
    PipelineQuery is the public API surface. Internal layers use RAGRequest
    and RAGConfig — but external callers (FastAPI endpoints, CLI, scripts)
    should not need to know about those internals. PipelineQuery translates
    to RAGRequest + RAGConfig via to_rag_request().

Chain of Responsibility:
    External caller constructs PipelineQuery → RAGPipeline.query() converts
    it to RAGRequest via to_rag_request() → BaseRAG.query() receives it.

Dependencies:
    pydantic, rag.models.rag_request
"""

# stdlib
from typing import Optional
from uuid import uuid4

# third-party
from pydantic import BaseModel, ConfigDict, Field, field_validator

# internal
from rag.models.rag_request import ConversationTurn, RAGConfig, RAGRequest


class PipelineQuery(BaseModel):
    """Simplified query interface for external callers.

    Exposes only the fields that external consumers need. Advanced
    users who need full control can bypass this and pass RAGRequest
    directly to pipeline.query_raw().

    Attributes:
        query: The user's natural language question.
        collection: Target Qdrant collection name.
        variant: RAG variant override ('simple'). None uses settings default.
        conversation_history: Optional prior turns for multi-turn queries.
        temperature: LLM temperature override. None uses settings default.
        top_k: Number of chunks to retrieve. None uses settings default.
        include_sources: Whether to include source chunks in the response.
        request_id: Optional caller-provided request ID for tracing.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's natural language question.",
    )
    collection: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Target Qdrant collection name.",
    )
    variant: Optional[str] = Field(
        default=None,
        description="RAG variant: 'simple'. None uses default.",
    )
    conversation_history: Optional[list[ConversationTurn]] = Field(
        default=None,
        description="Prior conversation turns for multi-turn context.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature override.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of chunks to retrieve.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source chunks in response.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Caller-provided request ID for tracing.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain profile: 'technical' or 'story'. None = no profile.",
    )
    user_id: str = Field(
        default="",
        description="Authenticated user ID. Empty string = no user scoping.",
    )

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: Optional[str]) -> Optional[str]:
        """Validate variant name against known variants.

        Args:
            v: Variant name or None.

        Returns:
            Lowercased variant name or None.

        Raises:
            ValueError: If variant is not a recognized name.
        """
        if v is None:
            return None
        allowed = {"simple"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            msg = f"Unknown variant '{v}'. Allowed: {sorted(allowed)}"
            raise ValueError(msg)
        return normalized

    def to_rag_request(self) -> RAGRequest:
        """Convert to internal RAGRequest + RAGConfig.

        Maps only the fields that were explicitly set by the caller.
        None values are left as None so RAGConfig uses settings defaults.

        Returns:
            RAGRequest ready for BaseRAG.query().
        """
        from rag.domain_profiles import apply_domain_profile

        # Only pass optional fields when explicitly set — RAGConfig fields
        # top_k and temperature are non-optional (int/float), so passing None
        # would fail Pydantic validation. Omitting them uses the field defaults.
        config_kwargs: dict = {
            "rag_variant": self.variant,
            "include_sources": self.include_sources,
            "domain": self.domain,
        }
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        # Apply domain profile defaults — caller-supplied values above win.
        config_kwargs = apply_domain_profile(config_kwargs, self.domain)

        config = RAGConfig(**config_kwargs)

        return RAGRequest(
            query=self.query,
            collection_name=self.collection,
            config=config,
            conversation_history=self.conversation_history,
            request_id=self.request_id or str(uuid4()),
            user_id=self.user_id,
        )


class PipelineHealthStatus(BaseModel):
    """Health check result for the pipeline and its subsystems.

    Attributes:
        ready: True only if ALL critical subsystems are healthy.
        llm: LLM provider status.
        vector_store: Qdrant status.
        cache: Cache subsystem status.
        details: Optional extra info per subsystem.
    """

    model_config = ConfigDict(frozen=True)

    ready: bool = Field(
        ...,
        description="True only if all critical subsystems are healthy.",
    )
    llm: str = Field(
        ...,
        description="LLM provider status: 'ok' or error description.",
    )
    vector_store: str = Field(
        ...,
        description="Qdrant status: 'ok' or error description.",
    )
    cache: str = Field(
        ...,
        description="Cache status: 'ok', 'degraded', or error description.",
    )
    details: dict = Field(
        default_factory=dict,
        description="Optional extra diagnostic info.",
    )


class IngestionResult(BaseModel):
    """Result of a document ingestion operation.

    Attributes:
        file_path: Path of the ingested file.
        collection: Target collection name.
        chunks_stored: Number of chunks successfully stored.
        total_chunks: Total chunks produced by chunker (before dedup).
        duplicates_skipped: Chunks skipped due to deduplication.
        elapsed_ms: Total ingestion time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str
    collection: str
    chunks_stored: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
