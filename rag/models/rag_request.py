"""Pydantic request models for RAG queries."""

import uuid
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

from config.settings import settings


SUPPORTED_RETRIEVAL_MODES = {"dense", "hybrid"}
SUPPORTED_RERANK_STRATEGIES = {"none", "mmr", "cross_encoder"}
SUPPORTED_CONFIDENCE_METHODS = {"retrieval", "llm", "hybrid"}
SUPPORTED_RAG_VARIANTS = {"simple"}
SUPPORTED_FILTER_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte", "in"}


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Message role: user, assistant, or system"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Message text content"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only content."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be blank or whitespace only.")
        return stripped


class MetadataFilter(BaseModel):
    """A single metadata filter condition for retrieval scoping."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(
        ...,
        min_length=1,
        description="Metadata field name to filter on",
    )
    value: Any = Field(
        ...,
        description="Value to compare against",
    )
    operator: str = Field(
        default="eq",
        description="Comparison operator: eq, neq, gt, gte, lt, lte, in",
    )

    @field_validator("field")
    @classmethod
    def validate_field_not_blank(cls, value: str) -> str:
        """Reject blank field names."""
        if not value.strip():
            raise ValueError("MetadataFilter field cannot be blank.")
        return value.strip()

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        """Validate operator is in the supported set."""
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_FILTER_OPERATORS:
            raise ValueError(
                f"Operator '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_FILTER_OPERATORS)}"
            )
        return cleaned


class RAGConfig(BaseModel):
    """Advanced per-request configuration overrides for the RAG pipeline."""

    model_config = ConfigDict(frozen=False)

    rag_variant: str | None = Field(
        default=None,
        description="RAG variant: 'simple'. None = use settings default.",
    )
    retrieval_mode: str = Field(
        default_factory=lambda: settings.RAG_RETRIEVAL_MODE,
        description="Retrieval mode: dense or hybrid (dense + SPLADE)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve (1-50)",
    )
    rerank_strategy: str = Field(
        # cross_encoder when RERANKER_ENABLED, else fall back to settings strategy
        default_factory=lambda: (
            "cross_encoder" if settings.RERANKER_ENABLED else settings.RAG_RERANK_STRATEGY
        ),
        description="Reranking strategy: none, mmr, cross_encoder",
    )
    max_context_tokens: int = Field(
        default=3072,
        ge=128,
        le=32768,
        description="Token budget for assembled context",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0.0-2.0)",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt override",
    )
    metadata_filters: list[MetadataFilter] | None = Field(
        default=None,
        description="Optional metadata filters for retrieval",
    )
    include_sources: bool = Field(
        default=True,
        description="Include retrieved chunks in response",
    )
    confidence_method: str = Field(
        default="retrieval",
        description="Confidence scoring method: retrieval, llm, hybrid",
    )
    force_agent: Optional[bool] = Field(
        default=None,
        description="Override agent routing: True=always, False=never, None=auto-detect.",
    )
    domain: str | None = Field(
        default=None,
        description="Domain profile: 'technical' or 'story'. None = no profile.",
    )
    min_context_chunks: int = Field(
        default_factory=lambda: settings.RAG_MIN_CONTEXT_CHUNKS,
        ge=1,
        description="Minimum chunks passed to context assembly.",
    )
    reranker_score_threshold: float = Field(
        default_factory=lambda: settings.RERANKER_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum reranker score before MMR recovery triggers.",
    )

    @field_validator("rag_variant")
    @classmethod
    def validate_rag_variant(cls, value: str | None) -> str | None:
        """Validate the RAG variant name if provided."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_RAG_VARIANTS:
            raise ValueError(
                f"RAG variant '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_RAG_VARIANTS)}"
            )
        return cleaned

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """Validate the retrieval mode."""
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(
                f"Retrieval mode '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_RETRIEVAL_MODES)}"
            )
        return cleaned

    @field_validator("rerank_strategy")
    @classmethod
    def validate_rerank_strategy(cls, value: str) -> str:
        """Validate the reranking strategy."""
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_RERANK_STRATEGIES:
            raise ValueError(
                f"Rerank strategy '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_RERANK_STRATEGIES)}"
            )
        return cleaned

    @field_validator("confidence_method")
    @classmethod
    def validate_confidence_method(cls, value: str) -> str:
        """Validate the confidence scoring method."""
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_CONFIDENCE_METHODS:
            raise ValueError(
                f"Confidence method '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_CONFIDENCE_METHODS)}"
            )
        return cleaned

    def resolve_variant(self) -> str:
        """Resolve the effective RAG variant name; per-request override wins."""
        if self.rag_variant is not None:
            return self.rag_variant
        return getattr(settings, "RAG_DEFAULT_VARIANT", "simple").strip().lower()


class RAGRequest(BaseModel):
    """Input model for all RAG queries."""

    model_config = ConfigDict(frozen=False)

    query: str = Field(
        ...,
        min_length=1,
        description="The user's question",
    )
    collection_name: str = Field(
        ...,
        min_length=1,
        description="Qdrant collection name to search",
    )
    config: RAGConfig = Field(
        default_factory=RAGConfig,
        description="Advanced configuration overrides",
    )
    conversation_history: list[ConversationTurn] | None = Field(
        default=None,
        description="Previous conversation turns for multi-turn context",
    )
    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique request ID for tracing (auto-generated)",
    )
    user_id: str = Field(
        default="",
        description="Authenticated user ID. Empty string = no per-user scoping.",
    )
    logical_collection: str = Field(
        default="",
        description=(
            "Logical collection ('folder') within the user's corpus. "
            "Empty string searches all of the user's logical collections."
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only queries."""
        if not value.strip():
            raise ValueError("Query cannot be blank or whitespace only.")
        return value.strip()

    @field_validator("collection_name")
    @classmethod
    def validate_collection_not_blank(cls, value: str) -> str:
        """Reject blank collection names."""
        if not value.strip():
            raise ValueError("Collection name cannot be blank.")
        return value.strip()

    def get_chat_messages(self) -> list[dict] | None:
        """Convert conversation_history to the BaseLLM.chat() message format."""
        if not self.conversation_history:
            return None
        return [turn.model_dump() for turn in self.conversation_history]
