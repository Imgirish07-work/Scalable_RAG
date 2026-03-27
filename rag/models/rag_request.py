"""
RAG request models.

Design decisions:
    - RAGRequest carries core fields (query, collection). Simple callers
      send just these two and get sane defaults for everything else.
    - RAGConfig carries advanced overrides (retrieval mode, reranking,
      filters, temperature, variant selection). Nested on RAGRequest as
      an optional field with a default constructor — power users override
      per-request, simple callers never touch it.
    - ConversationTurn is a lightweight role+content pair for multi-turn
      context. Maps directly to BaseLLM.chat() message format via
      model_dump(). Kept minimal — no message IDs, no timestamps.
    - MetadataFilter maps to Qdrant payload filters. Supports eq, neq,
      gt, gte, lt, lte, in operators for structured retrieval filtering.

All models use Pydantic v2 with ConfigDict. Frozen where immutability
is desired (ConversationTurn), mutable where per-request modification
is needed (RAGConfig).

Integration points:
    - RAGConfig.temperature → passed as kwarg to BaseLLM.generate()
    - RAGConfig.rag_variant → used by RAGFactory to select variant class
    - RAGConfig.retrieval_mode → used by RAGFactory to select retriever
    - ConversationTurn → maps to BaseLLM.chat() message format
    - MetadataFilter → maps to Qdrant payload filter conditions
    - request_id → flows through to RAGResponse for end-to-end tracing
"""

import uuid
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

from config.settings import settings


# Supported values for validated Literal fields
SUPPORTED_RETRIEVAL_MODES = {"dense", "hybrid"}
SUPPORTED_RERANK_STRATEGIES = {"none", "mmr", "cross_encoder"}
SUPPORTED_CONFIDENCE_METHODS = {"retrieval", "llm", "hybrid"}
SUPPORTED_RAG_VARIANTS = {"simple", "corrective", "chain"}
SUPPORTED_FILTER_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte", "in"}

class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation.

    Designed to map directly to BaseLLM.chat() message format:
        turn.model_dump() → {"role": "user", "content": "..."}

    Both Gemini and OpenAI providers accept this format. GeminiProvider
    converts internally (assistant → model, system → user).

    Attributes:
        role: Message role. Must be 'user', 'assistant', or 'system'.
        content: Message text content. Cannot be empty.
    """

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
        """Reject blank or whitespace-only content.

        Args:
            value: Raw content string.

        Returns:
            Stripped content string.

        Raises:
            ValueError: If content is whitespace only.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be blank or whitespace only.")
        return stripped

class MetadataFilter(BaseModel):
    """A single metadata filter condition for retrieval.

    Maps to Qdrant payload filter conditions. Multiple MetadataFilters
    are combined with AND logic at the retriever level.

    Examples:
        MetadataFilter(field="source_file", value="report.pdf")
        MetadataFilter(field="page_number", value=5, operator="gte")
        MetadataFilter(field="content_type", value=["code", "table"], operator="in")

    Attributes:
        field: Metadata field name to filter on.
        value: Value to compare against. Type depends on operator.
        operator: Comparison operator. Default is 'eq' (exact match).
    """

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
        """Reject blank field names.

        Args:
            value: Raw field name string.

        Returns:
            Stripped field name.

        Raises:
            ValueError: If field name is whitespace only.
        """
        if not value.strip():
            raise ValueError("MetadataFilter field cannot be blank.")
        return value.strip()

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        """Validate operator is in the supported set.

        Args:
            value: Raw operator string.

        Returns:
            Lowercase operator string.

        Raises:
            ValueError: If operator is not supported.
        """
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_FILTER_OPERATORS:
            raise ValueError(
                f"Operator '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_FILTER_OPERATORS)}"
            )
        return cleaned


class RAGConfig(BaseModel):
    """Advanced configuration overrides for a RAG request.

    Every field has a sensible default. Simple callers use RAGConfig()
    and get production-ready settings. Power users (and Layer 8 agents)
    override specific fields per-request.

    This is nested on RAGRequest as config: RAGConfig = RAGConfig().
    One object in, one object out — clean for FastAPI serialization.

    Attributes:
        rag_variant: Which RAG variant to use. None means use
            settings.RAG_DEFAULT_VARIANT (the smart default pattern).
        retrieval_mode: Dense-only or hybrid (dense + SPLADE).
        top_k: Number of chunks to retrieve.
        rerank_strategy: Post-retrieval reranking. 'mmr' for diversity,
            'cross_encoder' for relevance, 'none' to skip.
        max_context_tokens: Token budget for assembled context. Prevents
            exceeding the LLM's context window.
        temperature: LLM sampling temperature. Passed as kwarg to
            BaseLLM.generate(). Lower = more deterministic.
        system_prompt: Optional system prompt override. If None, the
            variant uses its default prompt template.
        metadata_filters: Optional list of filters applied to retrieval.
            Combined with AND logic at the Qdrant level.
        include_sources: Whether to include retrieved chunks in the
            response. True for debugging/citation, False for speed.
        confidence_method: How confidence score is computed.
            'retrieval' = average retrieval similarity (free).
            'llm' = LLM self-assessment (1 extra call).
            'hybrid' = weighted combination of both.
    """

    model_config = ConfigDict(frozen=False)

    rag_variant: str | None = Field(
        default=None,
        description="RAG variant: simple, corrective. None = use settings default.",
    )
    retrieval_mode: str = Field(
        default="dense",
        description="Retrieval mode: dense or hybrid (dense + SPLADE)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve (1-50)",
    )
    rerank_strategy: str = Field(
        default="mmr",
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
    # CORAG Config
    max_hops: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Maximum retrieval hops for CoRAG variant. None uses settings default.",
    )

    # Field validators

    @field_validator("rag_variant")
    @classmethod
    def validate_rag_variant(cls, value: str | None) -> str | None:
        """Validate RAG variant if provided.

        None is valid — means use settings default.

        Args:
            value: Raw variant string or None.

        Returns:
            Lowercase variant string or None.

        Raises:
            ValueError: If variant is not in SUPPORTED_RAG_VARIANTS.
        """
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
        """Validate retrieval mode.

        Args:
            value: Raw retrieval mode string.

        Returns:
            Lowercase retrieval mode.

        Raises:
            ValueError: If mode is not supported.
        """
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
        """Validate rerank strategy.

        Args:
            value: Raw strategy string.

        Returns:
            Lowercase strategy string.

        Raises:
            ValueError: If strategy is not supported.
        """
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
        """Validate confidence scoring method.

        Args:
            value: Raw method string.

        Returns:
            Lowercase method string.

        Raises:
            ValueError: If method is not supported.
        """
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_CONFIDENCE_METHODS:
            raise ValueError(
                f"Confidence method '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_CONFIDENCE_METHODS)}"
            )
        return cleaned

    def resolve_variant(self) -> str:
        """Resolve the effective RAG variant.

        If rag_variant is set, use it. Otherwise fall back to
        settings.RAG_DEFAULT_VARIANT. This is the smart default
        pattern — explicit overrides win, settings provide the fallback.

        Returns:
            Resolved variant name string (e.g. 'simple', 'corrective').
        """
        if self.rag_variant is not None:
            return self.rag_variant
        return getattr(settings, "RAG_DEFAULT_VARIANT", "simple").strip().lower()

    def resolve_max_hops(self) -> int:
        """Resolve max_hops with smart default pattern.

        Returns:
            Explicit override if set, else settings default.
        """
        if self.max_hops is not None:
            return self.max_hops
        return settings.CHAIN_RAG_MAX_HOPS


class RAGRequest(BaseModel):
    """Input model for all RAG queries.

    Core fields (query, collection_name) are required. Everything else
    has sensible defaults via RAGConfig.

    Simple usage:
        request = RAGRequest(query="What is RAG?", collection_name="docs")

    Advanced usage:
        request = RAGRequest(
            query="What are the latest compliance rules?",
            collection_name="legal_docs",
            config=RAGConfig(
                rag_variant="corrective",
                retrieval_mode="hybrid",
                top_k=10,
                temperature=0.1,
                metadata_filters=[
                    MetadataFilter(field="year", value=2024, operator="gte")
                ],
            ),
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about compliance."),
                ConversationTurn(role="assistant", content="Compliance refers to..."),
            ],
        )

    Attributes:
        query: The user's question. Cannot be empty.
        collection_name: Qdrant collection to search. Cannot be empty.
        config: Advanced overrides. Defaults to RAGConfig() (all defaults).
        conversation_history: Optional previous turns for multi-turn RAG.
            Used by pre_process() to resolve pronouns and context.
        request_id: UUID for end-to-end request tracing. Auto-generated
            if not provided. Flows through to RAGResponse.
    """

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

    # Field validators

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only queries.

        Args:
            value: Raw query string.

        Returns:
            Stripped query string.

        Raises:
            ValueError: If query is whitespace only.
        """
        if not value.strip():
            raise ValueError("Query cannot be blank or whitespace only.")
        return value.strip()

    @field_validator("collection_name")
    @classmethod
    def validate_collection_not_blank(cls, value: str) -> str:
        """Reject blank collection names.

        Args:
            value: Raw collection name.

        Returns:
            Stripped collection name.

        Raises:
            ValueError: If collection name is whitespace only.
        """
        if not value.strip():
            raise ValueError("Collection name cannot be blank.")
        return value.strip()

    def get_chat_messages(self) -> list[dict] | None:
        """Convert conversation_history to BaseLLM.chat() format.

        Returns None if no conversation history exists. Otherwise returns
        a list of dicts compatible with both OpenAI and Gemini providers.

        Returns:
            List of {"role": str, "content": str} dicts, or None.
        """
        if not self.conversation_history:
            return None
        return [turn.model_dump() for turn in self.conversation_history]