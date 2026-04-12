"""
Pydantic request models for RAG queries.

Design:
    Dataclass-style Pydantic v2 models with ConfigDict. RAGRequest carries
    the required fields (query, collection_name) and delegates all advanced
    options to the nested RAGConfig. Simple callers send only query and
    collection_name and get sensible production defaults. Power users or
    agent layers override specific RAGConfig fields per-request.
    ConversationTurn maps directly to BaseLLM.chat() message format via
    model_dump(). MetadataFilter maps to Qdrant payload filter conditions
    with AND logic across multiple filters.

Chain of Responsibility:
    Created by the API layer or agent → passed to RAGPipeline.query()
    → forwarded to BaseRAG.query() → config fields read by the pipeline
    to select variant, retriever, and generation parameters.

Dependencies:
    pydantic (BaseModel, Field, ConfigDict, field_validator)
    config.settings (settings)
"""

import uuid
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

from config.settings import settings


# Allowed values for Literal-validated fields
SUPPORTED_RETRIEVAL_MODES = {"dense", "hybrid"}
SUPPORTED_RERANK_STRATEGIES = {"none", "mmr", "cross_encoder"}
SUPPORTED_CONFIDENCE_METHODS = {"retrieval", "llm", "hybrid"}
SUPPORTED_RAG_VARIANTS = {"simple", "chain"}
SUPPORTED_FILTER_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte", "in"}


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation.

    Maps directly to the BaseLLM.chat() message format via model_dump():
        turn.model_dump() -> {"role": "user", "content": "..."}

    Both Gemini and OpenAI providers accept this format. GeminiProvider
    converts role names internally (assistant -> model, system -> user).

    Attributes:
        role: Message role. Must be 'user', 'assistant', or 'system'.
        content: Message text content. Cannot be empty or whitespace-only.
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
    """A single metadata filter condition for retrieval scoping.

    Maps to Qdrant payload filter conditions. Multiple MetadataFilters on
    a request are combined with AND logic at the retriever level.

    Examples:
        MetadataFilter(field="source_file", value="report.pdf")
        MetadataFilter(field="page_number", value=5, operator="gte")
        MetadataFilter(field="content_type", value=["code", "table"], operator="in")

    Attributes:
        field: Metadata field name to filter on.
        value: Value to compare against. Type depends on the operator.
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
            ValueError: If operator is not in SUPPORTED_FILTER_OPERATORS.
        """
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_FILTER_OPERATORS:
            raise ValueError(
                f"Operator '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_FILTER_OPERATORS)}"
            )
        return cleaned


class RAGConfig(BaseModel):
    """Advanced per-request configuration overrides for the RAG pipeline.

    Every field has a sensible default. Simple callers use RAGConfig() and
    get production-ready settings. Power users and agent layers override
    specific fields. Nested on RAGRequest as config: RAGConfig = RAGConfig().

    Attributes:
        rag_variant: Which RAG variant to use. None means use
            settings.RAG_DEFAULT_VARIANT (the smart default pattern).
        retrieval_mode: Dense-only or hybrid (dense + SPLADE).
        top_k: Number of chunks to retrieve per query.
        rerank_strategy: Post-retrieval reranking strategy.
            'mmr' for diversity, 'cross_encoder' for relevance, 'none' to skip.
        max_context_tokens: Token budget for assembled context. Prevents
            exceeding the LLM's context window.
        temperature: LLM sampling temperature passed to BaseLLM.generate().
            Lower values produce more deterministic output.
        system_prompt: Optional system prompt override. If None, the variant
            uses its default prompt template.
        metadata_filters: Optional filters applied at retrieval.
            Combined with AND logic at the Qdrant level.
        include_sources: Whether to include retrieved chunks in the response.
        confidence_method: How confidence is computed.
            'retrieval' = average retrieval similarity (free).
            'llm' = LLM self-assessment (1 extra call).
            'hybrid' = weighted combination of both.
        max_hops: Maximum retrieval hops for ChainRAG. None uses the
            settings default.
    """

    model_config = ConfigDict(frozen=False)

    rag_variant: str | None = Field(
        default=None,
        description="RAG variant: simple, chain. None = use settings default.",
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
        # When RERANKER_ENABLED=true, default to cross_encoder so the neural
        # reranker actually runs. Otherwise fall back to the settings strategy.
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
    # ChainRAG config
    max_hops: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Maximum retrieval hops for CoRAG variant. None uses settings default.",
    )
    # Agent routing override — bypasses should_decompose() heuristic.
    # True: always route to agent (if configured). False: never route to agent.
    # None (default): auto-detect via should_decompose().
    force_agent: Optional[bool] = Field(
        default=None,
        description="Override agent routing: True=always, False=never, None=auto-detect.",
    )

    # Field validators

    @field_validator("rag_variant")
    @classmethod
    def validate_rag_variant(cls, value: str | None) -> str | None:
        """Validate the RAG variant name if provided.

        None is valid — it means use the settings default.

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
        """Validate the retrieval mode.

        Args:
            value: Raw retrieval mode string.

        Returns:
            Lowercase retrieval mode string.

        Raises:
            ValueError: If mode is not in SUPPORTED_RETRIEVAL_MODES.
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
        """Validate the reranking strategy.

        Args:
            value: Raw strategy string.

        Returns:
            Lowercase strategy string.

        Raises:
            ValueError: If strategy is not in SUPPORTED_RERANK_STRATEGIES.
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
        """Validate the confidence scoring method.

        Args:
            value: Raw method string.

        Returns:
            Lowercase method string.

        Raises:
            ValueError: If method is not in SUPPORTED_CONFIDENCE_METHODS.
        """
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_CONFIDENCE_METHODS:
            raise ValueError(
                f"Confidence method '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_CONFIDENCE_METHODS)}"
            )
        return cleaned

    def resolve_variant(self) -> str:
        """Resolve the effective RAG variant name.

        Explicit per-request override wins over the settings default.
        This is the smart default pattern: callers that care specify it,
        callers that don't get the global default.

        Returns:
            Resolved variant name string (e.g. 'simple', 'chain').
        """
        if self.rag_variant is not None:
            return self.rag_variant
        return getattr(settings, "RAG_DEFAULT_VARIANT", "simple").strip().lower()

    def resolve_max_hops(self) -> int:
        """Resolve the effective max_hops value for ChainRAG.

        Returns:
            Explicit per-request override if set, else the settings default.
        """
        if self.max_hops is not None:
            return self.max_hops
        return settings.CHAIN_RAG_MAX_HOPS


class RAGRequest(BaseModel):
    """Input model for all RAG queries.

    Core fields (query, collection_name) are required. Everything else
    has sensible defaults via the nested RAGConfig.

    Simple usage:
        request = RAGRequest(query="What is RAG?", collection_name="docs")

    Advanced usage:
        request = RAGRequest(
            query="What are the latest compliance rules?",
            collection_name="legal_docs",
            config=RAGConfig(
                rag_variant="chain",
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
        query: The user's question. Cannot be empty or whitespace-only.
        collection_name: Qdrant collection to search. Cannot be empty.
        config: Advanced overrides. Defaults to RAGConfig() (all defaults).
        conversation_history: Optional previous turns for multi-turn RAG.
            Used by pre_process() to resolve pronouns and trailing references.
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
            value: Raw collection name string.

        Returns:
            Stripped collection name.

        Raises:
            ValueError: If collection name is whitespace only.
        """
        if not value.strip():
            raise ValueError("Collection name cannot be blank.")
        return value.strip()

    def get_chat_messages(self) -> list[dict] | None:
        """Convert conversation_history to the BaseLLM.chat() message format.

        Returns None when no history exists. Returns a list of dicts
        compatible with both OpenAI and Gemini provider implementations.

        Returns:
            List of {"role": str, "content": str} dicts, or None.
        """
        if not self.conversation_history:
            return None
        return [turn.model_dump() for turn in self.conversation_history]
