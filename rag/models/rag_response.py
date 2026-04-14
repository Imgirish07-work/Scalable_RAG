"""
Pydantic response models for RAG queries.

Design:
    Dataclass-style Pydantic v2 models, frozen where immutability is
    required. RetrievedChunk is decoupled from LangChain's Document class
    so LangChain types never leak into the API contract. RAGTimings splits
    latency into retrieval, ranking, and generation components so operators
    can identify pipeline bottlenecks without extra profiling. ConfidenceScore
    carries both value and method so callers always know how confidence was
    computed. RAGResponse wraps LLMResponse data (not the object itself) and
    adds RAG-specific fields. Factory classmethods handle construction from
    cache hits vs fresh generation.

Chain of Responsibility:
    Constructed by BaseRAG.query() → returned to RAGPipeline → serialized
    by the FastAPI API layer. from_cache() is called on cache hits;
    from_generation() is called after the full pipeline completes.

Dependencies:
    pydantic (BaseModel, ConfigDict, Field, field_validator, model_validator)
    llm.models.llm_response (LLMResponse)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.models.llm_response import LLMResponse


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store.

    Decoupled from LangChain's Document class. The from_document()
    classmethod handles conversion at the retriever boundary. This is
    what all downstream layers see — never raw LangChain objects.

    Attributes:
        content: Clean text content (post _restore_original_content).
        source_file: Original document filename.
        chunk_id: SHA-256 hash of chunk content (from hash_text()).
        relevance_score: Cosine similarity score from retrieval (0.0–1.0).
        section_heading: Section heading from structure_preserver, if any.
        page_number: Page number from document_loader, if any.
        content_type: Content type tag (text, code, table, list), if any.
        used_in_context: True if this chunk was included in the final context
            sent to the LLM. False if excluded by token budget or ranker.
        metadata: Catch-all dict for additional chunker metadata not covered
            by dedicated fields. Future-proofs for arbitrary agent signals.
        vector: Pre-fetched embedding vector from Qdrant for zero-cost MMR
            inter-chunk similarity. Excluded from API and cache output.
        reranker_score: Cross-encoder score (sigmoid 0.0–1.0) from the
            reranker. None when the reranker was not used. Excluded from
            API and cache output.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(
        ...,
        min_length=1,
        description="Clean text content of the chunk",
    )
    source_file: str = Field(
        default="unknown",
        description="Original document filename",
    )
    chunk_id: str = Field(
        default="",
        description="SHA-256 hash of chunk content",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from retrieval",
    )
    section_heading: str | None = Field(
        default=None,
        description="Section heading from structure preserver",
    )
    page_number: int | None = Field(
        default=None,
        ge=0,
        description="Page number from document loader",
    )
    content_type: str | None = Field(
        default=None,
        description="Content type: text, code, table, list",
    )
    used_in_context: bool = Field(
        default=False,
        description="Was this chunk included in the final LLM context?",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from chunker",
    )
    # Internal pipeline field — excluded from API/cache serialization.
    # Carries the pre-fetched Qdrant embedding vector so MMR can compute
    # inter-chunk similarity without re-embedding (saves ~2-4s per query).
    vector: list[float] | None = Field(
        default=None,
        exclude=True,
        description="Pre-fetched embedding vector for MMR. Not in API output.",
    )
    # Internal pipeline field — excluded from API/cache serialization.
    # Cross-encoder score from the reranker (sigmoid 0.0-1.0). Used by
    # base_rag.query() to detect low-confidence retrievals before assembling
    # context. None when reranker was not used (dense/mmr-only paths).
    reranker_score: float | None = Field(
        default=None,
        exclude=True,
        description="Cross-encoder score from reranker. Not in API output.",
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
        if not value.strip():
            raise ValueError("Chunk content cannot be blank.")
        return value

    @classmethod
    def from_document(
        cls,
        doc: Any,
        relevance_score: float = 0.0,
        vector: list[float] | None = None,
    ) -> "RetrievedChunk":
        """Convert a LangChain Document to a RetrievedChunk.

        Extracts known metadata fields into dedicated attributes.
        Any remaining metadata keys go into the catch-all dict.

        Args:
            doc: LangChain Document with page_content and metadata dict.
            relevance_score: Cosine similarity score from retrieval.
            vector: Pre-fetched embedding vector from Qdrant (optional).
                Forwarded to enable zero-cost MMR diversity scoring.

        Returns:
            RetrievedChunk instance with all available fields populated.
        """
        meta = getattr(doc, "metadata", {}) or {}

        # Extract known fields from metadata — "vector" excluded so it
        # doesn't leak into the catch-all extra_metadata dict.
        known_fields = {
            "source", "source_file", "chunk_id",
            "section_heading", "page_number", "content_type",
            "vector", "reranker_score", "relevance_score", "original_content",
            "embed_content", "ingested_at", "char_count",
            "doc_id", "user_id", "chunk_index", "total_chunks",
        }
        extra_metadata = {
            k: v for k, v in meta.items() if k not in known_fields
        }

        return cls(
            content=doc.page_content,
            source_file=meta.get("source_file", meta.get("source", "unknown")),
            chunk_id=meta.get("chunk_id", ""),
            relevance_score=relevance_score,
            section_heading=meta.get("section_heading"),
            page_number=meta.get("page_number"),
            content_type=meta.get("content_type"),
            metadata=extra_metadata,
            vector=vector,
        )


class ConfidenceScore(BaseModel):
    """Confidence score paired with the computation method.

    Carrying the method alongside the value lets callers understand the
    signal quality. Different variants can switch scoring methods without
    changing the response contract.

    Attributes:
        value: Confidence score in range 0.0–1.0.
        method: How the score was computed: retrieval, llm, hybrid,
            reranker, chain_eval, or cache.
    """

    model_config = ConfigDict(frozen=True)

    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = no confidence, 1.0 = full confidence)",
    )
    method: str = Field(
        ...,
        min_length=1,
        description="Scoring method: retrieval, llm, hybrid",
    )


class RAGTimings(BaseModel):
    """Split latency measurements for the RAG pipeline.

    One flat latency_ms number hides whether retrieval or LLM generation
    is the bottleneck. This model exposes each step separately.

    Attributes:
        retrieval_ms: Time spent in the retrieve() step.
        ranking_ms: Time spent in the rank() step (MMR, cross-encoder, etc.).
        generation_ms: Time spent in the generate() step (LLM call).
        total_ms: Wall-clock time for the entire query() pipeline.
            May exceed the sum of individual steps due to overhead in
            cache checks, context assembly, and serialization.
    """

    model_config = ConfigDict(frozen=True)

    retrieval_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent in retrieval step (ms)",
    )
    ranking_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent in ranking step (ms)",
    )
    generation_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent in LLM generation step (ms)",
    )
    total_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time for entire pipeline (ms)",
    )


class RAGResponse(BaseModel):
    """Output model for all RAG queries.

    Wraps the LLM-generated answer with retrieval context, timing
    diagnostics, confidence scoring, and cache metadata. Every RAG
    variant returns this exact model. Pipeline and agent layers import
    only RAGResponse — never raw LLMResponse for RAG queries.

    Attributes:
        answer: The generated answer text.
        sources: Retrieved chunks with relevance scores and metadata.
            Empty list when include_sources=False in RAGConfig.
        timings: Split latency measurements (retrieval, ranking, generation).
        confidence: Confidence score with computation method.
        cache_hit: True if the answer was served from cache.
        cache_layer: Cache layer that served the hit (L1, L2, semantic).
            None when cache_hit is False.
        rag_variant: Which variant produced this response.
        context_tokens_used: Tokens consumed by the assembled context.
        model_name: LLM model that generated the answer.
        request_id: Traces back to the originating RAGRequest.
        prompt_tokens: Input tokens consumed by the LLM call.
        completion_tokens: Output tokens generated by the LLM call.
        low_confidence: Flag set when relevance is below threshold.
            Caller decides how to surface this.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        ...,
        min_length=1,
        description="Generated answer text",
    )
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks with scores and metadata",
    )
    timings: RAGTimings = Field(
        default_factory=RAGTimings,
        description="Split latency measurements",
    )
    confidence: ConfidenceScore = Field(
        ...,
        description="Confidence score with computation method",
    )
    cache_hit: bool = Field(
        default=False,
        description="Was this served from cache?",
    )
    cache_layer: str | None = Field(
        default=None,
        description="Cache layer that served the hit (L1, L2, semantic)",
    )
    rag_variant: str = Field(
        ...,
        description="Which RAG variant produced this response",
    )
    context_tokens_used: int = Field(
        default=0,
        ge=0,
        description="Tokens consumed by assembled context",
    )
    model_name: str = Field(
        default="",
        description="LLM model that generated the answer",
    )
    request_id: str = Field(
        default="",
        description="Request ID from originating RAGRequest",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Input tokens consumed by LLM",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Output tokens generated by LLM",
    )
    low_confidence: bool = Field(
        default=False,
        description="Low confidence flag: retrieval relevance below threshold",
    )

    # Cross-field validators

    @model_validator(mode="after")
    def validate_cache_layer_consistency(self) -> RAGResponse:
        """Ensure cache_layer is only set when cache_hit is True.

        Returns:
            Self if the combination is valid.

        Raises:
            ValueError: If cache_layer is set but cache_hit is False.
        """
        if self.cache_layer is not None and not self.cache_hit:
            raise ValueError(
                "cache_layer cannot be set when cache_hit is False. "
                f"Got cache_layer='{self.cache_layer}' with cache_hit=False."
            )
        return self

    @field_validator("answer")
    @classmethod
    def validate_answer_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only answers.

        Args:
            value: Raw answer string.

        Returns:
            Stripped answer string.

        Raises:
            ValueError: If answer is whitespace only.
        """
        if not value.strip():
            raise ValueError("Answer cannot be blank or whitespace only.")
        return value

    # Factory classmethods

    @classmethod
    def from_cache(
        cls,
        cached_response: LLMResponse,
        request_id: str,
        rag_variant: str,
        cache_layer: str,
        lookup_latency_ms: float = 0.0,
        sources: list = [],
        confidence_value: float = 0.0,
    ) -> RAGResponse:
        """Build a RAGResponse from a cached LLMResponse.

        Used by BaseRAG.query() when the cache returns a hit. Retrieval
        and generation timings are omitted — only cache lookup latency is set.

        Args:
            cached_response: The LLMResponse retrieved from cache.
            request_id: Request ID from the originating RAGRequest.
            rag_variant: Variant configured for this request (informational).
            cache_layer: Cache layer that served the hit (L1, L2, semantic).
            lookup_latency_ms: Time spent on the cache lookup.
            sources: Retrieved chunks restored from the cache entry.
            confidence_value: Confidence score stored with the cache entry.

        Returns:
            RAGResponse with cache_hit=True and minimal timings.
        """
        return cls(
            answer=cached_response.text,
            sources=sources,
            timings=RAGTimings(total_ms=lookup_latency_ms),
            confidence=ConfidenceScore(value=confidence_value, method="cache"),
            cache_hit=True,
            cache_layer=cache_layer,
            rag_variant=rag_variant,
            context_tokens_used=0,
            model_name=cached_response.model,
            request_id=request_id,
            prompt_tokens=cached_response.prompt_tokens,
            completion_tokens=cached_response.completion_tokens,
        )

    @classmethod
    def from_generation(
        cls,
        answer: str,
        llm_response: LLMResponse,
        sources: list[RetrievedChunk],
        timings: RAGTimings,
        confidence: ConfidenceScore,
        request_id: str,
        rag_variant: str,
        context_tokens_used: int = 0,
        low_confidence: bool = False,
    ) -> RAGResponse:
        """Build a RAGResponse from a fresh LLM generation.

        Used by BaseRAG.query() after the full pipeline completes
        (retrieve → rank → assemble → generate).

        Args:
            answer: The generated answer text.
            llm_response: Raw LLMResponse from the LLM provider.
            sources: Retrieved chunks with relevance scores.
            timings: Split latency measurements for the pipeline.
            confidence: Confidence score with computation method.
            request_id: Request ID from the originating RAGRequest.
            rag_variant: Which variant produced this response.
            context_tokens_used: Tokens consumed by the assembled context.
            low_confidence: True if the variant flagged below-threshold relevance.

        Returns:
            RAGResponse with cache_hit=False and full diagnostics.
        """
        return cls(
            answer=answer,
            sources=sources,
            timings=timings,
            confidence=confidence,
            cache_hit=False,
            cache_layer=None,
            rag_variant=rag_variant,
            context_tokens_used=context_tokens_used,
            model_name=llm_response.model,
            request_id=request_id,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            low_confidence=low_confidence,
        )
