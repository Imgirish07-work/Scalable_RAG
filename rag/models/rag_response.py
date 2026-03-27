"""
RAG response models.

Design decisions:
    - RetrievedChunk is a clean Pydantic model decoupled from LangChain's
      Document class. This prevents LangChain types from leaking into the
      API contract when FastAPI serves these responses in Layer 10.
    - RAGTimings splits latency into retrieval, ranking, and generation
      components. This tells you whether retrieval or LLM is the bottleneck
      without additional profiling.
    - ConfidenceScore carries both the value AND the method used to compute
      it. CorrectiveRAG can switch scoring strategies without changing the
      response contract.
    - RAGResponse wraps LLMResponse data (not the object itself — it's frozen
      and we need to add RAG-specific fields). Factory classmethods handle
      construction from cache hits vs fresh generation.
    - metadata dict on RetrievedChunk is a catch-all for anything the chunker
      tagged that doesn't have a dedicated field. Future-proofs for agents
      that might need arbitrary metadata signals.

Integration points:
    - RetrievedChunk.from_document() converts LangChain Document → chunk
    - RAGResponse.from_cache() builds response from CacheResult
    - RAGResponse.from_generation() builds response from fresh LLMResponse
    - request_id flows from RAGRequest → RAGResponse for tracing
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.models.llm_response import LLMResponse


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store.

    Decoupled from LangChain's Document class. The from_document()
    classmethod handles conversion. This model is what callers see —
    never raw LangChain objects.

    Attributes:
        content: Clean text content (post _restore_original_content).
        source_file: Original document filename.
        chunk_id: SHA-256 hash of chunk content (from hash_text()).
        relevance_score: Cosine similarity score from retrieval (0.0-1.0).
        section_heading: Section heading from structure_preserver, if any.
        page_number: Page number from document_loader, if any.
        content_type: Content type tag (text, code, table, list), if any.
        used_in_context: Whether this chunk was included in the final
            context sent to the LLM. May be False if truncated by
            token budget or filtered by ranker.
        metadata: Catch-all dict for any additional chunker metadata.
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
    def from_document(cls, doc: Any, relevance_score: float = 0.0) -> RetrievedChunk:
        """Convert a LangChain Document to a RetrievedChunk.

        Extracts known metadata fields into dedicated attributes.
        Remaining metadata goes into the catch-all dict.

        Args:
            doc: LangChain Document with page_content and metadata.
            relevance_score: Similarity score from retrieval.

        Returns:
            RetrievedChunk instance.
        """
        meta = getattr(doc, "metadata", {}) or {}

        # Extract known fields from metadata
        known_fields = {
            "source", "source_file", "chunk_id",
            "section_heading", "page_number", "content_type",
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
        )


class ConfidenceScore(BaseModel):
    """Confidence score with the method used to compute it.

    The method field future-proofs the model — CorrectiveRAG can switch
    from retrieval-based to LLM-based scoring without changing the
    response contract. Callers always know HOW the score was computed.

    Attributes:
        value: Confidence score between 0.0 and 1.0.
        method: How the score was computed: retrieval, llm, hybrid.
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

    Knowing whether retrieval or generation is the bottleneck is
    critical for optimization. One flat latency_ms number hides this.

    Attributes:
        retrieval_ms: Time spent in retrieve() step.
        ranking_ms: Time spent in rank() step (MMR, cross-encoder, etc.).
        generation_ms: Time spent in generate() step (LLM call).
        total_ms: Wall-clock time for the entire query() pipeline.
            May be slightly greater than sum of parts due to overhead
            (cache checks, context assembly, serialization).
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
    diagnostics, confidence scoring, and cache metadata.

    Every RAG variant returns this exact model. Pipeline and agent
    layers import ONLY RAGResponse — never raw LLMResponse for
    RAG queries.

    Attributes:
        answer: The generated answer text.
        sources: Retrieved chunks with relevance scores and metadata.
            Empty list if include_sources=False in RAGConfig.
        timings: Split latency measurements (retrieval, ranking, gen).
        confidence: Confidence score with computation method.
        cache_hit: Whether the answer was served from cache.
        cache_layer: Which cache layer served the hit (L1, L2, semantic).
            None if cache_hit is False.
        rag_variant: Which variant produced this response.
        context_tokens_used: How many tokens the assembled context consumed.
            Critical for token budget optimization in Layer 5.
        model_name: Which LLM model generated the answer.
        request_id: Traces back to the originating RAGRequest.
        prompt_tokens: Input tokens consumed by the LLM call.
        completion_tokens: Output tokens generated by the LLM call.
        low_confidence: Flag set by CorrectiveRAG when relevance is below
            threshold after all retries. Caller decides how to handle.
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
        description="CorrectiveRAG flag: relevance below threshold after retries",
    )

    # Cross-field validators

    @model_validator(mode="after")
    def validate_cache_layer_consistency(self) -> RAGResponse:
        """Cache layer should only be set when cache_hit is True.

        Returns:
            Self if valid.

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
    ) -> RAGResponse:
        """Build RAGResponse from a cached LLMResponse.

        Used by BaseRAG.query() when cache returns a hit. No retrieval
        or generation timings — only the cache lookup latency.

        Args:
            cached_response: The LLMResponse from cache.
            request_id: Request ID from the originating RAGRequest.
            rag_variant: Which variant was configured (even though cache served).
            cache_layer: Which cache layer served the hit (L1, L2, semantic).
            lookup_latency_ms: Time spent on cache lookup.
            sources: Retrieved chunks restored from the cache entry.

        Returns:
            RAGResponse with cache_hit=True and minimal timings.
        """
        return cls(
            answer=cached_response.text,
            sources=sources,
            timings=RAGTimings(total_ms=lookup_latency_ms),
            confidence=ConfidenceScore(value=1.0, method="cache"),
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
        """Build RAGResponse from a fresh LLM generation.

        Used by BaseRAG.query() after the full pipeline completes
        (retrieve → rank → assemble → generate).

        Args:
            answer: The generated answer text.
            llm_response: The raw LLMResponse from the LLM provider.
            sources: Retrieved chunks with relevance scores.
            timings: Split latency measurements.
            confidence: Confidence score with computation method.
            request_id: Request ID from the originating RAGRequest.
            rag_variant: Which variant produced this response.
            context_tokens_used: Tokens consumed by assembled context.
            low_confidence: CorrectiveRAG flag for below-threshold relevance.

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