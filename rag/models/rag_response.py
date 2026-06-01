"""Pydantic response models for RAG queries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.models.llm_response import LLMResponse


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

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
    # pre-fetched qdrant vector for zero-cost mmr inter-chunk similarity
    vector: list[float] | None = Field(
        default=None,
        exclude=True,
        description="Pre-fetched embedding vector for MMR. Not in API output.",
    )
    # cross-encoder reranker score (sigmoid 0.0-1.0); none when reranker was not used
    reranker_score: float | None = Field(
        default=None,
        exclude=True,
        description="Cross-encoder score from reranker. Not in API output.",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only content."""
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
        """Convert a LangChain Document to a RetrievedChunk."""
        meta = getattr(doc, "metadata", {}) or {}

        # "vector" excluded so it doesn't leak into the catch-all extra_metadata
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
    """Confidence score paired with the computation method."""

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
    """Split latency measurements for the RAG pipeline."""

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
    """Output model for all RAG queries."""

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

    @model_validator(mode="after")
    def validate_cache_layer_consistency(self) -> RAGResponse:
        """Ensure cache_layer is only set when cache_hit is True."""
        if self.cache_layer is not None and not self.cache_hit:
            raise ValueError(
                "cache_layer cannot be set when cache_hit is False. "
                f"Got cache_layer='{self.cache_layer}' with cache_hit=False."
            )
        return self

    @field_validator("answer")
    @classmethod
    def validate_answer_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only answers."""
        if not value.strip():
            raise ValueError("Answer cannot be blank or whitespace only.")
        return value

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
        """Build a RAGResponse from a cached LLMResponse."""
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
        """Build a RAGResponse from a fresh LLM generation."""
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
