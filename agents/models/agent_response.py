"""Agent response models."""

from pydantic import BaseModel, ConfigDict, Field

from rag.models.rag_response import (
    ConfidenceScore,
    RAGResponse,
    RAGTimings,
    RetrievedChunk,
)


class SubQueryResult(BaseModel):
    """Result of a single sub-query retrieval."""

    model_config = ConfigDict(frozen=True)

    sub_query_id: str
    query: str
    collection: str
    purpose: str = ""
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    success: bool = True
    is_weak: bool = False
    failure_reason: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_retrieval(
        cls,
        sub_query_id: str,
        query: str,
        collection: str,
        chunks: list[RetrievedChunk],
        latency_ms: float,
        purpose: str = "",
    ) -> "SubQueryResult":
        """Create from a successful retrieval result."""
        confidence = (
            sum(c.reranker_score if c.reranker_score is not None else c.relevance_score
                for c in chunks) / len(chunks)
            if chunks else 0.0
        )
        return cls(
            sub_query_id=sub_query_id,
            query=query,
            collection=collection,
            purpose=purpose,
            chunks=chunks,
            confidence=round(confidence, 4),
            success=True,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_failure(
        cls,
        sub_query_id: str,
        query: str,
        collection: str,
        reason: str,
        latency_ms: float,
        purpose: str = "",
    ) -> "SubQueryResult":
        """Create from a failed sub-query retrieval."""
        return cls(
            sub_query_id=sub_query_id,
            query=query,
            collection=collection,
            purpose=purpose,
            success=False,
            failure_reason=reason,
            latency_ms=latency_ms,
        )


class AgentResponse(BaseModel):
    """Full agent response with sub-query transparency."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    plan_reasoning: str = ""
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore(value=0.0, method="agent"),
    )
    total_sub_queries: int = Field(default=0, ge=0)
    successful_sub_queries: int = Field(default=0, ge=0)
    failed_sub_queries: int = Field(default=0, ge=0)
    timings: RAGTimings = Field(
        default_factory=lambda: RAGTimings(
            retrieval_ms=0.0, ranking_ms=0.0,
            generation_ms=0.0, total_ms=0.0,
        ),
    )
    request_id: str = ""
    model_name: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    def to_rag_response(self) -> RAGResponse:
        """Collapse to a standard RAGResponse for pipeline callers."""
        all_sources = [
            chunk
            for result in self.sub_results
            if result.success
            for chunk in result.chunks
        ]
        return RAGResponse(
            answer=self.answer,
            sources=all_sources,
            timings=self.timings,
            confidence=self.confidence,
            cache_hit=False,
            cache_layer=None,
            rag_variant="agent",
            context_tokens_used=0,
            model_name=self.model_name,
            request_id=self.request_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
