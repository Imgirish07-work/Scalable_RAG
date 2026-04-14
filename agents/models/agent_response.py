"""
Agent response models.

Design:
    SubQueryResult captures the outcome of a single sub-query retrieval.
    Sub-queries in the new architecture perform retrieval-only (no per-query
    generation). All chunks are fused and a single LLM call generates the
    final answer. AgentResponse aggregates sub-results and exposes
    to_rag_response() for callers that don't need agent internals.

Chain of Responsibility:
    ChunkRetriever produces List[SubQueryResult] → ChunkQualityGate evaluates
    quality → AgentOrchestrator rewrites weak sub-queries → ContextFusion merges
    chunks → single LLM generation → AgentResponse → RAGPipeline.to_rag_response().

Dependencies:
    pydantic, rag.models.rag_response
"""

from pydantic import BaseModel, ConfigDict, Field

from rag.models.rag_response import (
    ConfidenceScore,
    RAGResponse,
    RAGTimings,
    RetrievedChunk,
)


class SubQueryResult(BaseModel):
    """Result of a single sub-query retrieval (no per-query LLM generation).

    Attributes:
        sub_query_id: Unique ID for tracing.
        query: Sub-query text executed (may differ from original if rewritten).
        collection: Qdrant collection queried.
        purpose: What this sub-query was meant to resolve.
        chunks: Retrieved and reranked chunks from the vector store.
        confidence: Average reranker/relevance score across chunks (0-1).
        success: True if usable chunks were retrieved.
        is_weak: True if chunks exist but quality is below threshold.
                 Weak sub-queries are rewritten once using the fast LLM.
        failure_reason: Why retrieval failed, populated only when success=False.
        latency_ms: Retrieval execution time in milliseconds.
    """

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
        """Create from a successful retrieval result.

        Args:
            sub_query_id: ID of the originating sub-query.
            query: The sub-query text (possibly rewritten by the fast LLM).
            collection: Collection that was queried.
            chunks: Retrieved and reranked chunks.
            latency_ms: Execution time in milliseconds.
            purpose: What this sub-query was meant to resolve.

        Returns:
            SubQueryResult marked as successful with computed confidence.
        """
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
        """Create from a failed sub-query retrieval.

        Args:
            sub_query_id: ID of the originating sub-query.
            query: The sub-query text.
            collection: Collection that was queried.
            reason: Human-readable failure description.
            latency_ms: Time spent before failure.
            purpose: What this sub-query was meant to resolve.

        Returns:
            SubQueryResult marked as failed with empty chunks.
        """
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
    """Full agent response with sub-query transparency.

    Attributes:
        answer: Final synthesized answer from the single generation LLM call.
        sub_results: Retrieval results from each sub-query.
        plan_reasoning: The planner's decomposition reasoning.
        confidence: Aggregate confidence across sub-queries.
        total_sub_queries: Number of sub-queries planned.
        successful_sub_queries: Number that returned usable chunks.
        failed_sub_queries: Number that returned no usable chunks.
        timings: Aggregate timing breakdown.
        request_id: Parent request ID for tracing.
        model_name: LLM model used for final synthesis.
        prompt_tokens: Total prompt tokens (plan + rewrites + synthesis).
        completion_tokens: Total completion tokens (plan + rewrites + synthesis).
    """

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
        """Collapse to a standard RAGResponse for pipeline callers.

        Returns:
            RAGResponse with the synthesized answer and all source chunks.
        """
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
