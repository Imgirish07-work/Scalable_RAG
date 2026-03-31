"""Agent response models.

AgentResponse wraps the synthesized final answer with full transparency
into what happened: which sub-queries ran, which succeeded, which failed,
and how the final answer was assembled. Provides to_rag_response() for
callers that don't need agent internals.
"""

# stdlib
from typing import Optional

# third-party
from pydantic import BaseModel, ConfigDict, Field

# internal
from rag.models.rag_response import (
    ConfidenceScore,
    RAGResponse,
    RAGTimings,
    RetrievedChunk,
)


class SubQueryResult(BaseModel):
    """Result of a single sub-query execution.

    Attributes:
        sub_query_id: ID of the sub-query that produced this result.
        query: The sub-query text that was executed.
        collection: Collection that was queried.
        answer: The answer from the RAG pipeline, or empty on failure.
        confidence: Confidence score from the RAG response.
        sources: Retrieved chunks used in the answer.
        success: Whether the sub-query produced a usable answer.
        failure_reason: Why the sub-query failed, if applicable.
        latency_ms: Execution time for this sub-query.
    """

    model_config = ConfigDict(frozen=True)

    sub_query_id: str
    query: str
    collection: str
    answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    success: bool = True
    failure_reason: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_rag_response(
        cls,
        sub_query_id: str,
        query: str,
        collection: str,
        response: RAGResponse,
        latency_ms: float,
    ) -> "SubQueryResult":
        """Create from a successful RAG pipeline response.

        Args:
            sub_query_id: ID of the originating sub-query.
            query: The sub-query text.
            collection: Collection that was queried.
            response: RAGResponse from the pipeline.
            latency_ms: Execution time in milliseconds.

        Returns:
            SubQueryResult marked as successful.
        """
        return cls(
            sub_query_id=sub_query_id,
            query=query,
            collection=collection,
            answer=response.answer,
            confidence=response.confidence.value if response.confidence else 0.0,
            sources=response.sources or [],
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
    ) -> "SubQueryResult":
        """Create from a failed sub-query execution.

        Args:
            sub_query_id: ID of the originating sub-query.
            query: The sub-query text.
            collection: Collection that was queried.
            reason: Human-readable failure description.
            latency_ms: Time spent before failure.

        Returns:
            SubQueryResult marked as failed.
        """
        return cls(
            sub_query_id=sub_query_id,
            query=query,
            collection=collection,
            success=False,
            failure_reason=reason,
            latency_ms=latency_ms,
        )


class AgentResponse(BaseModel):
    """Full agent response with sub-query transparency.

    Wraps the synthesized answer with metadata about the decomposition,
    individual sub-query results, and aggregate metrics.

    Attributes:
        answer: The final synthesized answer.
        sub_results: Results from each sub-query.
        plan_reasoning: The planner's decomposition reasoning.
        confidence: Aggregate confidence across sub-queries.
        total_sub_queries: Number of sub-queries planned.
        successful_sub_queries: Number that succeeded.
        failed_sub_queries: Number that failed.
        timings: Aggregate timing breakdown.
        request_id: Parent request ID for tracing.
        model_name: LLM model used for synthesis.
        prompt_tokens: Total prompt tokens across all calls.
        completion_tokens: Total completion tokens across all calls.
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
        """Collapse to a standard RAGResponse.

        Useful for callers that don't care about agent internals —
        the pipeline and backend can return a uniform type.

        Returns:
            RAGResponse with the synthesized answer and aggregate metadata.
        """
        # collect all sources from successful sub-queries
        all_sources = []
        for sr in self.sub_results:
            if sr.success:
                all_sources.extend(sr.sources)

        return RAGResponse.from_generation(
            answer=self.answer,
            sources=all_sources,
            timings=self.timings,
            confidence=self.confidence,
            rag_variant="agent",
            context_tokens_used=0,
            model_name=self.model_name,
            request_id=self.request_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
