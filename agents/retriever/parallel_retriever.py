"""Parallel retriever — executes sub-queries concurrently.

Takes a DecompositionPlan and executes each sub-query as a RAG
pipeline call. Independent sub-queries run in parallel via
asyncio.gather() with a semaphore to respect rate limits.
Sequential sub-queries run one at a time.

The retriever does NOT implement retrieval logic — it delegates
to pipeline.query_raw() for each sub-query. This means every
sub-query gets caching, ranking, context assembly, and all
existing infrastructure for free.
"""

# stdlib
import asyncio
import time
from typing import Protocol

# internal
from agents.models.agent_request import DecompositionPlan, SubQuery
from agents.models.agent_response import SubQueryResult
from config.settings import settings
from rag.models.rag_request import RAGRequest
from rag.models.rag_response import RAGResponse
from utils.logger import get_logger

logger = get_logger(__name__)

# default concurrency cap — prevents rate-limit storms
_DEFAULT_MAX_CONCURRENT = 4


class PipelineProtocol(Protocol):
    """Protocol for pipeline dependency — avoids circular imports.

    The parallel retriever needs to call pipeline.query_raw(), but
    importing RAGPipeline directly would create a circular dependency.
    This protocol defines the minimal interface needed.
    """

    async def query_raw(self, request: RAGRequest) -> RAGResponse: ...


class ParallelRetriever:
    """Executes sub-queries concurrently via the RAG pipeline.

    Uses asyncio.Semaphore to cap concurrent LLM calls. Falls back
    to sequential execution if the plan is marked as not parallel-safe.

    Attributes:
        _pipeline: Pipeline instance for executing sub-queries.
        _max_concurrent: Maximum concurrent sub-query executions.
    """

    def __init__(
        self,
        pipeline: PipelineProtocol,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Initialize ParallelRetriever.

        Args:
            pipeline: Pipeline instance with query_raw() method.
            max_concurrent: Max concurrent sub-query executions.
        """
        self._pipeline = pipeline
        self._max_concurrent = max_concurrent

    async def execute(
        self,
        plan: DecompositionPlan,
        parent_request_id: str,
    ) -> list[SubQueryResult]:
        """Execute all sub-queries from a decomposition plan.

        Routes to parallel or sequential execution based on the
        plan's parallel_safe flag.

        Args:
            plan: DecompositionPlan with sub-queries.
            parent_request_id: Parent request ID for tracing.

        Returns:
            List of SubQueryResult, one per sub-query.
        """
        if plan.parallel_safe and len(plan.sub_queries) > 1:
            logger.info(
                "Executing %d sub-queries in parallel (max_concurrent=%d)",
                len(plan.sub_queries), self._max_concurrent,
            )
            return await self._execute_parallel(
                plan.sub_queries, parent_request_id,
            )

        logger.info(
            "Executing %d sub-queries sequentially",
            len(plan.sub_queries),
        )
        return await self._execute_sequential(
            plan.sub_queries, parent_request_id,
        )

    async def _execute_parallel(
        self,
        sub_queries: list[SubQuery],
        parent_request_id: str,
    ) -> list[SubQueryResult]:
        """Execute sub-queries concurrently with semaphore control.

        Args:
            sub_queries: List of sub-queries to execute.
            parent_request_id: Parent request ID for tracing.

        Returns:
            List of SubQueryResult in the same order as input.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def guarded_execute(sq: SubQuery) -> SubQueryResult:
            async with semaphore:
                return await self._execute_single(sq, parent_request_id)

        results = await asyncio.gather(
            *[guarded_execute(sq) for sq in sub_queries],
            return_exceptions=False,
        )
        return list(results)

    async def _execute_sequential(
        self,
        sub_queries: list[SubQuery],
        parent_request_id: str,
    ) -> list[SubQueryResult]:
        """Execute sub-queries one at a time.

        Args:
            sub_queries: List of sub-queries to execute.
            parent_request_id: Parent request ID for tracing.

        Returns:
            List of SubQueryResult in order.
        """
        results = []
        for sq in sub_queries:
            result = await self._execute_single(sq, parent_request_id)
            results.append(result)
        return results

    async def _execute_single(
        self,
        sub_query: SubQuery,
        parent_request_id: str,
    ) -> SubQueryResult:
        """Execute a single sub-query via the pipeline.

        Catches all exceptions and converts them to failed
        SubQueryResult — individual sub-query failures do NOT
        crash the entire agent execution.

        Args:
            sub_query: The sub-query to execute.
            parent_request_id: Parent request ID for tracing.

        Returns:
            SubQueryResult — success or failure.
        """
        start = time.perf_counter()

        try:
            request = sub_query.to_rag_request(parent_request_id)
            response = await self._pipeline.query_raw(request)
            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                "Sub-query '%s' succeeded in %.1fms, confidence=%.3f",
                sub_query.sub_query_id, elapsed,
                response.confidence.value if response.confidence else 0.0,
            )

            return SubQueryResult.from_rag_response(
                sub_query_id=sub_query.sub_query_id,
                query=sub_query.query,
                collection=sub_query.collection,
                response=response,
                latency_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000

            logger.warning(
                "Sub-query '%s' failed in %.1fms: %s",
                sub_query.sub_query_id, elapsed, exc,
            )

            return SubQueryResult.from_failure(
                sub_query_id=sub_query.sub_query_id,
                query=sub_query.query,
                collection=sub_query.collection,
                reason=str(exc),
                latency_ms=elapsed,
            )
