"""
ChunkRetriever — retrieval-only executor for agent sub-queries.

Design:
    Executes each sub-query as a hybrid-retrieve + rerank call with NO LLM
    generation. Sub-queries run concurrently up to max_concurrent via a
    semaphore. Individual failures are caught and recorded as failed results
    so a single bad sub-query never aborts the entire batch.

Chain of Responsibility:
    AgentOrchestrator → ChunkRetriever.retrieve_all()
    → asyncio.gather() per sub-query → RAGFactory.create_retriever()
    → BaseRetriever.retrieve() → ContextRanker.rank()
    → List[SubQueryResult] returned to orchestrator.

Dependencies:
    asyncio, agents.models.*, rag.context.context_ranker,
    rag.rag_factory, config.settings
"""

import asyncio
import time
from typing import Awaitable, Callable

from agents.models.agent_request import SubQuery
from agents.models.agent_response import SubQueryResult
from rag.context.context_ranker import ContextRanker
from rag.rag_factory import RAGFactory
from utils.logger import get_logger

logger = get_logger(__name__)


class ChunkRetriever:
    """Executes sub-queries as retrieval-only calls (no LLM generation).

    Builds a fresh retriever per sub-query (lightweight — no model loading)
    and applies the shared ContextRanker for reranking. Runs sub-queries
    in parallel bounded by a semaphore.

    Attributes:
        _store_factory: Async callable returning QdrantStore for a collection.
        _ranker: Shared ContextRanker instance (reranker + MMR).
        _retrieval_mode: 'dense' or 'hybrid'.
        _top_k: Max chunks to return per sub-query after reranking.
        _semaphore: Controls max concurrent sub-query executions.
    """

    def __init__(
        self,
        store_factory: Callable[[str], Awaitable],
        ranker: ContextRanker,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        max_concurrent: int = 4,
    ) -> None:
        """Initialize ChunkRetriever.

        Args:
            store_factory: Async callable: collection_name → QdrantStore.
            ranker: Shared ContextRanker for reranking retrieved chunks.
            retrieval_mode: 'dense' or 'hybrid' retrieval.
            top_k: Max chunks to return per sub-query after reranking.
            max_concurrent: Max parallel sub-query executions.
        """
        self._store_factory = store_factory
        self._ranker = ranker
        self._retrieval_mode = retrieval_mode
        self._top_k = top_k
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve_all(
        self,
        sub_queries: list[SubQuery],
        parent_request_id: str,
        user_id: str = "",
    ) -> list[SubQueryResult]:
        """Execute all sub-query retrievals concurrently.

        Failures in individual sub-queries are caught and converted to
        SubQueryResult.from_failure() — they never abort the batch.

        Args:
            sub_queries: Sub-queries from the decomposition plan.
            parent_request_id: Parent request ID for log tracing.

        Returns:
            SubQueryResult per sub-query in the same order as input.
        """
        tasks = [
            self._retrieve_with_semaphore(sq, parent_request_id, user_id)
            for sq in sub_queries
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for sub_query, outcome in zip(sub_queries, outcomes):
            if isinstance(outcome, Exception):
                logger.warning(
                    "Sub-query retrieval exception | id=%s | query='%s' | error=%s",
                    sub_query.sub_query_id, sub_query.query[:60], outcome,
                )
                results.append(SubQueryResult.from_failure(
                    sub_query_id=sub_query.sub_query_id,
                    query=sub_query.query,
                    collection=sub_query.collection,
                    reason=str(outcome),
                    latency_ms=0.0,
                    purpose=sub_query.purpose,
                ))
            else:
                results.append(outcome)

        return results

    async def retrieve_one(
        self,
        sub_query: SubQuery,
        parent_request_id: str,
        user_id: str = "",
    ) -> SubQueryResult:
        """Execute a single sub-query retrieval: fetch → rerank → return chunks.

        Args:
            sub_query: The sub-query to execute.
            parent_request_id: Parent request ID for log tracing.

        Returns:
            SubQueryResult with retrieved chunks or failure metadata.
        """
        start = time.perf_counter()

        try:
            store = await self._store_factory(sub_query.collection)

            # Use retrieval_top_k from ranker — cross_encoder needs more candidates.
            coarse_top_k = self._ranker.retrieval_top_k
            retriever = RAGFactory.create_retriever(
                store=store,
                mode=self._retrieval_mode,
            )

            raw_chunks = await retriever.retrieve(
                sub_query.query, top_k=coarse_top_k, user_id=user_id
            )
            ranked_chunks = await self._ranker.rank(raw_chunks, sub_query.query)

            # Cap to configured top_k after reranking.
            final_chunks = ranked_chunks[:self._top_k]
            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Sub-query retrieved | id=%s | chunks=%d | latency=%.1fms | query='%s'",
                sub_query.sub_query_id, len(final_chunks), latency_ms,
                sub_query.query[:60],
            )

            return SubQueryResult.from_retrieval(
                sub_query_id=sub_query.sub_query_id,
                query=sub_query.query,
                collection=sub_query.collection,
                chunks=final_chunks,
                latency_ms=latency_ms,
                purpose=sub_query.purpose,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Sub-query retrieval error | id=%s | query='%s' | error=%s",
                sub_query.sub_query_id, sub_query.query[:60], exc,
            )
            return SubQueryResult.from_failure(
                sub_query_id=sub_query.sub_query_id,
                query=sub_query.query,
                collection=sub_query.collection,
                reason=str(exc),
                latency_ms=latency_ms,
                purpose=sub_query.purpose,
            )

    async def _retrieve_with_semaphore(
        self,
        sub_query: SubQuery,
        parent_request_id: str,
        user_id: str = "",
    ) -> SubQueryResult:
        """Wrap retrieve_one with semaphore for concurrency control."""
        async with self._semaphore:
            return await self.retrieve_one(sub_query, parent_request_id, user_id)
