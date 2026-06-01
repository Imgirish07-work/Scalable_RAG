"""Retrieval-only executor for agent sub-queries; runs them concurrently under a semaphore."""

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
    """Executes sub-queries as retrieval-only calls (no LLM generation)."""

    def __init__(
        self,
        store_factory: Callable[[str], Awaitable],
        ranker: ContextRanker,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        max_concurrent: int = 4,
    ) -> None:
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
        logical_collection: str = "",
    ) -> list[SubQueryResult]:
        """Execute all sub-query retrievals concurrently."""
        tasks = [
            self._retrieve_with_semaphore(
                sq, parent_request_id, user_id, logical_collection,
            )
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
        logical_collection: str = "",
    ) -> SubQueryResult:
        """Execute a single sub-query retrieval and rerank."""
        start = time.perf_counter()

        try:
            store = await self._store_factory(sub_query.collection)

            # cross_encoder needs more candidates so use ranker's retrieval_top_k
            coarse_top_k = self._ranker.retrieval_top_k
            retriever = RAGFactory.create_retriever(
                store=store,
                mode=self._retrieval_mode,
            )

            raw_chunks = await retriever.retrieve(
                sub_query.query,
                top_k=coarse_top_k,
                user_id=user_id,
                collection=logical_collection,
            )
            ranked_chunks = await self._ranker.rank(raw_chunks, sub_query.query)

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
        logical_collection: str = "",
    ) -> SubQueryResult:
        """Wrap retrieve_one with semaphore for concurrency control."""
        async with self._semaphore:
            return await self.retrieve_one(
                sub_query, parent_request_id, user_id, logical_collection,
            )
