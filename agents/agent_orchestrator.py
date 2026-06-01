"""Coordinates the full complex query flow: plan, fetch, gate, rewrite, fuse, synthesize."""

import asyncio
import time
from typing import Awaitable, Callable, Optional

from agents.exceptions.agent_exceptions import AgentRetrievalError
from agents.fusion.context_fusion import ContextFusion
from agents.models.agent_request import SubQuery
from agents.models.agent_response import AgentResponse, SubQueryResult
from agents.planner.query_planner import QueryPlanner
from agents.prompts.agent_prompt_templates import (
    build_rewrite_prompt,
    build_synthesis_prompt,
)
from agents.quality.chunk_quality_gate import ChunkQualityGate
from agents.retriever.chunk_retriever import ChunkRetriever
from config.settings import settings
from llm.contracts.base_llm import BaseLLM
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.models.rag_request import RAGRequest
from rag.models.rag_response import ConfidenceScore, RAGTimings
from vectorstore.reranker import get_reranker
from utils.logger import get_logger

logger = get_logger(__name__)

_REWRITE_MAX_TOKENS = 120
_SYNTHESIS_MAX_TOKENS = 2048


class AgentOrchestrator:
    """Coordinates the complex query decomposition and synthesis flow."""

    def __init__(
        self,
        strong_llm: BaseLLM,
        fast_llm: BaseLLM,
        store_factory: Callable[[str], Awaitable],
        collections: dict[str, str],
        embeddings_fn: Optional[object] = None,
        max_concurrent: int = 4,
        fallback_llm: Optional[BaseLLM] = None,
    ) -> None:
        rerank_strategy = getattr(settings, "RAG_RERANK_STRATEGY", "cross_encoder")
        top_k = getattr(settings, "RAG_TOP_K", 5)
        max_context_tokens = getattr(settings, "RAG_MAX_CONTEXT_TOKENS", 3072)

        reranker = get_reranker() if rerank_strategy == "cross_encoder" else None
        ranker = ContextRanker(
            strategy=rerank_strategy,
            embeddings_fn=embeddings_fn,
            reranker=reranker,
            top_k=top_k,
        )
        assembler = ContextAssembler(llm=strong_llm, max_tokens=max_context_tokens)

        self._planner = QueryPlanner(llm=strong_llm, collections=collections)
        self._chunk_retriever = ChunkRetriever(
            store_factory=store_factory,
            ranker=ranker,
            retrieval_mode=getattr(settings, "RAG_RETRIEVAL_MODE", "hybrid"),
            top_k=top_k,
            max_concurrent=max_concurrent,
        )
        self._quality_gate = ChunkQualityGate()
        self._context_fusion = ContextFusion(ranker=ranker, assembler=assembler)
        self._strong_llm = strong_llm
        self._fast_llm = fast_llm
        self._fallback_llm = fallback_llm

    async def execute(self, request: RAGRequest) -> AgentResponse:
        """Execute the full complex query flow."""
        total_start = time.perf_counter()
        query = request.query
        request_id = request.request_id
        total_prompt_tokens = 0
        total_completion_tokens = 0

        logger.info(
            "Agent executing | request_id=%s | query='%s'",
            request_id, query[:100],
        )

        plan_start = time.perf_counter()
        plan = await self._planner.plan(query)
        plan_ms = (time.perf_counter() - plan_start) * 1000

        logger.info(
            "Agent plan complete | sub_queries=%d | parallel=%s | plan_ms=%.1f",
            len(plan.sub_queries), plan.parallel_safe, plan_ms,
        )

        retrieval_start = time.perf_counter()
        sub_results = await self._chunk_retriever.retrieve_all(
            sub_queries=plan.sub_queries,
            parent_request_id=request_id,
            user_id=request.user_id,
            logical_collection=request.logical_collection,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        sub_results = self._quality_gate.evaluate(sub_results)

        rewrite_start = time.perf_counter()
        sub_results, rewrite_tokens = await self._rewrite_and_refetch_weak(
            sub_results=sub_results,
            parent_request_id=request_id,
            user_id=request.user_id,
            logical_collection=request.logical_collection,
        )
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000
        total_prompt_tokens += rewrite_tokens[0]
        total_completion_tokens += rewrite_tokens[1]

        any_success = any(r.success and r.chunks for r in sub_results)
        if not any_success:
            raise AgentRetrievalError(
                message="All sub-queries failed — no chunks retrieved",
                details={
                    "request_id": request_id,
                    "sub_query_count": len(sub_results),
                    "failures": [r.failure_reason for r in sub_results],
                },
            )

        fusion_start = time.perf_counter()
        structured_context, used_chunks = await self._context_fusion.fuse(
            sub_results=sub_results,
            query=query,
        )
        fusion_ms = (time.perf_counter() - fusion_start) * 1000

        # both LLMs failing is degraded but preserves retrieval results in sources
        generation_start = time.perf_counter()
        gen_model_name = "unavailable"
        try:
            answer, gen_prompt_tokens, gen_completion_tokens, gen_model_name = await self._generate_answer(
                query=query,
                structured_context=structured_context,
            )
            total_prompt_tokens += gen_prompt_tokens
            total_completion_tokens += gen_completion_tokens
        except Exception as exc:
            logger.error(
                "All LLMs failed for synthesis — degraded response | "
                "request_id=%s | error=%s",
                request_id, exc,
            )
            answer = (
                "Generation unavailable — all LLM providers failed. "
                "Retrieved context is available in sources."
            )
        generation_ms = (time.perf_counter() - generation_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        successful = [r for r in sub_results if r.success and r.chunks]
        failed = [r for r in sub_results if not r.success or not r.chunks]
        confidence = _compute_confidence(sub_results)

        logger.info(
            "Agent complete | request_id=%s | total_ms=%.1f | "
            "sub_queries=%d/%d succeeded | confidence=%.3f",
            request_id, total_ms,
            len(successful), len(sub_results),
            confidence.value,
        )

        return AgentResponse(
            answer=answer,
            sub_results=sub_results,
            plan_reasoning=plan.reasoning,
            confidence=confidence,
            total_sub_queries=len(sub_results),
            successful_sub_queries=len(successful),
            failed_sub_queries=len(failed),
            timings=RAGTimings(
                retrieval_ms=round(retrieval_ms + rewrite_ms, 1),
                ranking_ms=round(fusion_ms, 1),
                generation_ms=round(plan_ms + generation_ms, 1),
                total_ms=round(total_ms, 1),
            ),
            request_id=request_id,
            model_name=gen_model_name,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

    async def _rewrite_and_refetch_weak(
        self,
        sub_results: list[SubQueryResult],
        parent_request_id: str,
        user_id: str = "",
        logical_collection: str = "",
    ) -> tuple[list[SubQueryResult], tuple[int, int]]:
        """Rewrite weak sub-queries with the fast LLM then re-retrieve once."""
        weak_results = [r for r in sub_results if r.is_weak]

        if not weak_results:
            return sub_results, (0, 0)

        logger.info(
            "Rewriting %d weak sub-queries | request_id=%s",
            len(weak_results), parent_request_id,
        )

        rewrite_tasks = [self._rewrite_one(r) for r in weak_results]
        rewrite_outcomes = await asyncio.gather(*rewrite_tasks, return_exceptions=True)

        total_prompt = 0
        total_completion = 0
        rewritten_sub_queries: list[tuple[SubQueryResult, SubQuery]] = []

        for result, outcome in zip(weak_results, rewrite_outcomes):
            if isinstance(outcome, Exception):
                logger.warning(
                    "Sub-query rewrite failed | id=%s | error=%s — keeping original",
                    result.sub_query_id, outcome,
                )
                continue

            rewritten_query, prompt_t, completion_t = outcome
            total_prompt += prompt_t
            total_completion += completion_t

            if not rewritten_query or rewritten_query.strip() == result.query.strip():
                logger.debug(
                    "Rewrite unchanged | id=%s — skipping re-retrieval",
                    result.sub_query_id,
                )
                continue

            logger.info(
                "Sub-query rewritten | id=%s | original='%s' | rewritten='%s'",
                result.sub_query_id, result.query[:60], rewritten_query[:60],
            )
            rewritten_sub_queries.append((result, SubQuery(
                query=rewritten_query,
                collection=result.collection,
                purpose=result.purpose,
                sub_query_id=result.sub_query_id,
            )))

        if rewritten_sub_queries:
            refetch_tasks = [
                self._chunk_retriever.retrieve_one(
                    sq, parent_request_id, user_id, logical_collection,
                )
                for _, sq in rewritten_sub_queries
            ]
            refetch_results = await asyncio.gather(*refetch_tasks, return_exceptions=True)

            replacements: dict[str, SubQueryResult] = {}
            for (original, _), new_result in zip(rewritten_sub_queries, refetch_results):
                if isinstance(new_result, Exception):
                    logger.warning(
                        "Re-retrieval failed after rewrite | id=%s | error=%s",
                        original.sub_query_id, new_result,
                    )
                else:
                    replacements[original.sub_query_id] = new_result

            sub_results = [
                replacements.get(r.sub_query_id, r) if r.is_weak else r
                for r in sub_results
            ]

        return sub_results, (total_prompt, total_completion)

    async def _rewrite_one(
        self,
        result: SubQueryResult,
    ) -> tuple[str, int, int]:
        """Rewrite a single weak sub-query using the fast LLM."""
        best_chunk = max(
            result.chunks,
            key=lambda c: c.reranker_score if c.reranker_score is not None else c.relevance_score,
        )
        system_prompt, user_prompt = build_rewrite_prompt(
            query=result.query,
            purpose=result.purpose,
            best_chunk_content=best_chunk.content,
        )
        response = await self._fast_llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=_REWRITE_MAX_TOKENS,
        )
        return response.text.strip(), response.prompt_tokens, response.completion_tokens

    async def _generate_answer(
        self,
        query: str,
        structured_context: str,
    ) -> tuple[str, int, int]:
        """Generate the final answer from fused context."""
        from llm.exceptions.llm_exceptions import LLMError

        system_prompt, user_prompt = build_synthesis_prompt(
            query=query,
            structured_context=structured_context,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        active_llm = self._strong_llm
        try:
            response = await self._strong_llm.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=_SYNTHESIS_MAX_TOKENS,
            )
        except LLMError as exc:
            if self._fallback_llm is None:
                raise
            logger.warning(
                "Strong LLM failed for synthesis — routing to fallback | "
                "error=%s | fallback=%s",
                type(exc).__name__, self._fallback_llm.provider_name,
            )
            active_llm = self._fallback_llm
            response = await self._fallback_llm.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=_SYNTHESIS_MAX_TOKENS,
            )

        answer = response.text.strip()
        return answer, response.prompt_tokens, response.completion_tokens, active_llm.model_name


def _compute_confidence(results: list[SubQueryResult]) -> ConfidenceScore:
    """Compute aggregate confidence as avg of successful sub-queries scaled by success rate."""
    successful = [r for r in results if r.success and r.chunks]

    if not successful:
        return ConfidenceScore(value=0.0, method="agent")

    avg_confidence = sum(r.confidence for r in successful) / len(successful)
    success_rate = len(successful) / len(results)

    return ConfidenceScore(
        value=round(min(avg_confidence * success_rate, 1.0), 4),
        method="agent",
    )
