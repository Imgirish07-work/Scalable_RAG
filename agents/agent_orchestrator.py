"""
Agent orchestrator — coordinates the full complex query processing flow.

Design:
    Mediator that wires together: QueryPlanner → ChunkRetriever
    → ChunkQualityGate → (optional) weak sub-query rewrite → ContextFusion
    → single LLM generation. Only 2 LLM calls for a clean execution;
    up to 1 extra call per weak sub-query (fast model).

    Flow:
        1. Plan   — strong LLM decomposes the query into 2-3 sub-queries.
        2. Fetch  — ChunkRetriever runs sub-queries in parallel (no LLM).
        3. Gate   — ChunkQualityGate classifies results (deterministic).
        4. Rewrite — weak sub-queries rewritten once with fast LLM, re-fetched.
        5. Fuse   — ContextFusion merges chunks with slot reservation + MMR.
        6. Answer — strong LLM generates the final answer from fused context.

Chain of Responsibility:
    RAGPipeline._execute_query() → AgentOrchestrator.execute()
    → AgentResponse → RAGPipeline (converts via to_rag_response()).

Dependencies:
    agents.planner.query_planner, agents.retriever.chunk_retriever,
    agents.quality.chunk_quality_gate, agents.fusion.context_fusion,
    agents.prompts.agent_prompt_templates, llm.contracts.base_llm
"""

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

# Max tokens for the rewrite LLM call — it only needs to return a short query string.
_REWRITE_MAX_TOKENS = 120

# Max tokens for the final synthesis call.
_SYNTHESIS_MAX_TOKENS = 2048


class AgentOrchestrator:
    """Coordinates the complex query decomposition and synthesis flow.

    Uses two LLM instances:
        strong_llm — query planning and final answer generation.
        fast_llm   — weak sub-query rewrite (cheap, bounded to 1 call per
                     weak sub-query; falls back to strong_llm if not provided).

    Attributes:
        _planner: QueryPlanner backed by the strong LLM.
        _chunk_retriever: ChunkRetriever for parallel retrieval-only sub-queries.
        _quality_gate: ChunkQualityGate for deterministic quality evaluation.
        _context_fusion: ContextFusion for slot reservation + MMR + token budget.
        _strong_llm: Strong model for planning and synthesis.
        _fast_llm: Fast model for weak sub-query rewriting.
    """

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
        """Initialize AgentOrchestrator with all required dependencies.

        Args:
            strong_llm: LLM for query planning and final answer generation.
            fast_llm: LLM for weak sub-query rewriting (cheap model).
            store_factory: Async callable: collection_name → QdrantStore.
            collections: Dict of collection_name → description for the planner.
            embeddings_fn: Callable returning the embedding model (for MMR).
            max_concurrent: Max parallel sub-query retrieval executions.
        """
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
        """Execute the full complex query flow.

        Args:
            request: The original RAGRequest from the pipeline.

        Returns:
            AgentResponse with the synthesized answer and sub-query metadata.

        Raises:
            AgentRetrievalError: If all sub-queries fail retrieval or return no chunks.
        """
        total_start = time.perf_counter()
        query = request.query
        request_id = request.request_id
        total_prompt_tokens = 0
        total_completion_tokens = 0

        logger.info(
            "Agent executing | request_id=%s | query='%s'",
            request_id, query[:100],
        )

        # Step 1: plan — decompose into sub-queries (strong LLM, 1 call).
        plan_start = time.perf_counter()
        plan = await self._planner.plan(query)
        plan_ms = (time.perf_counter() - plan_start) * 1000

        logger.info(
            "Agent plan complete | sub_queries=%d | parallel=%s | plan_ms=%.1f",
            len(plan.sub_queries), plan.parallel_safe, plan_ms,
        )

        # Step 2: retrieve all sub-queries in parallel (no LLM).
        retrieval_start = time.perf_counter()
        sub_results = await self._chunk_retriever.retrieve_all(
            sub_queries=plan.sub_queries,
            parent_request_id=request_id,
            user_id=request.user_id,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 3: quality gate — classify strong / weak / failed (deterministic).
        sub_results = self._quality_gate.evaluate(sub_results)

        # Step 4: rewrite weak sub-queries with fast LLM, then re-retrieve.
        rewrite_start = time.perf_counter()
        sub_results, rewrite_tokens = await self._rewrite_and_refetch_weak(
            sub_results=sub_results,
            parent_request_id=request_id,
            user_id=request.user_id,
        )
        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000
        total_prompt_tokens += rewrite_tokens[0]
        total_completion_tokens += rewrite_tokens[1]

        # Fail fast if every sub-query has no usable chunks.
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

        # Step 5: fuse context — slot reservation + MMR + token budget.
        fusion_start = time.perf_counter()
        structured_context, used_chunks = await self._context_fusion.fuse(
            sub_results=sub_results,
            query=query,
        )
        fusion_ms = (time.perf_counter() - fusion_start) * 1000

        # Step 6: generate final answer (strong LLM, 1 call).
        # Both LLMs failing is gracefully degraded — retrieval work is preserved.
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
    ) -> tuple[list[SubQueryResult], tuple[int, int]]:
        """Rewrite weak sub-queries with the fast LLM then re-retrieve once.

        Weak sub-queries run in parallel. Failed sub-queries are skipped
        entirely — no retry for zero-chunk results.

        Args:
            sub_results: All sub-query results after quality gate evaluation.
            parent_request_id: Parent request ID for log tracing.

        Returns:
            Tuple of (updated_results, (prompt_tokens, completion_tokens)).
        """
        weak_results = [r for r in sub_results if r.is_weak]

        if not weak_results:
            return sub_results, (0, 0)

        logger.info(
            "Rewriting %d weak sub-queries | request_id=%s",
            len(weak_results), parent_request_id,
        )

        # Rewrite all weak sub-queries in parallel.
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

        # Re-retrieve all rewritten sub-queries in parallel.
        if rewritten_sub_queries:
            refetch_tasks = [
                self._chunk_retriever.retrieve_one(sq, parent_request_id, user_id)
                for _, sq in rewritten_sub_queries
            ]
            refetch_results = await asyncio.gather(*refetch_tasks, return_exceptions=True)

            # Build a lookup of original id → new result for replacement.
            replacements: dict[str, SubQueryResult] = {}
            for (original, _), new_result in zip(rewritten_sub_queries, refetch_results):
                if isinstance(new_result, Exception):
                    logger.warning(
                        "Re-retrieval failed after rewrite | id=%s | error=%s",
                        original.sub_query_id, new_result,
                    )
                else:
                    replacements[original.sub_query_id] = new_result

            # Replace original weak results with re-retrieved results where available.
            sub_results = [
                replacements.get(r.sub_query_id, r) if r.is_weak else r
                for r in sub_results
            ]

        return sub_results, (total_prompt, total_completion)

    async def _rewrite_one(
        self,
        result: SubQueryResult,
    ) -> tuple[str, int, int]:
        """Rewrite a single weak sub-query using the fast LLM.

        Uses the best available chunk as context for the rewrite even though
        its relevance score is low — any signal is better than none.

        Args:
            result: A weak SubQueryResult with at least one chunk.

        Returns:
            Tuple of (rewritten_query, prompt_tokens, completion_tokens).
        """
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
        """Generate the final answer from fused context (strong LLM, 1 call).

        Args:
            query: The original user query.
            structured_context: Fused context string from ContextFusion.

        Returns:
            Tuple of (answer_text, prompt_tokens, completion_tokens).
        """
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
            # All pool models exhausted (429s) or hard provider error.
            # Route to fallback LLM (Gemini) to avoid crashing the pipeline.
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
    """Compute aggregate confidence across sub-query retrieval results.

    Average confidence of successful sub-queries, penalized by failure rate.

    Args:
        results: All sub-query results after quality gate and rewrite.

    Returns:
        ConfidenceScore with method="agent".
    """
    successful = [r for r in results if r.success and r.chunks]

    if not successful:
        return ConfidenceScore(value=0.0, method="agent")

    avg_confidence = sum(r.confidence for r in successful) / len(successful)
    success_rate = len(successful) / len(results)

    return ConfidenceScore(
        value=round(min(avg_confidence * success_rate, 1.0), 4),
        method="agent",
    )
