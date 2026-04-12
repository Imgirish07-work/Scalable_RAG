"""
Agent orchestrator — coordinates the full agent decomposition flow.

Design:
    Mediator pattern that wires together: QueryPlanner → ParallelRetriever
    → ResultVerifier → AnswerSynthesizer. The orchestrator owns timing,
    error aggregation, and the AgentResponse assembly — each component
    does exactly one job.

Chain of Responsibility:
    RAGPipeline.query() → AgentOrchestrator.execute() (when should_decompose=True)
    → QueryPlanner → ParallelRetriever → ResultVerifier → AnswerSynthesizer
    → AgentResponse → RAGPipeline (converts via to_rag_response())

Dependencies:
    agents.planner.query_planner, agents.retriever.parallel_retriever,
    agents.synthesizer.answer_synthesizer, agents.verifier.result_verifier,
    llm.contracts.base_llm
"""

# stdlib
import time
from typing import Optional

# internal
from agents.exceptions.agent_exceptions import (
    AgentError,
    AgentPlanningError,
    AgentRetrievalError,
    AgentSynthesisError,
)
from agents.models.agent_response import AgentResponse, SubQueryResult
from agents.planner.query_planner import QueryPlanner
from agents.retriever.parallel_retriever import ParallelRetriever, PipelineProtocol
from agents.synthesizer.answer_synthesizer import AnswerSynthesizer
from agents.verifier.result_verifier import ResultVerifier
from llm.contracts.base_llm import BaseLLM
from rag.models.rag_request import RAGRequest
from rag.models.rag_response import ConfidenceScore, RAGTimings
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Coordinates the full agent decomposition and synthesis flow.

    Lifecycle:
        1. Planner decomposes query into sub-queries
        2. Parallel retriever executes sub-queries via pipeline
        3. Verifier checks result quality
        4. Synthesizer combines results into final answer

    Attributes:
        _planner: Query decomposition planner.
        _retriever: Parallel sub-query executor.
        _verifier: Result quality checker.
        _synthesizer: Answer combiner.
    """

    def __init__(
        self,
        llm: BaseLLM,
        pipeline: PipelineProtocol,
        collections: dict[str, str],
        max_concurrent: int = 4,
        use_llm_verification: bool = False,
    ) -> None:
        """Initialize AgentOrchestrator.

        Builds all internal components from shared dependencies.

        Args:
            llm: LLM provider shared across planner, verifier, synthesizer.
            pipeline: Pipeline instance for sub-query execution.
            collections: Dict of collection_name → description.
            max_concurrent: Max concurrent sub-query executions.
            use_llm_verification: Whether to use LLM-based result verification.
        """
        self._planner = QueryPlanner(llm=llm, collections=collections)
        self._retriever = ParallelRetriever(
            pipeline=pipeline,
            max_concurrent=max_concurrent,
        )
        self._verifier = ResultVerifier(
            llm=llm,
            use_llm=use_llm_verification,
        )
        self._synthesizer = AnswerSynthesizer(llm=llm)
        self._llm = llm

    async def execute(self, request: RAGRequest) -> AgentResponse:
        """Execute the full agent flow for a complex query.

        Args:
            request: The original RAGRequest from the pipeline.

        Returns:
            AgentResponse with synthesized answer and sub-query details.

        Raises:
            AgentPlanningError: If query decomposition fails.
            AgentRetrievalError: If all sub-queries fail.
            AgentSynthesisError: If answer synthesis fails.
        """
        total_start = time.perf_counter()
        query = request.query
        request_id = request.request_id

        logger.info(
            "Agent executing query, request_id=%s, query='%s'",
            request_id, query[:100],
        )

        # Step 1 — decompose query into sub-queries.
        plan_start = time.perf_counter()
        plan = await self._planner.plan(query)
        plan_ms = (time.perf_counter() - plan_start) * 1000

        logger.info(
            "Agent plan: %d sub-queries, parallel=%s, plan_ms=%.1f",
            len(plan.sub_queries), plan.parallel_safe, plan_ms,
        )

        # Step 2 — execute sub-queries in parallel or sequentially.
        retrieval_start = time.perf_counter()
        sub_results = await self._retriever.execute(
            plan=plan,
            parent_request_id=request_id,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Raise immediately if every sub-query failed — nothing to verify or synthesize.
        any_success = any(r.success for r in sub_results)
        if not any_success:
            raise AgentRetrievalError(
                message="All sub-queries failed",
                details={
                    "request_id": request_id,
                    "sub_query_count": len(sub_results),
                    "failures": [r.failure_reason for r in sub_results],
                },
            )

        # Step 3 — verify result quality before synthesis.
        verify_start = time.perf_counter()
        verified_results = await self._verifier.verify(sub_results)
        verify_ms = (time.perf_counter() - verify_start) * 1000

        # Re-check after verification — some results may have been downgraded.
        any_verified = any(r.success for r in verified_results)
        if not any_verified:
            raise AgentRetrievalError(
                message="All sub-queries failed verification",
                details={
                    "request_id": request_id,
                    "sub_query_count": len(verified_results),
                },
            )

        # Step 4 — synthesize verified results into a final answer.
        synthesis_start = time.perf_counter()
        answer = await self._synthesizer.synthesize(
            query=query,
            sub_results=verified_results,
        )
        synthesis_ms = (time.perf_counter() - synthesis_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # Assemble response with full transparency into the agent's work.
        successful = [r for r in verified_results if r.success]
        failed = [r for r in verified_results if not r.success]

        confidence = _compute_agent_confidence(verified_results)

        response = AgentResponse(
            answer=answer,
            sub_results=verified_results,
            plan_reasoning=plan.reasoning,
            confidence=confidence,
            total_sub_queries=len(verified_results),
            successful_sub_queries=len(successful),
            failed_sub_queries=len(failed),
            timings=RAGTimings(
                retrieval_ms=round(retrieval_ms, 1),
                ranking_ms=round(verify_ms, 1),
                generation_ms=round(plan_ms + synthesis_ms, 1),
                total_ms=round(total_ms, 1),
            ),
            request_id=request_id,
            model_name=self._llm.model_name,
            prompt_tokens=_sum_tokens(verified_results, "prompt"),
            completion_tokens=_sum_tokens(verified_results, "completion"),
        )

        logger.info(
            "Agent complete: request_id=%s total_ms=%.1f "
            "sub_queries=%d/%d succeeded confidence=%.3f",
            request_id, total_ms,
            len(successful), len(verified_results),
            confidence.value,
        )

        return response


# Module-level pure functions


def _compute_agent_confidence(results: list[SubQueryResult]) -> ConfidenceScore:
    """Compute aggregate confidence across sub-query results.

    Uses weighted average of successful sub-query confidences,
    penalized by the failure rate.

    Args:
        results: All sub-query results (verified).

    Returns:
        ConfidenceScore with method="agent".
    """
    successful = [r for r in results if r.success]

    if not successful:
        return ConfidenceScore(value=0.0, method="agent")

    # Average confidence of successful results, then penalize by failure rate.
    avg_confidence = sum(r.confidence for r in successful) / len(successful)
    success_rate = len(successful) / len(results)
    penalized = avg_confidence * success_rate

    return ConfidenceScore(
        value=round(min(penalized, 1.0), 4),
        method="agent",
    )


def _sum_tokens(results: list[SubQueryResult], token_type: str) -> int:
    """Sum prompt or completion tokens across all sub-query results.

    Args:
        results: Sub-query results (success or failure).
        token_type: "prompt" or "completion".

    Returns:
        Total token count across all sub-queries.
    """
    if token_type == "prompt":
        return sum(r.prompt_tokens for r in results)
    if token_type == "completion":
        return sum(r.completion_tokens for r in results)
    return 0
