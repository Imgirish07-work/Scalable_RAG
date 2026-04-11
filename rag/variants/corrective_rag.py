"""
CorrectiveRAG — retrieval quality evaluation with query rewrite and retry.

Design:
    Overrides retrieve() to add a post-retrieval relevance evaluation branch.
    This is the only variant that verifies retrieval quality before sending
    results to the LLM. After initial retrieval, each of the top N chunks is
    scored for relevance via a lightweight LLM call. The average score drives
    three branches:
        - score >= pass_threshold (0.7)  → accept, proceed to rank
        - score >= retry_threshold (0.4) → rewrite query, retry once
        - score < retry_threshold        → flag low_confidence, proceed anyway

    Maximum one retry prevents infinite loops and unbounded cost. Overrides
    _compute_confidence() to use LLM-evaluated relevance scores instead of
    raw cosine similarity, which is a more accurate signal.

    Cost (N = evaluated chunks, default N=3):
        Best case:  1 retrieval + N eval LLM calls + 1 generation = N+2 calls
        Worst case: 2 retrievals + 2N eval calls + 1 rewrite + 1 gen = 2N+4 calls

Chain of Responsibility:
    Created by RAGFactory → BaseRAG.query() calls retrieve()
    → CorrectiveRAG.retrieve() → _evaluate_relevance() → maybe _retry_with_rewrite()
    → returns list[RetrievedChunk] to the rank step.

Dependencies:
    rag.base_rag (BaseRAG)
    rag.prompts.rag_prompt_templates (build_relevance_eval_prompt, build_query_rewrite_prompt)
    rag.exceptions.rag_exceptions (RAGRetrievalError)
    config.settings (settings)
"""

import json
import time

from llm.contracts.base_llm import BaseLLM
from rag.base_rag import BaseRAG
from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk, ConfidenceScore
from rag.retrieval.base_retriever import BaseRetriever
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.prompts.rag_prompt_templates import (
    build_relevance_eval_prompt,
    build_query_rewrite_prompt,
)
from rag.exceptions.rag_exceptions import RAGRetrievalError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Default thresholds — overridden by settings when available
_DEFAULT_PASS_THRESHOLD = 0.7
_DEFAULT_RETRY_THRESHOLD = 0.4
_DEFAULT_MAX_RETRIES = 1

# Evaluate only the top N chunks — if the top 3 are irrelevant, the rest are too.
# This caps eval cost without meaningfully reducing detection accuracy.
_EVAL_CHUNK_COUNT = 3


class CorrectiveRAG(BaseRAG):
    """Corrective RAG — retrieval quality evaluation with query rewrite retry.

    Adds a relevance evaluation branch after retrieval to catch the most
    dangerous RAG failure mode: retrieval returns confident-looking but
    irrelevant chunks, causing the LLM to generate a fluent but wrong answer.

    Attributes:
        _pass_threshold: Minimum average relevance to accept chunks without retry.
        _retry_threshold: Minimum average relevance to attempt a query rewrite.
            Below this, chunks are accepted but flagged as low_confidence.
        _max_retries: Maximum query rewrite and retry attempts.
        _eval_chunk_count: Number of top chunks to evaluate per retrieval.
        _last_avg_relevance: Relevance score from the most recent evaluation.
            Used by _compute_confidence() to report evaluation-based confidence.
        _is_low_confidence: Set to True when relevance falls below the retry
            threshold after all attempts. Read by BaseRAG._get_low_confidence_flag().
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache: object | None = None,
        ranker: ContextRanker | None = None,
        assembler: ContextAssembler | None = None,
        fallback_llm: BaseLLM | None = None,
        pass_threshold: float | None = None,
        retry_threshold: float | None = None,
        max_retries: int | None = None,
        eval_chunk_count: int | None = None,
    ) -> None:
        """Initialize CorrectiveRAG.

        Args:
            retriever: Vector store retriever (dense or hybrid).
            llm: LLM provider for generation, evaluation, and query rewriting.
            cache: Optional CacheManager.
            ranker: Optional ContextRanker.
            assembler: Optional ContextAssembler.
            fallback_llm: Optional secondary LLM used when the primary fails.
            pass_threshold: Minimum relevance to accept without retry.
                Falls back to settings.CRAG_RELEVANCE_THRESHOLD_PASS.
            retry_threshold: Minimum relevance to attempt a query rewrite.
                Below this, chunks are accepted but flagged low_confidence.
                Falls back to settings.CRAG_RELEVANCE_THRESHOLD_RETRY.
            max_retries: Maximum retry attempts.
                Falls back to settings.CRAG_MAX_RETRIES.
            eval_chunk_count: Number of top chunks to evaluate per retrieval.
                Default 3.
        """
        super().__init__(
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
            fallback_llm=fallback_llm,
        )

        self._pass_threshold = (
            pass_threshold
            if pass_threshold is not None
            else getattr(settings, "CRAG_RELEVANCE_THRESHOLD_PASS", _DEFAULT_PASS_THRESHOLD)
        )
        self._retry_threshold = (
            retry_threshold
            if retry_threshold is not None
            else getattr(settings, "CRAG_RELEVANCE_THRESHOLD_RETRY", _DEFAULT_RETRY_THRESHOLD)
        )
        self._max_retries = (
            max_retries
            if max_retries is not None
            else getattr(settings, "CRAG_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        )
        self._eval_chunk_count = eval_chunk_count or _EVAL_CHUNK_COUNT

        # Per-query state — reset at the start of each retrieve() call
        self._last_avg_relevance = 0.0
        self._is_low_confidence = False

        logger.info(
            "CorrectiveRAG initialized | pass_threshold=%.2f | "
            "retry_threshold=%.2f | max_retries=%d | eval_chunks=%d",
            self._pass_threshold,
            self._retry_threshold,
            self._max_retries,
            self._eval_chunk_count,
        )

    @property
    def variant_name(self) -> str:
        """Return the variant identifier.

        Returns:
            The string 'corrective'.
        """
        return "corrective"

    # Override: retrieve() — core corrective logic

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
        request=None,
    ) -> list[RetrievedChunk]:
        """Retrieve with relevance evaluation and optional query rewrite retry.

        Flow:
            1. Initial retrieval via the injected retriever.
            2. Evaluate relevance of the top N chunks via LLM.
            3. Branch on average relevance score:
               - >= pass_threshold  → accept, return chunks
               - >= retry_threshold → rewrite query, retry once
               - < retry_threshold  → flag low_confidence, return what we have

        Args:
            query: Processed query string (output of pre_process).
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters.
            request: Unused. Accepted for interface compatibility with ChainRAG.

        Returns:
            List of RetrievedChunk ordered by relevance.

        Raises:
            RAGRetrievalError: If initial retrieval fails.
        """
        # Reset per-query state before each retrieve call
        self._last_avg_relevance = 0.0
        self._is_low_confidence = False

        # Step 1: Initial retrieval
        chunks = await self._do_retrieval(query, top_k, filters)

        if not chunks:
            logger.warning(
                "CorrectiveRAG: initial retrieval returned empty results"
            )
            self._is_low_confidence = True
            return chunks

        # Step 2: Evaluate relevance of top N chunks
        avg_relevance = await self._evaluate_relevance(query, chunks)
        self._last_avg_relevance = avg_relevance

        # Step 3: Branch on score
        if avg_relevance >= self._pass_threshold:
            logger.info(
                "CorrectiveRAG: relevance PASS | score=%.3f >= %.2f",
                avg_relevance,
                self._pass_threshold,
            )
            return chunks

        if avg_relevance >= self._retry_threshold:
            # Score is marginal — attempt query rewrite and retry
            return await self._retry_with_rewrite(
                original_query=query,
                top_k=top_k,
                filters=filters,
                attempt=1,
            )

        # Below retry threshold — flag as low confidence and proceed
        logger.warning(
            "CorrectiveRAG: relevance below retry threshold | "
            "score=%.3f < %.2f | flagging low_confidence",
            avg_relevance,
            self._retry_threshold,
        )
        self._is_low_confidence = True
        return chunks

    # Override: use evaluation scores for confidence, not retrieval cosine similarity

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute confidence from LLM-evaluated relevance scores.

        Overrides BaseRAG._compute_confidence() to use the LLM-evaluated
        relevance score instead of raw cosine similarity. LLM evaluation
        is more accurate because it assesses semantic relevance directly,
        not just embedding distance.

        Args:
            chunks: Updated chunks with used_in_context flags.
            method: Ignored — CorrectiveRAG always reports 'corrective_eval'.

        Returns:
            ConfidenceScore with the evaluation-based relevance value.
        """
        return ConfidenceScore(
            value=round(self._last_avg_relevance, 4),
            method="corrective_eval",
        )

    # Private: core corrective methods

    async def _do_retrieval(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None,
    ) -> list[RetrievedChunk]:
        """Execute a single retrieval call via the injected retriever.

        Thin wrapper around the retriever for logging and consistent error
        context. Used for both the initial retrieval and retry retrievals.

        Args:
            query: Query string for this retrieval attempt.
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of RetrievedChunk.

        Raises:
            RAGRetrievalError: If the retriever fails.
        """
        logger.info(
            "CorrectiveRAG retrieving | query_len=%d | top_k=%d",
            len(query),
            top_k,
        )
        return await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    async def _evaluate_relevance(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> float:
        """Evaluate relevance of the top N chunks via LLM scoring.

        Sends each of the top _eval_chunk_count chunks to the LLM with
        the relevance evaluation prompt and averages the returned scores.
        Evaluating only the top N chunks caps cost — if the top 3 are
        irrelevant, the rest almost certainly are too.

        Args:
            query: The user's query.
            chunks: Retrieved chunks ordered by retrieval score.

        Returns:
            Average relevance score (0.0–1.0). Returns 0.0 if all
            individual evaluations fail.
        """
        eval_chunks = chunks[:self._eval_chunk_count]
        scores = []

        start = time.perf_counter()

        for i, chunk in enumerate(eval_chunks):
            score = await self._evaluate_single_chunk(query, chunk, index=i)
            if score is not None:
                scores.append(score)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not scores:
            logger.warning(
                "CorrectiveRAG: all chunk evaluations failed | "
                "chunks_evaluated=%d | latency=%.1f ms",
                len(eval_chunks),
                elapsed_ms,
            )
            return 0.0

        avg = sum(scores) / len(scores)

        logger.info(
            "CorrectiveRAG: relevance evaluated | avg=%.3f | "
            "scores=%s | chunks=%d | latency=%.1f ms",
            avg,
            [round(s, 3) for s in scores],
            len(eval_chunks),
            elapsed_ms,
        )

        return avg

    async def _evaluate_single_chunk(
        self,
        query: str,
        chunk: RetrievedChunk,
        index: int,
    ) -> float | None:
        """Evaluate a single chunk's relevance via LLM scoring.

        Sends the query and chunk content to the LLM with the relevance
        evaluation prompt. Parses the JSON response for the relevance score.

        Args:
            query: The user's query.
            chunk: Single RetrievedChunk to evaluate.
            index: 0-based chunk position for log messages.

        Returns:
            Relevance score in range 0.0–1.0, or None if evaluation fails.
        """
        system_prompt, user_prompt = build_relevance_eval_prompt(
            query=query,
            document=chunk.content,
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._llm.chat(
                messages,
                temperature=0.0,
                max_tokens=100,
            )

            score = self._parse_relevance_score(response.text)

            if score is not None:
                logger.debug(
                    "Chunk %d relevance=%.3f | source=%s",
                    index,
                    score,
                    chunk.source_file,
                )

            return score

        except Exception as exc:
            logger.warning(
                "Chunk %d relevance eval failed | source=%s | error=%s",
                index,
                chunk.source_file,
                str(exc),
            )
            return None

    def _parse_relevance_score(self, text: str) -> float | None:
        """Parse a relevance score from the LLM evaluation response.

        Expects JSON of the form {"relevance": 0.85, "reason": "..."}.
        Falls back to scanning tokens for any float in range [0, 1]
        when JSON parsing fails (handles malformed or markdown-fenced output).

        This is a synchronous CPU-only function — no I/O, no await needed.

        Args:
            text: Raw LLM response text.

        Returns:
            Relevance score in range 0.0–1.0, or None if parsing fails.
        """
        if not text or not text.strip():
            return None

        cleaned = text.strip()

        # Primary: JSON parsing
        try:
            # Strip markdown code fences before parsing
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            if isinstance(data, dict) and "relevance" in data:
                score = float(data["relevance"])
                return max(0.0, min(1.0, score))

        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass

        # Fallback: scan tokens for any float in [0.0, 1.0]
        try:
            for token in cleaned.replace(",", " ").split():
                token = token.strip("()[]{}\"'")
                try:
                    val = float(token)
                    if 0.0 <= val <= 1.0:
                        return val
                except ValueError:
                    continue
        except Exception:
            pass

        logger.warning(
            "Could not parse relevance score from LLM response | text=%s",
            cleaned[:100],
        )
        return None

    async def _retry_with_rewrite(
        self,
        original_query: str,
        top_k: int,
        filters: list[MetadataFilter] | None,
        attempt: int,
    ) -> list[RetrievedChunk]:
        """Rewrite the query and retry retrieval once.

        Uses the LLM to produce an improved query, re-retrieves, and
        re-evaluates. Bounded by _max_retries to prevent infinite loops
        and unbounded LLM cost.

        Args:
            original_query: The query that produced marginal-relevance results.
            top_k: Maximum chunks to retrieve on retry.
            filters: Optional metadata filters.
            attempt: Current retry attempt number (1-based).

        Returns:
            List of RetrievedChunk from the retry, or from a fresh retrieval
            of the original query if retries are exhausted or rewrite fails.
        """
        if attempt > self._max_retries:
            logger.warning(
                "CorrectiveRAG: max retries exhausted | attempts=%d | "
                "flagging low_confidence",
                attempt,
            )
            self._is_low_confidence = True
            # Return a fresh retrieval — the original chunks are not held in scope
            return await self._do_retrieval(original_query, top_k, filters)

        # Step 1: Rewrite the query
        rewritten_query = await self._rewrite_query(original_query)

        if not rewritten_query or rewritten_query == original_query:
            logger.warning(
                "CorrectiveRAG: query rewrite produced no change | "
                "flagging low_confidence"
            )
            self._is_low_confidence = True
            return await self._do_retrieval(original_query, top_k, filters)

        logger.info(
            "CorrectiveRAG: query rewritten | attempt=%d | "
            "original=%s | rewritten=%s",
            attempt,
            original_query[:80],
            rewritten_query[:80],
        )

        # Step 2: Retry retrieval with the rewritten query
        retry_chunks = await self._do_retrieval(rewritten_query, top_k, filters)

        if not retry_chunks:
            logger.warning(
                "CorrectiveRAG: retry retrieval returned empty results"
            )
            self._is_low_confidence = True
            return retry_chunks

        # Step 3: Re-evaluate relevance against the original query
        retry_relevance = await self._evaluate_relevance(
            original_query, retry_chunks
        )
        self._last_avg_relevance = retry_relevance

        if retry_relevance >= self._pass_threshold:
            logger.info(
                "CorrectiveRAG: retry PASS | score=%.3f >= %.2f",
                retry_relevance,
                self._pass_threshold,
            )
            return retry_chunks

        if retry_relevance >= self._retry_threshold:
            # Retry improved but didn't fully pass — accept with moderate confidence
            logger.info(
                "CorrectiveRAG: retry improved but below pass | "
                "score=%.3f | accepting with moderate confidence",
                retry_relevance,
            )
            return retry_chunks

        # Retry still below retry threshold — flag low confidence
        logger.warning(
            "CorrectiveRAG: retry still below threshold | "
            "score=%.3f < %.2f | flagging low_confidence",
            retry_relevance,
            self._retry_threshold,
        )
        self._is_low_confidence = True
        return retry_chunks

    async def _rewrite_query(self, query: str) -> str | None:
        """Rewrite a query to improve retrieval using the LLM.

        Uses the query rewrite prompt template to generate an alternative
        query formulation. Rejects rewrites that are unchanged or suspiciously
        longer than the original — both indicate the LLM failed to rewrite.

        Args:
            query: Original query that produced low-relevance retrieval results.

        Returns:
            Rewritten query string, or None if rewriting fails or produces
            no usable change.
        """
        system_prompt, user_prompt = build_query_rewrite_prompt(query)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._llm.chat(
                messages,
                temperature=0.3,
                max_tokens=200,
            )

            rewritten = response.text.strip()

            if not rewritten:
                logger.warning("Query rewrite returned empty response")
                return None

            # Guard against runaway rewrites — 5x length ratio is a strong signal
            # that the LLM generated an explanation rather than a rewritten query.
            if len(rewritten) > len(query) * 5:
                logger.warning(
                    "Query rewrite suspiciously long | original_len=%d | "
                    "rewritten_len=%d | using original",
                    len(query),
                    len(rewritten),
                )
                return None

            return rewritten

        except Exception as exc:
            logger.warning(
                "Query rewrite failed | error=%s",
                str(exc),
            )
            return None
