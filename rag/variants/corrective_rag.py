"""
CorrectiveRAG — retrieval quality evaluation with query rewrite retry.

Design:
    - Overrides retrieve() to add a post-retrieval relevance evaluation
      branch. This is the ONLY variant that questions whether retrieval
      actually worked before sending results to the LLM.
    - After initial retrieval, each chunk is scored for relevance via
      a lightweight LLM call. The average score determines the branch:
        - score >= PASS threshold (0.7)  → accept chunks, proceed
        - score >= RETRY threshold (0.4) → rewrite query, retry once
        - score < RETRY threshold        → flag as low_confidence
    - Maximum 1 retry to prevent infinite loops and unbounded cost.
    - Overrides _compute_confidence() to use the relevance evaluation
      scores instead of raw retrieval cosine similarity — more accurate
      because the LLM evaluates semantic relevance, not just embedding
      distance.

Cost analysis:
    - Best case: 1 retrieval + N eval LLM calls + 1 generation = N+2 calls
    - Worst case (retry): 2 retrievals + 2×N eval calls + 1 rewrite + 1 gen = 2N+4 calls
    - Where N = number of chunks evaluated (default: top_k from config)
    - Optimization: evaluate only top 3 chunks instead of all top_k to
      reduce eval cost. If top 3 are irrelevant, the rest likely are too.

When to use:
    - High-stakes queries (financial, medical, legal, compliance)
    - Domains where retrieval often returns similar-but-wrong chunks
    - When confidence in the answer matters more than latency

Pipeline flow:
    1. Cache check          → inherited from BaseRAG
    2. pre_process()        → inherited from BaseRAG
    3. retrieve()           → ★ OVERRIDDEN: retrieve + evaluate + maybe retry
    4. rank()               → inherited (MMR)
    5. assemble_context()   → inherited (token-bounded)
    6. generate()           → inherited (grounded LLM call)
    7. Cache write          → inherited from BaseRAG

Integration:
    - RELEVANCE_EVAL prompts from rag/prompts/rag_prompt_templates.py
    - QUERY_REWRITE prompts from rag/prompts/rag_prompt_templates.py
    - Settings: CRAG_RELEVANCE_THRESHOLD_PASS, CRAG_RELEVANCE_THRESHOLD_RETRY,
      CRAG_MAX_RETRIES from config/settings.py
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

# Default thresholds — overridden by settings if available
_DEFAULT_PASS_THRESHOLD = 0.7
_DEFAULT_RETRY_THRESHOLD = 0.4
_DEFAULT_MAX_RETRIES = 1

# Number of top chunks to evaluate for relevance
# Evaluating all top_k is expensive. Top 3 is a good proxy —
# if the top 3 are irrelevant, the rest almost certainly are too.
_EVAL_CHUNK_COUNT = 3


class CorrectiveRAG(BaseRAG):
    """Corrective RAG — retrieval evaluation with query rewrite retry.

    Adds a relevance evaluation branch after retrieval. Catches the
    most dangerous RAG failure mode: retrieval returns confident-looking
    but irrelevant chunks, and the LLM generates a fluent, well-structured,
    completely wrong answer.

    Attributes:
        _pass_threshold: Minimum average relevance to accept chunks.
        _retry_threshold: Minimum average relevance to attempt retry.
            Below this, chunks are flagged as low_confidence.
        _max_retries: Maximum query rewrite + retry attempts.
        _eval_chunk_count: Number of top chunks to evaluate.
        _last_avg_relevance: Stores the relevance score from the most
            recent evaluation for confidence computation.
        _is_low_confidence: Flag set when relevance is below retry
            threshold after all attempts exhausted.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache: object | None = None,
        ranker: ContextRanker | None = None,
        assembler: ContextAssembler | None = None,
        pass_threshold: float | None = None,
        retry_threshold: float | None = None,
        max_retries: int | None = None,
        eval_chunk_count: int | None = None,
    ) -> None:
        """Initialize CorrectiveRAG.

        Args:
            retriever: Vector store retriever (dense or hybrid).
            llm: LLM provider for generation, evaluation, and rewriting.
            cache: Optional CacheManager.
            ranker: Optional ContextRanker.
            assembler: Optional ContextAssembler.
            pass_threshold: Minimum relevance to accept chunks without retry.
                Falls back to settings.CRAG_RELEVANCE_THRESHOLD_PASS.
            retry_threshold: Minimum relevance to attempt query rewrite.
                Below this, result is flagged low_confidence.
                Falls back to settings.CRAG_RELEVANCE_THRESHOLD_RETRY.
            max_retries: Maximum retry attempts. Falls back to
                settings.CRAG_MAX_RETRIES.
            eval_chunk_count: Number of top chunks to evaluate.
                Default 3.
        """
        super().__init__(
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
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

        # Per-query state — reset on each retrieve() call
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

    # ================================================================
    # Override: retrieve() — the core corrective logic
    # ================================================================

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with relevance evaluation and optional retry.

        Flow:
            1. Initial retrieval via the injected retriever.
            2. Evaluate relevance of top N chunks via LLM.
            3. Branch on average relevance score:
               - >= pass_threshold → accept chunks
               - >= retry_threshold → rewrite query, retry once
               - < retry_threshold → flag low_confidence, return what we have

        Args:
            query: Processed query string.
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of RetrievedChunk ordered by relevance.

        Raises:
            RAGRetrievalError: If initial retrieval fails.
        """
        # Reset per-query state
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

        # Step 2: Evaluate relevance
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
            # Attempt query rewrite + retry
            return await self._retry_with_rewrite(
                original_query=query,
                top_k=top_k,
                filters=filters,
                attempt=1,
            )

        # Below retry threshold — flag as low confidence
        logger.warning(
            "CorrectiveRAG: relevance below retry threshold | "
            "score=%.3f < %.2f | flagging low_confidence",
            avg_relevance,
            self._retry_threshold,
        )
        self._is_low_confidence = True
        return chunks

    # ================================================================
    # Override: confidence uses evaluation scores, not retrieval scores
    # ================================================================

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute confidence from relevance evaluation scores.

        Overrides BaseRAG._compute_confidence() to use the LLM-based
        relevance evaluation score instead of raw cosine similarity.
        This is more accurate because the LLM evaluates semantic
        relevance, not just embedding distance.

        Args:
            chunks: Updated chunks with used_in_context flags.
            method: Ignored — CorrectiveRAG always uses 'corrective_eval'.

        Returns:
            ConfidenceScore with evaluation-based value.
        """
        return ConfidenceScore(
            value=round(self._last_avg_relevance, 4),
            method="corrective_eval",
        )

    # ================================================================
    # Private: core corrective methods
    # ================================================================

    async def _do_retrieval(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None,
    ) -> list[RetrievedChunk]:
        """Execute a single retrieval call.

        Thin wrapper for logging and error context.

        Args:
            query: Query string.
            top_k: Maximum chunks.
            filters: Optional metadata filters.

        Returns:
            List of RetrievedChunk.

        Raises:
            RAGRetrievalError: If retriever fails.
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
        """Evaluate relevance of top chunks via LLM.

        Sends each of the top N chunks to the LLM with a relevance
        scoring prompt. Returns the average score.

        Only evaluates the top _eval_chunk_count chunks. If the top 3
        are irrelevant, evaluating the rest is wasted tokens.

        Args:
            query: The user's query.
            chunks: Retrieved chunks (ordered by retrieval score).

        Returns:
            Average relevance score (0.0-1.0). Returns 0.0 if all
            evaluations fail.
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
        """Evaluate a single chunk's relevance via LLM.

        Sends the query + chunk to the LLM with the relevance eval
        prompt. Parses the JSON response for the score.

        Args:
            query: The user's query.
            chunk: Single RetrievedChunk to evaluate.
            index: Chunk position for logging.

        Returns:
            Relevance score (0.0-1.0), or None if evaluation fails.
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
        """Parse relevance score from LLM evaluation response.

        Expects JSON: {"relevance": 0.85, "reason": "..."}
        Falls back to extracting any float from the response.

        This is a sync CPU-only function (Rule 2).

        Args:
            text: Raw LLM response text.

        Returns:
            Relevance score (0.0-1.0), or None if parsing fails.
        """
        if not text or not text.strip():
            return None

        cleaned = text.strip()

        # Try JSON parsing first
        try:
            # Handle markdown code fences
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

        # Fallback: extract any float from the text
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
        """Rewrite the query and retry retrieval.

        Uses the LLM to rewrite the original query for better
        retrieval, then re-retrieves and re-evaluates. Maximum
        one retry to bound cost and prevent infinite loops.

        Args:
            original_query: The query that produced low-relevance results.
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters.
            attempt: Current retry attempt number (1-based).

        Returns:
            List of RetrievedChunk from retry, or original chunks
            if retry doesn't improve results.
        """
        if attempt > self._max_retries:
            logger.warning(
                "CorrectiveRAG: max retries exhausted | attempts=%d | "
                "flagging low_confidence",
                attempt,
            )
            self._is_low_confidence = True
            # Return whatever we had from the last attempt
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

        # Step 2: Retry retrieval with rewritten query
        retry_chunks = await self._do_retrieval(rewritten_query, top_k, filters)

        if not retry_chunks:
            logger.warning(
                "CorrectiveRAG: retry retrieval returned empty results"
            )
            self._is_low_confidence = True
            return retry_chunks

        # Step 3: Re-evaluate relevance
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

        # Retry didn't reach pass threshold
        if retry_relevance >= self._retry_threshold:
            logger.info(
                "CorrectiveRAG: retry improved but below pass | "
                "score=%.3f | accepting with moderate confidence",
                retry_relevance,
            )
            return retry_chunks

        # Retry still below retry threshold
        logger.warning(
            "CorrectiveRAG: retry still below threshold | "
            "score=%.3f < %.2f | flagging low_confidence",
            retry_relevance,
            self._retry_threshold,
        )
        self._is_low_confidence = True
        return retry_chunks

    async def _rewrite_query(self, query: str) -> str | None:
        """Rewrite a query to improve retrieval.

        Uses the LLM with the query rewrite prompt template.
        Returns the rewritten query, or None if rewriting fails.

        Args:
            query: Original query that produced low-relevance results.

        Returns:
            Rewritten query string, or None on failure.
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

            # Sanity check: rewritten query shouldn't be absurdly long
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