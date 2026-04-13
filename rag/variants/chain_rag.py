"""
ChainRAG (CoRAG) — multi-hop retrieval that follows document reference chains.

Design:
    Overrides retrieve() to perform iterative retrieval. Per-hop flow:
        1. Retrieve chunks for the current query (original or follow-up).
        2. Merge + deduplicate against chunks from prior hops.
        3. Rerank with cross-encoder (if enabled); cap to top_k to prevent
           token explosion across hops.
        4. Gate 1 (free): avg cross-encoder score ≥ threshold → context is
           already sufficient, mark complete and skip the LLM call entirely.
        5. Gate 2 (one LLM call): combined draft + completeness evaluation
           in a single prompt — returns draft text, is_complete, reasoning,
           and a targeted follow-up query when incomplete.
        6. Validate the follow-up before the next hop: checks length, string
           identity, word overlap, and cross-encoder semantic similarity.

    LLM call reduction vs. the previous design:
        Before: 2 calls/hop (draft + completeness) × max_hops = up to 6 calls.
        After:  1 call/hop (combined)              × max_hops = up to 3 calls.
                0 calls when Gate 1 fires on hop 1.

    All downstream steps (rank, assemble_context, generate, cache) are
    inherited unchanged from BaseRAG.

Chain of Responsibility:
    Created by RAGFactory → BaseRAG.query() calls retrieve()
    → ChainRAG.retrieve() → _single_hop_retrieve() per hop
    → _rerank_and_filter() → Gate 1 (score threshold)
    → _evaluate_combined() (Gate 2, single LLM call)
    → returns accumulated deduplicated list[RetrievedChunk].

Dependencies:
    rag.base_rag (BaseRAG)
    rag.prompts.rag_prompt_templates (build_chain_combined_prompt)
    rag.exceptions.rag_exceptions (RAGRetrievalError)
    llm.exceptions.llm_exceptions (LLMRateLimitError)
    config.settings (settings)
"""

# stdlib
import asyncio
import hashlib
import json
import re
import time
from typing import Optional

# internal
from config.settings import settings
from llm.contracts.base_llm import BaseLLM
from llm.exceptions.llm_exceptions import LLMRateLimitError
from rag.base_rag import BaseRAG
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.exceptions.rag_exceptions import RAGRetrievalError
from rag.models.rag_request import MetadataFilter, RAGRequest
from rag.models.rag_response import ConfidenceScore, RetrievedChunk
from rag.prompts.rag_prompt_templates import build_chain_combined_prompt
from rag.retrieval.base_retriever import BaseRetriever
from utils.logger import get_logger

logger = get_logger(__name__)

# Combined call token budget = draft + completeness in one response
_COMBINED_MAX_TOKENS = (
    settings.CHAIN_RAG_DRAFT_MAX_TOKENS + settings.CHAIN_RAG_COMPLETENESS_MAX_TOKENS
)

# Reject follow-up queries far longer than the original (likely LLM explanation)
_MAX_FOLLOW_UP_LENGTH_RATIO = 5

# Fallback when combined response cannot be parsed — treat as complete to
# avoid an infinite loop on malformed LLM output.
_FALLBACK_COMBINED = {
    "draft": "",
    "is_complete": True,
    "reasoning": "Response parsing failed — treating as complete",
    "follow_up_query": "",
}


class ChainRAG(BaseRAG):
    """CoRAG variant that resolves multi-document retrieval dependency chains.

    When a query requires information spread across documents connected by
    references (e.g., policy → regulation → appendix), ChainRAG retrieves
    iteratively until the full context is assembled or max_hops is reached.

    Overrides:
        retrieve() — multi-hop retrieval with cross-encoder gating and a
            single combined LLM call per hop (draft + completeness).
        _compute_confidence() — uses chain evaluation with an incomplete-chain
            penalty; prefers reranker_score over dense relevance_score.
        _get_low_confidence_flag() — True if max hops were exhausted without
            completion.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache=None,
        ranker: Optional[ContextRanker] = None,
        assembler: Optional[ContextAssembler] = None,
        fallback_llm: Optional[BaseLLM] = None,
    ) -> None:
        super().__init__(
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
            fallback_llm=fallback_llm,
        )
        # True when the chain resolves within max_hops; False when truncated
        self._chain_completed: bool = False
        # Avg relevance across all hops — used for confidence scoring
        self._chain_avg_relevance: float = 0.0

    @property
    def variant_name(self) -> str:
        return "chain"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[list[MetadataFilter]] = None,
        request: Optional[RAGRequest] = None,
    ) -> list[RetrievedChunk]:
        """Multi-hop retrieval with cross-encoder gating and combined LLM evaluation.

        Args:
            query: The user's original query.
            top_k: Number of chunks to retrieve per hop.
            filters: Optional metadata filters applied to every hop.
            request: Optional RAGRequest for per-request config overrides.

        Returns:
            Accumulated, deduplicated, reranked chunks from all hops.

        Raises:
            RAGRetrievalError: If the initial retrieval (hop 1) fails entirely.
        """
        # Per-query state must be reset — instance is reused across queries.
        self._chain_completed = False
        self._chain_avg_relevance = 0.0

        max_hops = self._resolve_max_hops(request)
        accumulated_chunks: list[RetrievedChunk] = []
        current_query = query

        logger.info(
            "CoRAG starting | max_hops=%d | reranker=%s | query='%s'",
            max_hops,
            "enabled" if self._get_reranker() else "disabled",
            query[:100],
        )

        for hop in range(1, max_hops + 1):
            hop_start = time.perf_counter()

            hop_chunks = await self._single_hop_retrieve(current_query, top_k, filters)

            if not hop_chunks and hop == 1:
                logger.warning("CoRAG hop 1 returned no chunks | aborting chain")
                return []

            # Merge new chunks into accumulated set, deduplicating by chunk_id
            accumulated_chunks = _merge_chunks(accumulated_chunks, hop_chunks)

            # Rerank merged set — filters noise, caps size, prevents token explosion
            accumulated_chunks = await self._rerank_and_filter(
                accumulated_chunks, query, top_k,
            )

            hop_ms = (time.perf_counter() - hop_start) * 1000
            logger.info(
                "CoRAG hop %d/%d | retrieved=%d | accumulated=%d | elapsed_ms=%.1f",
                hop, max_hops, len(hop_chunks), len(accumulated_chunks), hop_ms,
            )

            # Gate 1 (free): cross-encoder avg score above threshold — no LLM needed
            if self._check_gate1_sufficient(accumulated_chunks):
                logger.info(
                    "CoRAG Gate 1 passed at hop %d | avg_score≥%.2f | 0 LLM calls",
                    hop, settings.CHAIN_RAG_RELEVANCE_THRESHOLD,
                )
                self._chain_completed = True
                break

            # Gate 2: single combined LLM call — draft + completeness + follow-up
            combined = await self._evaluate_combined(query, accumulated_chunks)

            if combined["is_complete"]:
                logger.info(
                    "CoRAG Gate 2 complete at hop %d | reasoning='%s'",
                    hop, combined["reasoning"],
                )
                self._chain_completed = True
                break

            # Validate follow-up before using it for the next retrieval hop
            follow_up = combined.get("follow_up_query", "")
            if not await self._is_valid_follow_up(follow_up, query):
                logger.warning(
                    "CoRAG hop %d | invalid or duplicate follow-up | stopping chain",
                    hop,
                )
                break

            logger.info(
                "CoRAG hop %d incomplete | follow_up='%s'",
                hop, follow_up[:100],
            )
            current_query = follow_up

        else:
            # for-else: loop ran to completion without breaking — hops exhausted
            logger.warning(
                "CoRAG exhausted max_hops=%d without completing chain", max_hops,
            )
            self._chain_completed = False

        # Prefer reranker_score for confidence (more accurate than dense scores)
        reranker = self._get_reranker()
        if reranker:
            scores = [
                c.reranker_score for c in accumulated_chunks
                if c.reranker_score is not None and c.reranker_score > 0
            ]
        else:
            scores = [c.relevance_score for c in accumulated_chunks if c.relevance_score > 0]
        self._chain_avg_relevance = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "CoRAG finished | completed=%s | total_chunks=%d | avg_relevance=%.3f",
            self._chain_completed, len(accumulated_chunks), self._chain_avg_relevance,
        )
        return accumulated_chunks

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute confidence from chain completion status and chunk scores.

        Prefers reranker_score (cross-encoder, more accurate) over relevance_score
        (dense cosine). Applies a 20% penalty when the chain did not complete
        within max_hops.
        """
        used = [c for c in chunks if c.used_in_context]
        if not used:
            return ConfidenceScore(value=0.0, method="chain_eval")

        reranker = self._get_reranker()
        if reranker:
            scores = [
                c.reranker_score for c in used
                if c.reranker_score is not None and c.reranker_score > 0
            ]
        else:
            scores = [c.relevance_score for c in used if c.relevance_score > 0]

        avg = sum(scores) / len(scores) if scores else 0.0

        # Penalise incomplete chains — answer may be missing referenced content
        if not self._chain_completed:
            avg *= 0.8

        return ConfidenceScore(value=round(min(avg, 1.0), 4), method="chain_eval")

    def _get_low_confidence_flag(self) -> bool:
        """Return True when the chain did not complete within max_hops."""
        return not self._chain_completed

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_reranker(self):
        """Return the CrossEncoderReranker instance if loaded, else None."""
        if self._ranker and getattr(self._ranker, "_reranker", None):
            return self._ranker._reranker
        return None

    async def _rerank_and_filter(
        self,
        chunks: list[RetrievedChunk],
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank merged chunks with cross-encoder; fall back to score-based filter.

        Runs blocking ONNX/PyTorch inference in a thread pool to keep the
        async event loop responsive. Without a reranker, returns top_k chunks
        sorted by dense relevance score.
        """
        if not chunks:
            return chunks

        reranker = self._get_reranker()
        if reranker:
            # Blocking model inference — run off the event loop
            return await asyncio.to_thread(reranker.rerank, query, chunks, top_k)

        # Fallback: simple score-based cap
        return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)[:top_k]

    def _check_gate1_sufficient(self, chunks: list[RetrievedChunk]) -> bool:
        """Return True if accumulated chunks are already good enough to skip the LLM.

        Compares avg score against CHAIN_RAG_RELEVANCE_THRESHOLD. Uses
        reranker_score when available (more accurate); falls back to dense
        relevance_score.
        """
        if not chunks:
            return False

        reranker = self._get_reranker()
        if reranker:
            scores = [
                c.reranker_score for c in chunks
                if c.reranker_score is not None and c.reranker_score > 0
            ]
        else:
            scores = [c.relevance_score for c in chunks if c.relevance_score > 0]

        if not scores:
            return False

        return (sum(scores) / len(scores)) >= settings.CHAIN_RAG_RELEVANCE_THRESHOLD

    async def _single_hop_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[list[MetadataFilter]],
    ) -> list[RetrievedChunk]:
        return await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    async def _evaluate_combined(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> dict:
        """Single LLM call: generate draft + evaluate completeness + get follow-up.

        On rate-limit, waits CHAIN_RAG_RATE_LIMIT_RETRY_WAIT seconds and
        retries once before falling back to treating the answer as complete.

        Returns:
            Dict with keys: draft (str), is_complete (bool), reasoning (str),
            follow_up_query (str).
        """
        context = _build_draft_context(chunks)
        system_prompt, user_prompt = build_chain_combined_prompt(context, query)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            try:
                response = await self._llm.chat(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=_COMBINED_MAX_TOKENS,
                )
                return _parse_combined_response(response.text)

            except LLMRateLimitError:
                if attempt == 0:
                    wait = settings.CHAIN_RAG_RATE_LIMIT_RETRY_WAIT
                    logger.warning(
                        "CoRAG combined eval rate-limited | "
                        "waiting %.1fs then retrying | attempt=%d",
                        wait, attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue
                # Second attempt also rate-limited — stop the chain gracefully
                logger.warning(
                    "CoRAG combined eval rate-limited after retry | "
                    "treating as complete | stopping chain"
                )
                return _FALLBACK_COMBINED.copy()

            except Exception:
                logger.exception("CoRAG combined eval failed")
                return _FALLBACK_COMBINED.copy()

        return _FALLBACK_COMBINED.copy()  # satisfies type checker

    async def _is_valid_follow_up(
        self,
        follow_up: str,
        original_query: str,
    ) -> bool:
        """Validate a follow-up query before using it for the next hop.

        Checks in order from cheapest to most expensive:
            1. Empty string.
            2. Suspiciously long (likely LLM explanation, not a query).
            3. Identical string to original (infinite loop guard).
            4. High word overlap ≥ 75% (near-duplicate without exact match).
            5. High cross-encoder similarity ≥ threshold (semantic duplicate).
        """
        if not follow_up or not follow_up.strip():
            return False

        if len(follow_up) > len(original_query) * _MAX_FOLLOW_UP_LENGTH_RATIO:
            logger.warning(
                "CoRAG follow-up too long | %d chars vs original %d chars",
                len(follow_up), len(original_query),
            )
            return False

        if follow_up.strip().lower() == original_query.strip().lower():
            logger.warning("CoRAG follow-up identical to original query")
            return False

        # Word overlap: >75% shared words → not a meaningful refinement
        original_words = set(original_query.lower().split())
        followup_words = set(follow_up.lower().split())
        if original_words:
            overlap = len(original_words & followup_words) / len(original_words)
            if overlap > 0.75:
                logger.warning(
                    "CoRAG follow-up word overlap too high | overlap=%.2f", overlap,
                )
                return False

        # Cross-encoder semantic similarity — catches rephrased duplicates
        reranker = self._get_reranker()
        if reranker:
            scores = await asyncio.to_thread(
                reranker._score_pairs, [[original_query, follow_up]],
            )
            if scores and scores[0] >= settings.CHAIN_RAG_FOLLOWUP_SIMILARITY_THRESHOLD:
                logger.warning(
                    "CoRAG follow-up semantically similar to original | "
                    "cross_encoder_score=%.3f | threshold=%.2f",
                    scores[0], settings.CHAIN_RAG_FOLLOWUP_SIMILARITY_THRESHOLD,
                )
                return False

        return True

    def _resolve_max_hops(self, request: Optional[RAGRequest]) -> int:
        if request and request.config:
            return request.config.resolve_max_hops()
        return settings.CHAIN_RAG_MAX_HOPS


# ── Module-level pure functions — no class state, easily unit-testable ─────────

def _merge_chunks(
    existing: list[RetrievedChunk],
    new_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge new chunks into an existing list with deduplication.

    When a chunk_id appears in both lists, the copy with the higher
    relevance score is kept. Chunk identity falls back to SHA-256 of
    content when chunk_id is absent.

    Args:
        existing: Previously accumulated chunks from earlier hops.
        new_chunks: Freshly retrieved chunks from the current hop.

    Returns:
        Merged, deduplicated chunk list.
    """
    # Index by chunk_id for O(1) lookup; SHA-256 content hash as fallback —
    # never use id() since object identity changes across hops.
    chunk_map: dict[str, RetrievedChunk] = {}
    for chunk in existing:
        key = chunk.chunk_id or hashlib.sha256(chunk.content.encode()).hexdigest()
        chunk_map[key] = chunk

    for chunk in new_chunks:
        key = chunk.chunk_id or hashlib.sha256(chunk.content.encode()).hexdigest()
        if key in chunk_map:
            # Keep the higher-scoring copy of a duplicate
            if chunk.relevance_score > chunk_map[key].relevance_score:
                chunk_map[key] = chunk
        else:
            chunk_map[key] = chunk

    return list(chunk_map.values())


def _build_draft_context(chunks: list[RetrievedChunk]) -> str:
    """Build a context string from the top-scoring chunks for the combined LLM call.

    Sorts by reranker_score (if set) or relevance_score descending and caps at
    CHAIN_RAG_DRAFT_CONTEXT_MAX_CHUNKS to prevent token explosion.

    Args:
        chunks: Accumulated chunks to include in the context.

    Returns:
        Numbered, newline-separated chunk content string.
    """
    cap = settings.CHAIN_RAG_DRAFT_CONTEXT_MAX_CHUNKS

    # Sort by best available score — reranker_score is more accurate when set
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (c.reranker_score or 0.0) if c.reranker_score is not None
                      else c.relevance_score,
        reverse=True,
    )[:cap]

    return "\n\n".join(f"[{i}] {c.content}" for i, c in enumerate(sorted_chunks, 1))


def _parse_combined_response(text: str) -> dict:
    """Parse the combined draft + completeness LLM response.

    Two-stage parsing: direct JSON, then markdown-fence-stripped JSON.
    Falls back to _FALLBACK_COMBINED on all parsing failures.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict with guaranteed keys: draft, is_complete, reasoning, follow_up_query.
    """
    cleaned = text.strip()

    # Stage 1 — direct JSON parse
    result = _try_json_parse(cleaned)
    if result is not None:
        return _validate_combined_dict(result)

    # Stage 2 — strip markdown fences and retry
    stripped = re.sub(r"^```(?:json)?\s*", "", cleaned)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    result = _try_json_parse(stripped)
    if result is not None:
        return _validate_combined_dict(result)

    logger.warning("CoRAG combined parse failed | raw='%s'", text[:200])
    return _FALLBACK_COMBINED.copy()


def _validate_combined_dict(raw: dict) -> dict:
    """Validate and normalise a parsed combined response dict.

    Coerces unexpected types gracefully. Clears follow_up_query when
    is_complete is True to prevent stale follow-ups from triggering a hop.

    Args:
        raw: Parsed JSON dict from the LLM response.

    Returns:
        Validated dict with guaranteed draft, is_complete, reasoning, follow_up_query.
    """
    draft = str(raw.get("draft", ""))

    is_complete = raw.get("is_complete")
    if not isinstance(is_complete, bool):
        is_complete = str(is_complete).lower() in ("true", "1", "yes")

    reasoning = str(raw.get("reasoning", ""))
    follow_up = str(raw.get("follow_up_query", ""))

    # Clear stale follow-up when the answer is already complete
    if is_complete:
        follow_up = ""

    return {
        "draft": draft,
        "is_complete": is_complete,
        "reasoning": reasoning,
        "follow_up_query": follow_up,
    }


def _try_json_parse(text: str) -> Optional[dict]:
    """Attempt to parse text as JSON, returning None on any failure."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None
