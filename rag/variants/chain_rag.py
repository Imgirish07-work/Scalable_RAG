"""CoRAG (Chain-of-RAG) variant — resolves retrieval dependency chains.

Handles queries where initial retrieval reveals references to external
documents, regulations, or sections not present in the first retrieval.
Follows the breadcrumb trail with bounded hops until the answer is
complete or max_hops is reached.

Overrides only retrieve() and _compute_confidence() in BaseRAG.
Everything downstream (rank, assemble_context, generate, cache) is inherited.
"""

# stdlib
import hashlib
import json
import re
import time
from typing import Optional

# internal
from config.settings import settings
from llm.contracts.base_llm import BaseLLM
from rag.base_rag import BaseRAG
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.exceptions.rag_exceptions import RAGRetrievalError
from rag.models.rag_request import MetadataFilter, RAGRequest
from rag.models.rag_response import ConfidenceScore, RetrievedChunk
from rag.prompts.rag_prompt_templates import (
    build_chain_completeness_prompt,
    build_chain_draft_prompt,
)
from rag.retrieval.base_retriever import BaseRetriever
from utils.logger import get_logger

logger = get_logger(__name__)

# draft generation budget — kept low to minimize cost per hop
_DRAFT_MAX_TOKENS = settings.CHAIN_RAG_DRAFT_MAX_TOKENS
_COMPLETENESS_MAX_TOKENS = settings.CHAIN_RAG_COMPLETENESS_MAX_TOKENS

# safety guard — reject follow-up queries that explode in length
_MAX_FOLLOW_UP_LENGTH_RATIO = 5

# fallback completeness result when parsing fails
_FALLBACK_COMPLETENESS = {
    "is_complete": True,
    "reasoning": "Completeness parsing failed — treating as complete",
    "follow_up_query": "",
}


class ChainRAG(BaseRAG):
    """CoRAG variant that follows retrieval dependency chains.

    When a query requires information spread across multiple documents
    connected by references (e.g., policy → regulation → appendix),
    ChainRAG retrieves iteratively until the full context is assembled.

    Overrides:
        retrieve() — multi-hop retrieval with completeness evaluation.
        _compute_confidence() — uses chain evaluation method.
        _get_low_confidence_flag() — True if max hops exhausted without completion.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache=None,
        ranker: Optional[ContextRanker] = None,
        assembler: Optional[ContextAssembler] = None,
    ) -> None:
        """Initialize ChainRAG variant.

        Args:
            retriever: Retriever strategy (dense or hybrid).
            llm: LLM provider for draft generation and completeness evaluation.
            cache: Optional CacheManager instance.
            ranker: Optional ContextRanker. Defaults to MMR in BaseRAG.
            assembler: Optional ContextAssembler. Defaults in BaseRAG.
        """
        super().__init__(
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
        )
        # tracks whether the chain completed within max_hops
        self._chain_completed: bool = False
        # average relevance across all hops for confidence scoring
        self._chain_avg_relevance: float = 0.0

    @property
    def variant_name(self) -> str:
        """Return the variant identifier.

        Returns:
            The string 'chain'.
        """
        return "chain"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[list[MetadataFilter]] = None,
        request: Optional[RAGRequest] = None,
    ) -> list[RetrievedChunk]:
        """Multi-hop retrieval with completeness evaluation.

        Retrieves chunks, generates a draft answer, evaluates completeness,
        and follows up with targeted retrieval if references are unresolved.
        Repeats until complete or max_hops is reached.

        Args:
            query: The user's original query.
            top_k: Number of chunks to retrieve per hop.
            filters: Optional metadata filters applied to every hop.
            request: Optional RAGRequest for accessing config overrides.

        Returns:
            Accumulated, deduplicated chunks from all hops.

        Raises:
            RAGRetrievalError: If the initial retrieval fails entirely.
        """
        max_hops = self._resolve_max_hops(request)
        accumulated_chunks: list[RetrievedChunk] = []
        current_query = query

        logger.info(
            "CoRAG starting chain retrieval, max_hops=%d, query='%s'",
            max_hops,
            query[:100],
        )

        for hop in range(1, max_hops + 1):
            hop_start = time.perf_counter()

            # retrieve for current query
            hop_chunks = await self._single_hop_retrieve(
                current_query, top_k, filters,
            )

            if not hop_chunks and hop == 1:
                # first hop returned nothing — no point continuing
                logger.warning("CoRAG hop 1 returned no chunks, aborting chain")
                self._chain_completed = False
                return []

            # merge with accumulated chunks, deduplicate
            accumulated_chunks = _merge_chunks(accumulated_chunks, hop_chunks)

            hop_ms = (time.perf_counter() - hop_start) * 1000
            logger.info(
                "CoRAG hop %d/%d retrieved %d chunks (%d accumulated) in %.1fms",
                hop, max_hops, len(hop_chunks),
                len(accumulated_chunks), hop_ms,
            )

            # generate draft from all accumulated chunks
            draft = await self._generate_draft(query, accumulated_chunks)
            if not draft:
                logger.warning("CoRAG hop %d draft generation returned empty", hop)
                break

            # evaluate completeness against the ORIGINAL query
            completeness = await self._evaluate_completeness(query, draft)

            if completeness["is_complete"]:
                logger.info(
                    "CoRAG chain complete at hop %d: %s",
                    hop, completeness["reasoning"],
                )
                self._chain_completed = True
                break

            # not complete — prepare follow-up query for next hop
            follow_up = completeness.get("follow_up_query", "")
            if not self._is_valid_follow_up(follow_up, query):
                logger.warning(
                    "CoRAG hop %d produced invalid follow-up, stopping chain",
                    hop,
                )
                break

            logger.info(
                "CoRAG hop %d incomplete, follow-up: '%s'",
                hop, follow_up[:100],
            )
            current_query = follow_up

        else:
            # loop completed without break — max hops exhausted
            logger.warning(
                "CoRAG exhausted max_hops=%d without completing chain", max_hops,
            )
            self._chain_completed = False

        # compute average relevance for confidence scoring
        scores = [c.relevance_score for c in accumulated_chunks if c.relevance_score > 0]
        self._chain_avg_relevance = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "CoRAG chain finished, completed=%s, total_chunks=%d, avg_relevance=%.3f",
            self._chain_completed, len(accumulated_chunks), self._chain_avg_relevance,
        )
        return accumulated_chunks

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute confidence based on chain completion and relevance.

        Uses the average relevance across all hops. Method is "chain_eval"
        to distinguish from retrieval-only or corrective evaluation.

        Args:
            chunks: All retrieved chunks (used_in_context filtered upstream).

        Returns:
            ConfidenceScore with chain_eval method.
        """
        used = [c for c in chunks if c.used_in_context]
        if not used:
            return ConfidenceScore(value=0.0, method="chain_eval")

        scores = [c.relevance_score for c in used if c.relevance_score > 0]
        avg = sum(scores) / len(scores) if scores else 0.0

        # penalize incomplete chains — reduce confidence by 20%
        if not self._chain_completed:
            avg *= 0.8

        return ConfidenceScore(value=round(min(avg, 1.0), 4), method="chain_eval")

    def _get_low_confidence_flag(self) -> bool:
        """Flag low confidence when the chain did not complete.

        Returns:
            True if max hops were exhausted without completion.
        """
        return not self._chain_completed

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    async def _single_hop_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[list[MetadataFilter]],
    ) -> list[RetrievedChunk]:
        """Execute a single retrieval hop via the injected retriever.

        Args:
            query: Query string for this hop.
            top_k: Number of chunks to retrieve.
            filters: Optional metadata filters.

        Returns:
            Retrieved chunks for this hop.

        Raises:
            RAGRetrievalError: If retrieval fails.
        """
        return await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    async def _generate_draft(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Generate a concise draft answer from accumulated chunks.

        Uses low max_tokens to keep cost per hop minimal. Temperature
        fixed at 0.0 for deterministic, factual output.

        Args:
            query: The original user query.
            chunks: All accumulated chunks across hops.

        Returns:
            Draft answer string, or empty string on failure.
        """
        # assemble context from chunks for draft prompt
        context = _build_draft_context(chunks)
        system_prompt, user_prompt = build_chain_draft_prompt(context, query)

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_DRAFT_MAX_TOKENS,
            )
            return response.text.strip()

        except Exception:
            logger.exception("CoRAG draft generation failed")
            return ""

    async def _evaluate_completeness(
        self,
        query: str,
        draft_answer: str,
    ) -> dict:
        """Evaluate whether the draft answer fully resolves the query.

        Expects structured JSON response from LLM. Falls back to
        treating the answer as complete if parsing fails — safe default
        that prevents infinite loops.

        Args:
            query: The original user query.
            draft_answer: Current draft answer to evaluate.

        Returns:
            Dict with keys: is_complete (bool), reasoning (str),
            follow_up_query (str).
        """
        system_prompt, user_prompt = build_chain_completeness_prompt(
            query, draft_answer,
        )

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_COMPLETENESS_MAX_TOKENS,
            )
            return _parse_completeness_response(response.text)

        except Exception:
            logger.exception("CoRAG completeness evaluation failed")
            return _FALLBACK_COMPLETENESS.copy()

    def _is_valid_follow_up(self, follow_up: str, original_query: str) -> bool:
        """Validate a follow-up query before using it for the next hop.

        Guards against empty, excessively long, or identical follow-ups.

        Args:
            follow_up: The proposed follow-up query.
            original_query: The original user query for length comparison.

        Returns:
            True if the follow-up is usable.
        """
        if not follow_up or not follow_up.strip():
            return False

        # reject follow-ups that are suspiciously long
        if len(follow_up) > len(original_query) * _MAX_FOLLOW_UP_LENGTH_RATIO:
            logger.warning(
                "CoRAG follow-up too long: %d chars vs original %d chars",
                len(follow_up), len(original_query),
            )
            return False

        # reject follow-ups identical to the original (would loop)
        if follow_up.strip().lower() == original_query.strip().lower():
            logger.warning("CoRAG follow-up is identical to original query")
            return False

        return True

    def _resolve_max_hops(self, request: Optional[RAGRequest]) -> int:
        """Resolve max_hops from request config or settings default.

        Args:
            request: Optional RAGRequest carrying config overrides.

        Returns:
            Resolved max_hops value.
        """
        if request and request.config:
            return request.config.resolve_max_hops()
        return settings.CHAIN_RAG_MAX_HOPS


# ──────────────────────────────────────────────
# Module-level pure functions (no state, easily testable)
# ──────────────────────────────────────────────

def _merge_chunks(
    existing: list[RetrievedChunk],
    new_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge new chunks into existing list with deduplication.

    When a chunk_id appears in both lists, the one with the higher
    relevance score is kept. New unique chunks are appended at the end.

    Args:
        existing: Previously accumulated chunks.
        new_chunks: Newly retrieved chunks from the current hop.

    Returns:
        Merged, deduplicated chunk list.
    """
    # index existing chunks by chunk_id for O(1) lookup
    # fall back to SHA-256 of content when chunk_id is missing — never use
    # id(chunk) since Python object identity changes across hops for same content
    chunk_map: dict[str, RetrievedChunk] = {}
    for chunk in existing:
        key = chunk.chunk_id or hashlib.sha256(chunk.content.encode()).hexdigest()
        chunk_map[key] = chunk

    for chunk in new_chunks:
        key = chunk.chunk_id or hashlib.sha256(chunk.content.encode()).hexdigest()
        if key in chunk_map:
            # keep the one with the higher relevance score
            if chunk.relevance_score > chunk_map[key].relevance_score:
                chunk_map[key] = chunk
        else:
            chunk_map[key] = chunk

    return list(chunk_map.values())


def _build_draft_context(chunks: list[RetrievedChunk]) -> str:
    """Build a simple context string from chunks for draft generation.

    Uses a lightweight format — no source labels. Draft context is
    disposable; the real formatting happens in ContextAssembler for
    the final generation.

    Args:
        chunks: Chunks to include in context.

    Returns:
        Newline-separated chunk content.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk.content}")
    return "\n\n".join(parts)


def _parse_completeness_response(text: str) -> dict:
    """Parse the LLM's completeness evaluation response.

    Two-stage parsing: try clean JSON first, then strip markdown
    fences and retry. Falls back to treating as complete if all
    parsing fails — prevents infinite loops on malformed output.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict with is_complete, reasoning, follow_up_query keys.
    """
    cleaned = text.strip()

    # stage 1 — try direct JSON parse
    result = _try_json_parse(cleaned)
    if result is not None:
        return _validate_completeness_dict(result)

    # stage 2 — strip markdown fences and retry
    stripped = re.sub(r"^```(?:json)?\s*", "", cleaned)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    result = _try_json_parse(stripped)
    if result is not None:
        return _validate_completeness_dict(result)

    # all parsing failed — safe fallback
    logger.warning("CoRAG completeness parse failed, raw text: '%s'", text[:200])
    return _FALLBACK_COMPLETENESS.copy()


def _try_json_parse(text: str) -> Optional[dict]:
    """Attempt JSON parsing, returning None on failure.

    Args:
        text: String to parse as JSON.

    Returns:
        Parsed dict or None.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _validate_completeness_dict(raw: dict) -> dict:
    """Validate and normalize a parsed completeness response.

    Ensures all required keys exist with correct types. Coerces
    unexpected types gracefully rather than crashing.

    Args:
        raw: Raw parsed JSON dict.

    Returns:
        Validated dict with guaranteed keys and types.
    """
    is_complete = raw.get("is_complete")
    if not isinstance(is_complete, bool):
        # coerce truthy/falsy strings
        is_complete = str(is_complete).lower() in ("true", "1", "yes")

    reasoning = str(raw.get("reasoning", ""))
    follow_up = str(raw.get("follow_up_query", ""))

    # if marked complete, clear any stale follow-up
    if is_complete:
        follow_up = ""

    return {
        "is_complete": is_complete,
        "reasoning": reasoning,
        "follow_up_query": follow_up,
    }