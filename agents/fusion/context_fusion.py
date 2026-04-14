"""
ContextFusion — merges sub-query chunk results into a single coherent context.

Design:
    Three steps executed in order:
        1. Slot reservation — guarantee 1 best chunk per non-empty sub-query
           so no sub-topic is silenced by MMR's diversity pressure.
        2. MMR on remainder — applies ContextRanker over non-reserved chunks
           for diversity; deduplication by chunk_id prevents repeats.
        3. Token budget — ContextAssembler trims the merged set to fit the
           configured token limit.

    Output is a structured string with [Sub-query N: ...] labels so the
    synthesis LLM knows which chunks address which sub-topic. Failed
    sub-queries appear as explicit gap markers in the structured context.

Chain of Responsibility:
    AgentOrchestrator → ContextFusion.fuse()
    → (structured_context_str, used_chunks) returned to orchestrator
    → orchestrator calls strong_llm.chat() with structured_context_str.

Dependencies:
    rag.context.context_ranker, rag.context.context_assembler,
    agents.models.agent_response, rag.models.rag_response
"""

from agents.models.agent_response import SubQueryResult
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SEPARATOR = "\n\n---\n\n"


class ContextFusion:
    """Merges chunks from multiple sub-queries into one token-bounded context.

    Guarantees at least one chunk per sub-topic (slot reservation) before
    MMR diversification and token budget enforcement.

    Attributes:
        _ranker: ContextRanker for MMR diversification of the remainder pool.
        _assembler: ContextAssembler for token-bounded context assembly.
    """

    def __init__(
        self,
        ranker: ContextRanker,
        assembler: ContextAssembler,
    ) -> None:
        """Initialize ContextFusion.

        Args:
            ranker: Shared ContextRanker for MMR diversification.
            assembler: ContextAssembler for token budget enforcement.
        """
        self._ranker = ranker
        self._assembler = assembler

    async def fuse(
        self,
        sub_results: list[SubQueryResult],
        query: str,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Merge sub-query chunks into a token-bounded structured context.

        Args:
            sub_results: All sub-query results (strong, weak-after-rewrite, failed).
            query: Original user query used for MMR relevance scoring.

        Returns:
            Tuple of (structured_context_str, used_chunks).
            structured_context_str has [Sub-query N: ...] labels per sub-topic.
            used_chunks is the deduplicated list of chunks that fit the token budget.
        """
        successful = [r for r in sub_results if r.success and r.chunks]
        failed = [r for r in sub_results if not r.success or not r.chunks]

        if not successful:
            logger.warning("Context fusion | no successful sub-queries | empty context")
            return "", []

        # Step 1: reserve the best-scored chunk from each successful sub-query.
        # Sorted descending by reranker_score (fallback to relevance_score).
        reserved: list[tuple[SubQueryResult, RetrievedChunk]] = []
        remainder: list[RetrievedChunk] = []

        for result in successful:
            sorted_chunks = sorted(
                result.chunks,
                key=lambda c: c.reranker_score if c.reranker_score is not None else c.relevance_score,
                reverse=True,
            )
            reserved.append((result, sorted_chunks[0]))
            remainder.extend(sorted_chunks[1:])

        # Step 2: deduplicate remainder against reserved and within itself.
        seen_ids = {chunk.chunk_id for _, chunk in reserved}
        unique_remainder: list[RetrievedChunk] = []
        for chunk in remainder:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_remainder.append(chunk)

        # Step 3: apply MMR to remainder for diversity.
        diverse_remainder = (
            await self._ranker.rank(unique_remainder, query)
            if unique_remainder else []
        )

        # Reserved chunks lead — they guarantee sub-topic coverage.
        all_chunks = [chunk for _, chunk in reserved] + diverse_remainder

        # Step 4: enforce token budget via assembler.
        _, used_chunks, tokens_used = await self._assembler.assemble(all_chunks)

        # Step 5: build structured context string with sub-query labels.
        structured_context = _build_structured_context(reserved, used_chunks, failed)

        logger.info(
            "Context fusion complete | sub_queries=%d | reserved=%d | "
            "remainder=%d | used_chunks=%d | tokens=%d | gaps=%d",
            len(successful), len(reserved), len(diverse_remainder),
            len(used_chunks), tokens_used, len(failed),
        )

        return structured_context, used_chunks


def _build_structured_context(
    reserved: list[tuple[SubQueryResult, RetrievedChunk]],
    used_chunks: list[RetrievedChunk],
    failed: list[SubQueryResult],
) -> str:
    """Build a context string grouped by sub-query label.

    Labels inform the synthesis LLM which chunks cover which sub-topic.
    Failed sub-queries appear as explicit gap markers.

    Args:
        reserved: (sub_result, best_chunk) pairs — one per successful sub-query.
        used_chunks: All chunks that passed the token budget.
        failed: Sub-queries with no usable chunks.

    Returns:
        Structured context string with [Sub-query N: ...] section headers.
    """
    used_ids = {c.chunk_id for c in used_chunks}
    sections: list[str] = []

    for i, (result, _) in enumerate(reserved, 1):
        label = f"[Sub-query {i}: {result.query}]"
        sub_chunks = [c for c in result.chunks if c.chunk_id in used_ids]

        if sub_chunks:
            chunk_texts = _CHUNK_SEPARATOR.join(c.content for c in sub_chunks)
            sections.append(f"{label}\n{chunk_texts}")
        else:
            sections.append(f"{label}\n[Limited information available for this aspect]")

    for result in failed:
        sections.append(
            f"[Sub-query: {result.query}]\n[No information found — gap acknowledged]"
        )

    return "\n\n".join(sections)
