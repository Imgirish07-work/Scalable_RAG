"""Merges sub-query chunks into a token-bounded structured context with [Sub-query N: ...] labels."""

from agents.models.agent_response import SubQueryResult
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SEPARATOR = "\n\n---\n\n"


class ContextFusion:
    """Merges chunks from multiple sub-queries into one token-bounded context."""

    def __init__(
        self,
        ranker: ContextRanker,
        assembler: ContextAssembler,
    ) -> None:
        self._ranker = ranker
        self._assembler = assembler

    async def fuse(
        self,
        sub_results: list[SubQueryResult],
        query: str,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Merge sub-query chunks into a token-bounded structured context."""
        successful = [r for r in sub_results if r.success and r.chunks]
        failed = [r for r in sub_results if not r.success or not r.chunks]

        if not successful:
            logger.warning("Context fusion | no successful sub-queries | empty context")
            return "", []

        # step 1: reserve best chunk per sub-query so mmr cannot silence a sub-topic
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

        # step 2: dedupe remainder
        seen_ids = {chunk.chunk_id for _, chunk in reserved}
        unique_remainder: list[RetrievedChunk] = []
        for chunk in remainder:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_remainder.append(chunk)

        # step 3: mmr on remainder
        diverse_remainder = (
            await self._ranker.rank(unique_remainder, query)
            if unique_remainder else []
        )

        # reserved chunks lead to guarantee sub-topic coverage
        all_chunks = [chunk for _, chunk in reserved] + diverse_remainder

        # step 4: enforce token budget
        _, used_chunks, tokens_used = await self._assembler.assemble(all_chunks)

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
    """Build a context string grouped by sub-query label."""
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
