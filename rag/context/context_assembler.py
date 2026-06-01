"""Greedy whole-chunk context assembler with token budget enforcement."""

from rag.models.rag_response import RetrievedChunk
from rag.exceptions.rag_exceptions import RAGContextError
from utils.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SEPARATOR = "\n\n---\n\n"
_OVERHEAD_PER_CHUNK = 20
_MIN_CONTEXT_TOKENS = 20


class ContextAssembler:
    """Assembles retrieved chunks into a token-bounded context string."""

    def __init__(
        self,
        llm: object,
        max_tokens: int = 3072,
        include_source_labels: bool = True,
    ) -> None:
        if max_tokens < _MIN_CONTEXT_TOKENS:
            raise ValueError(
                f"max_tokens ({max_tokens}) is below minimum "
                f"({_MIN_CONTEXT_TOKENS}). Context would be too small "
                f"for useful generation."
            )

        self._llm = llm
        self._max_tokens = max_tokens
        self._include_source_labels = include_source_labels

        logger.info(
            "ContextAssembler initialized | max_tokens=%d | source_labels=%s",
            self._max_tokens,
            self._include_source_labels,
        )

    async def assemble(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[RetrievedChunk], int]:
        """Assemble chunks into a token-bounded context string."""
        if not chunks:
            raise RAGContextError(
                "No chunks provided for context assembly.",
                details={"max_tokens": self._max_tokens},
            )

        context_parts = []
        updated_chunks = []
        tokens_used = 0
        included_count = 0

        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(chunk, index=i + 1)

            chunk_tokens = await self._llm.count_tokens(formatted)
            separator_tokens = (
                await self._llm.count_tokens(_CHUNK_SEPARATOR)
                if context_parts
                else 0
            )
            total_addition = chunk_tokens + separator_tokens

            if tokens_used + total_addition > self._max_tokens:
                logger.info(
                    "Token budget reached | included=%d | excluded=%d | "
                    "tokens_used=%d | budget=%d",
                    included_count,
                    len(chunks) - included_count,
                    tokens_used,
                    self._max_tokens,
                )
                updated_chunks.append(chunk)
                continue

            if context_parts:
                context_parts.append(_CHUNK_SEPARATOR)
            context_parts.append(formatted)
            tokens_used += total_addition
            included_count += 1

            # retrievedchunk is frozen; forward vector and reranker_score explicitly so
            # later mmr can reuse pre-fetched vectors and _compute_confidence can use reranker scores
            updated_chunk = RetrievedChunk(
                content=chunk.content,
                source_file=chunk.source_file,
                chunk_id=chunk.chunk_id,
                relevance_score=chunk.relevance_score,
                section_heading=chunk.section_heading,
                page_number=chunk.page_number,
                content_type=chunk.content_type,
                used_in_context=True,
                metadata=chunk.metadata,
                vector=chunk.vector,
                reranker_score=chunk.reranker_score,
            )
            updated_chunks.append(updated_chunk)

        if included_count == 0:
            raise RAGContextError(
                "No chunks fit within the token budget. Even the highest-ranked "
                "chunk exceeds max_context_tokens.",
                details={
                    "max_tokens": self._max_tokens,
                    "first_chunk_tokens": await self._llm.count_tokens(
                        self._format_chunk(chunks[0], index=1)
                    ),
                    "total_chunks": len(chunks),
                },
            )

        context_str = "".join(context_parts)

        logger.info(
            "Context assembled | chunks_included=%d/%d | tokens=%d/%d",
            included_count,
            len(chunks),
            tokens_used,
            self._max_tokens,
        )

        return context_str, updated_chunks, tokens_used

    def _format_chunk(self, chunk: RetrievedChunk, index: int) -> str:
        """Format a single chunk for inclusion in the context string."""
        parts = []

        if self._include_source_labels:
            label = self._build_source_label(chunk, index)
            parts.append(label)

        parts.append(chunk.content)

        return "\n".join(parts)

    def _build_source_label(self, chunk: RetrievedChunk, index: int) -> str:
        """Build a source attribution label for a chunk."""
        label_parts = [f"[Source {index}: {chunk.source_file}"]

        if chunk.section_heading:
            label_parts.append(f"Section: {chunk.section_heading}")

        if chunk.page_number is not None:
            label_parts.append(f"Page: {chunk.page_number}")

        return " | ".join(label_parts) + "]"
