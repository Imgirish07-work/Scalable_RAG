"""
Assembles retrieved chunks into a prompt-ready, token-bounded context string.

Design:
    Takes a ranked list of RetrievedChunks and greedily adds them in rank
    order (highest relevance first) until the token budget is exhausted.
    Whole chunks only — never truncates mid-chunk, because a clause cut in
    half is worse than omitting it entirely. Chunks included in the final
    context are flagged with used_in_context=True on a new (frozen)
    RetrievedChunk instance. Token counting uses BaseLLM.count_tokens(),
    which is async because Gemini's implementation makes an API call.

Chain of Responsibility:
    Called by BaseRAG.assemble_context() after ContextRanker.rank()
    → passes (context_str, updated_chunks, tokens_used) back to BaseRAG
    → context_str is forwarded to BaseRAG._generate().

Dependencies:
    rag.models.rag_response (RetrievedChunk)
    rag.exceptions.rag_exceptions (RAGContextError)
"""

from rag.models.rag_response import RetrievedChunk
from rag.exceptions.rag_exceptions import RAGContextError
from utils.logger import get_logger

logger = get_logger(__name__)

# Separator rendered between chunks in the assembled context string
_CHUNK_SEPARATOR = "\n\n---\n\n"

# Conservative token overhead per chunk for separators, labels, and numbering
_OVERHEAD_PER_CHUNK = 20

# Minimum useful context budget — below this, generation quality degrades severely
_MIN_CONTEXT_TOKENS = 20


class ContextAssembler:
    """Assembles retrieved chunks into a token-bounded context string.

    Takes a ranked list of RetrievedChunks and produces a formatted
    context string that fits within the configured token budget.
    Returns both the context string and updated chunks with the
    used_in_context flag set for source attribution.

    Attributes:
        _llm: BaseLLM instance used for token counting.
        _max_tokens: Maximum tokens allowed for the assembled context.
        _include_source_labels: Whether to prepend source info to each chunk.
    """

    def __init__(
        self,
        llm: object,
        max_tokens: int = 3072,
        include_source_labels: bool = True,
    ) -> None:
        """Initialize the context assembler.

        Args:
            llm: BaseLLM instance for token counting. Must implement
                async count_tokens(text) -> int.
            max_tokens: Maximum token budget for the assembled context.
                Must be >= _MIN_CONTEXT_TOKENS (20).
            include_source_labels: Whether to prepend [Source N: file.pdf]
                labels to each chunk. Helps the LLM cite sources, but
                costs a few extra tokens per chunk.

        Raises:
            ValueError: If max_tokens is below the minimum threshold.
        """
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
        """Assemble chunks into a token-bounded context string.

        Adds chunks in rank order (caller must pre-sort by relevance).
        Stops when adding the next chunk would exceed the token budget.
        Includes whole chunks only — never truncates mid-chunk.

        Args:
            chunks: List of RetrievedChunk ordered by relevance (highest
                first). Typically the output of ContextRanker.rank().

        Returns:
            Tuple of:
                - context_str: Formatted context string for the LLM prompt.
                - updated_chunks: New chunk list with used_in_context flags set.
                    Included chunks have used_in_context=True; excluded
                    chunks retain used_in_context=False.
                - tokens_used: Actual token count of the assembled context.

        Raises:
            RAGContextError: If the chunks list is empty, or if no chunk
                fits within the token budget.
        """
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
            # Format this chunk with optional source label
            formatted = self._format_chunk(chunk, index=i + 1)

            # Count tokens for this chunk plus the separator that precedes it
            chunk_tokens = await self._llm.count_tokens(formatted)
            separator_tokens = (
                await self._llm.count_tokens(_CHUNK_SEPARATOR)
                if context_parts
                else 0
            )
            total_addition = chunk_tokens + separator_tokens

            # Stop adding when the next chunk would exceed the budget
            if tokens_used + total_addition > self._max_tokens:
                logger.info(
                    "Token budget reached | included=%d | excluded=%d | "
                    "tokens_used=%d | budget=%d",
                    included_count,
                    len(chunks) - included_count,
                    tokens_used,
                    self._max_tokens,
                )
                # Remaining chunks are excluded — add them with used_in_context=False
                updated_chunks.append(chunk)
                continue

            # Append chunk to context
            if context_parts:
                context_parts.append(_CHUNK_SEPARATOR)
            context_parts.append(formatted)
            tokens_used += total_addition
            included_count += 1

            # Create new chunk with used_in_context=True.
            # RetrievedChunk is frozen, so we rebuild with the updated flag.
            # vector and reranker_score are internal pipeline fields (exclude=True
            # on the model) — they must be forwarded explicitly so that:
            #   - MMR in a second-pass rank() can still use pre-fetched vectors
            #   - _compute_confidence() in base_rag can use reranker scores
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
        """Format a single chunk for inclusion in the context string.

        Optionally prepends a source label with file name and section.
        The 1-based index lets the LLM reference sources by number.

        Args:
            chunk: RetrievedChunk to format.
            index: 1-based position number used in the source label.

        Returns:
            Formatted chunk string ready for concatenation into context.
        """
        parts = []

        if self._include_source_labels:
            label = self._build_source_label(chunk, index)
            parts.append(label)

        parts.append(chunk.content)

        return "\n".join(parts)

    def _build_source_label(self, chunk: RetrievedChunk, index: int) -> str:
        """Build a source attribution label for a chunk.

        Format: [Source 1: report.pdf | Section: Introduction | Page: 3]
        Section and page are omitted when not present in the chunk metadata.

        Args:
            chunk: RetrievedChunk with source metadata.
            index: 1-based source number.

        Returns:
            Formatted source label string.
        """
        label_parts = [f"[Source {index}: {chunk.source_file}"]

        if chunk.section_heading:
            label_parts.append(f"Section: {chunk.section_heading}")

        if chunk.page_number is not None:
            label_parts.append(f"Page: {chunk.page_number}")

        return " | ".join(label_parts) + "]"
