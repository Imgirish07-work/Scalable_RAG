"""
Context assembler — formats retrieved chunks into a prompt-ready string.

Design:
    - Takes ranked RetrievedChunks and assembles them into a single context
      string that fits within the token budget (max_context_tokens).
    - Chunks are added in rank order (highest relevance first). When the
      token budget is exhausted, remaining chunks are excluded.
    - Chunks that make it into the context get used_in_context=True on
      a new RetrievedChunk instance (originals are frozen/immutable).
    - Token counting uses BaseLLM.count_tokens() — async because Gemini's
      implementation makes an API call. This is an I/O-bound operation
      (Rule 1: async def).

Why not just concatenate and truncate:
    - Truncating mid-chunk destroys context. A legal clause cut in half
      is worse than omitting it entirely. This assembler includes whole
      chunks or excludes them — never partial chunks.
    - Source attribution requires knowing WHICH chunks were used. The
      used_in_context flag enables this.

Integration points:
    - RetrievedChunk from rag/models/rag_response.py
    - BaseLLM.count_tokens() from llm/contracts/base_llm.py
    - rag_prompt_templates.py for context formatting patterns
"""

from rag.models.rag_response import RetrievedChunk
from rag.exceptions.rag_exceptions import RAGContextError
from utils.logger import get_logger

logger = get_logger(__name__)

# Separator between chunks in assembled context
_CHUNK_SEPARATOR = "\n\n---\n\n"

# Overhead tokens for separators, source labels, numbering
# Conservative estimate — actual overhead varies by chunk count
_OVERHEAD_PER_CHUNK = 20

# Minimum useful context — below this, generation quality degrades
_MIN_CONTEXT_TOKENS = 20


class ContextAssembler:
    """Assembles retrieved chunks into a token-bounded context string.

    Takes a ranked list of RetrievedChunks and produces a formatted
    context string that fits within the configured token budget.
    Returns both the context string and updated chunks with the
    used_in_context flag set.

    Attributes:
        _llm: BaseLLM instance for token counting.
        _max_tokens: Maximum tokens allowed for assembled context.
        _include_source_labels: Whether to prepend source info to each chunk.
    """

    def __init__(
        self,
        llm: object,
        max_tokens: int = 3072,
        include_source_labels: bool = True,
    ) -> None:
        """Initialize context assembler.

        Args:
            llm: BaseLLM instance for token counting. Must implement
                async count_tokens(text) -> int.
            max_tokens: Maximum token budget for assembled context.
                Must be >= _MIN_CONTEXT_TOKENS.
            include_source_labels: Whether to prepend [Source: file.pdf]
                labels to each chunk. Helps LLM cite sources but uses
                a few extra tokens per chunk.

        Raises:
            ValueError: If max_tokens is below minimum threshold.
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
        Whole chunks only — never truncates mid-chunk.

        Args:
            chunks: List of RetrievedChunk, ordered by relevance
                (highest first). Typically output of ContextRanker.

        Returns:
            Tuple of:
                - context_str: Formatted context string for the LLM prompt.
                - updated_chunks: New chunk list with used_in_context flags set.
                    Chunks included in context have used_in_context=True.
                    Excluded chunks retain used_in_context=False.
                - tokens_used: Actual token count of the assembled context.

        Raises:
            RAGContextError: If no chunks can fit within the token budget,
                or if the chunks list is empty.
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
            # Format this chunk
            formatted = self._format_chunk(chunk, index=i + 1)

            # Count tokens for this chunk + separator overhead
            chunk_tokens = await self._llm.count_tokens(formatted)
            separator_tokens = (
                await self._llm.count_tokens(_CHUNK_SEPARATOR)
                if context_parts
                else 0
            )
            total_addition = chunk_tokens + separator_tokens

            # Check if adding this chunk would exceed the budget
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

            # Add chunk to context
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
        """Format a single chunk for inclusion in the context.

        Optionally prepends a source label with file name and section.
        Numbering helps the LLM reference specific sources in its answer.

        Args:
            chunk: RetrievedChunk to format.
            index: 1-based position number for source reference.

        Returns:
            Formatted chunk string.
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

        Only includes section and page if they exist in the chunk metadata.

        Args:
            chunk: RetrievedChunk with source metadata.
            index: 1-based source number.

        Returns:
            Source label string.
        """
        label_parts = [f"[Source {index}: {chunk.source_file}"]

        if chunk.section_heading:
            label_parts.append(f"Section: {chunk.section_heading}")

        if chunk.page_number is not None:
            label_parts.append(f"Page: {chunk.page_number}")

        return " | ".join(label_parts) + "]"