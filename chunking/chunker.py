"""
Splits structure-tagged documents into embedding-ready chunks.

Design:
    Single-responsibility class (Chunker) that routes each document page to
    the appropriate splitter based on the structure_type metadata written by
    StructurePreserver. Post-split steps (filter, deduplicate, enrich, prepend
    context) run in a fixed pipeline regardless of splitter used.

Chain of Responsibility:
    Receives List[Document] from StructurePreserver.preserve() →
    produces chunked, enriched List[Document] → passed to
    VectorStore.add_documents() for embedding and upsert.

Dependencies:
    tiktoken, langchain_text_splitters.RecursiveCharacterTextSplitter,
    langchain_core.documents.Document, config.settings,
    utils.helpers.hash_text, utils.logger.
"""

import re
from typing import Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger
from utils.helpers import hash_text

logger = get_logger(__name__)

# Module-level encoder — tiktoken caches internally; defining it here
# makes the shared encoding name explicit and avoids repeated lookups.
_ENCODER = tiktoken.get_encoding("cl100k_base")


class Chunker:
    """Splits cleaned, structure-tagged documents into chunks for embedding.

    Attributes:
        _chunk_size: Target chunk size in tokens (default 512).
        _chunk_overlap: Overlap in tokens for standard splits (default 100).
        _code_chunk_overlap: Higher overlap for code splits (default 150).
        _min_chunk_tokens: Minimum tokens required to keep a chunk (default 20).
        _splitter: Standard RecursiveCharacterTextSplitter for paragraphs/headings.
        _code_splitter: Function-boundary-aware splitter for code pages.
        _resplit_splitter: Zero-overlap splitter used only to break oversized chunks.
        _rlm_splitter: Character-based splitter for RLM recursive processing.
    """

    def __init__(self) -> None:
        self._chunk_size: int = settings.chunk_size
        self._chunk_overlap: int = settings.chunk_overlap
        self._code_chunk_overlap: int = settings.code_chunk_overlap
        self._min_chunk_tokens: int = settings.min_chunk_tokens

        # Standard splitter for paragraph and heading pages
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Code splitter splits at function and class definition boundaries
        self._code_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_chunk_overlap,
            separators=["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
        )

        # Zero-overlap splitter used ONLY to break oversized chunks in _filter_chunks.
        # Must not use self._splitter here — its 100-token overlap would cause adjacent
        # sub-chunks to share content, making the LLM see the same text twice.
        self._resplit_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # RLM splitter for recursive processing (Strategy D) — character-based,
        # smaller chunks with minimal overlap for recursive summarization
        self._rlm_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.max_tokens_per_chunk,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            "Chunker initialized: chunk_size=%d tokens, overlap=%d tokens, min_tokens=%d",
            self._chunk_size,
            self._chunk_overlap,
            self._min_chunk_tokens,
        )

    # Main entry point

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split structure-tagged documents into chunks ready for embedding.

        Processes each page through structure-aware splitting, then applies
        filtering, deduplication, metadata enrichment, and context prepending
        in a fixed pipeline order.

        Args:
            documents: List of Documents from StructurePreserver.preserve().

        Returns:
            List of chunked, filtered, enriched Documents.
        """
        if not documents:
            logger.info("Chunker received empty document list")
            return []

        all_chunks = []
        seen_hashes: set = set()

        try:
            for doc in documents:
                chunks = self._split_by_structure(doc)
                chunks = self._filter_chunks(chunks)
                chunks = self._deduplicate(chunks, seen_hashes)
                chunks = self._enrich_metadata(chunks, doc)
                chunks = self._prepend_context(chunks)
                all_chunks.extend(chunks)

            all_chunks = self._add_total_chunks(all_chunks)

            logger.info(
                "Chunker complete: pages=%d, chunks=%d",
                len(documents),
                len(all_chunks),
            )
            return all_chunks

        except Exception as e:
            logger.exception("split_documents failed: %s", e)
            return []

    # Structure-aware routing

    def _split_by_structure(self, doc: Document) -> List[Document]:
        """Route a document page to the correct splitter based on structure_type.

        Routing table:
            table     → _split_table()    (row-group split, header repeated)
            code      → _split_code()     (function-boundary split, 150-token overlap)
            list      → _split_list()     (item-group split, 1-item overlap)
            heading   → _standard_split() (standard recursive split)
            paragraph → _standard_split() (standard recursive split)

        Args:
            doc: Single Document with structure_type set in metadata.

        Returns:
            List of chunk Documents.
        """
        structure_type = doc.metadata.get("structure_type", "paragraph")

        if structure_type == "table":
            return self._split_table(doc)
        if structure_type == "code":
            return self._split_code(doc)
        if structure_type == "list":
            return self._split_list(doc)

        return self._standard_split(doc)

    def _standard_split(self, doc: Document) -> List[Document]:
        """Standard recursive character split for paragraphs and headings.

        Args:
            doc: Document to split.

        Returns:
            List of chunk Documents.
        """
        return self._splitter.split_documents([doc])

    # Table splitting

    def _split_table(self, doc: Document) -> List[Document]:
        """Split a table page by row groups, repeating the header in each chunk.

        Small table (fits within chunk_size) → kept as a single chunk.
        Large table (exceeds chunk_size) → split by row groups with the
        header row prepended to each chunk for context. No row overlap is
        used — repeating the header is more useful than duplicating rows.

        Args:
            doc: Document tagged as structure_type='table'.

        Returns:
            List of chunk Documents.
        """
        token_count = self._count_tokens(doc.page_content)

        if token_count <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large table (%d tokens) — splitting by rows: page=%s",
            token_count,
            doc.metadata.get("page", "?"),
        )

        lines = doc.page_content.strip().split("\n")
        header = lines[0] if lines else ""
        rows = lines[1:]

        chunks: List[Document] = []
        current_rows: List[str] = [header]
        current_tokens = self._count_tokens(header)

        for row in rows:
            row_tokens = self._count_tokens(row)

            # Current group is full — flush it and start a new group with the header
            if current_tokens + row_tokens > self._chunk_size and len(current_rows) > 1:
                chunks.append(Document(
                    page_content="\n".join(current_rows),
                    metadata=doc.metadata.copy(),
                ))
                current_rows = [header, row]
                current_tokens = self._count_tokens(header) + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        # Flush the final group
        if len(current_rows) > 1:
            chunks.append(Document(
                page_content="\n".join(current_rows),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    # Code splitting

    def _split_code(self, doc: Document) -> List[Document]:
        """Split a code page at function and class boundaries.

        Small code block (fits within chunk_size) → kept intact.
        Large code block (exceeds chunk_size) → split at def/class boundaries
        with 150-token overlap to preserve function context across chunks.

        Args:
            doc: Document tagged as structure_type='code'.

        Returns:
            List of chunk Documents.
        """
        if self._count_tokens(doc.page_content) <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large code block — splitting by function boundary: page=%s",
            doc.metadata.get("page", "?"),
        )
        return self._code_splitter.split_documents([doc])

    # List splitting

    def _split_list(self, doc: Document) -> List[Document]:
        """Split a list page by item groups with 1-item overlap between groups.

        Small list (fits within chunk_size) → kept as a single chunk.
        Large list (exceeds chunk_size) → split into item groups where the
        last item of each group is carried into the next for continuity.

        Args:
            doc: Document tagged as structure_type='list'.

        Returns:
            List of chunk Documents.
        """
        token_count = self._count_tokens(doc.page_content)

        if token_count <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large list (%d tokens) — splitting by item groups: page=%s",
            token_count,
            doc.metadata.get("page", "?"),
        )

        # Split on bullet or numbered list item markers
        items = re.split(
            r"(?=^[\-\•\*]\s|^\d+[\.\)]\s)",
            doc.page_content,
            flags=re.MULTILINE,
        )
        items = [item for item in items if item.strip()]

        chunks: List[Document] = []
        current_items: List[str] = []
        current_tokens = 0

        for item in items:
            item_tokens = self._count_tokens(item)

            # Current group is full — flush it and carry the last item forward
            if current_tokens + item_tokens > self._chunk_size and current_items:
                chunks.append(Document(
                    page_content="".join(current_items),
                    metadata=doc.metadata.copy(),
                ))
                # 1-item overlap provides continuity between adjacent chunks
                last_item = current_items[-1]
                current_items = [last_item, item]
                current_tokens = self._count_tokens(last_item) + item_tokens
            else:
                current_items.append(item)
                current_tokens += item_tokens

        # Flush the final group
        if current_items:
            chunks.append(Document(
                page_content="".join(current_items),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    # Post-split processing

    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove low-quality chunks and attempt to break oversized ones.

        Filters applied:
            empty       → blank content after strip
            too short   → token_count < min_chunk_tokens
            boilerplate → standalone page numbers, copyright lines

        Oversized chunks (> chunk_size tokens) are re-split with the zero-overlap
        splitter. If re-splitting produces more than one sub-chunk, those pass
        through basic quality checks and replace the original. If re-splitting
        cannot break the chunk (single indivisible block), it is kept with a
        warning to avoid silent data loss.

        Args:
            chunks: List of raw chunks from splitting.

        Returns:
            Filtered list of chunks.
        """
        _BOILERPLATE = re.compile(
            r"^(\s*\d+\s*|page\s+\d+|all rights reserved|confidential)$",
            re.IGNORECASE,
        )

        filtered = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            token_count = self._count_tokens(content)

            if not content:
                continue

            if token_count < self._min_chunk_tokens:
                logger.debug("Filtered: too short (%d tokens)", token_count)
                continue

            if _BOILERPLATE.match(content):
                logger.debug("Filtered: boilerplate '%s'", content[:40])
                continue

            # Oversized chunks are re-split with zero overlap to prevent the LLM
            # seeing duplicate content that would appear in overlapping sub-chunks
            if token_count > self._chunk_size:
                sub_chunks = self._resplit_splitter.split_documents([chunk])
                if len(sub_chunks) > 1:
                    for sub in sub_chunks:
                        sub_content = sub.page_content.strip()
                        sub_tokens = self._count_tokens(sub_content)
                        if (sub_content
                                and sub_tokens >= self._min_chunk_tokens
                                and not _BOILERPLATE.match(sub_content)):
                            filtered.append(sub)
                    logger.info(
                        "Oversized chunk (%d tokens) re-split → %d sub-chunks: source=%s",
                        token_count,
                        len(sub_chunks),
                        chunk.metadata.get("source", "?"),
                    )
                    continue
                # Re-split ineffective — keep the oversized chunk rather than lose it
                logger.warning(
                    "Oversized chunk kept (%d > %d tokens) — re-split ineffective: source=%s",
                    token_count,
                    self._chunk_size,
                    chunk.metadata.get("source", "?"),
                )

            filtered.append(chunk)

        return filtered

    def _deduplicate(
        self,
        chunks: List[Document],
        seen_hashes: set,
    ) -> List[Document]:
        """Remove duplicate chunks using SHA-256 content hashing.

        Uses hash_text() from utils/helpers.py so fingerprints are consistent
        with cache keys and other project identifiers. seen_hashes is shared
        across all pages in a single split_documents() call, catching
        cross-page duplicates as well as within-page ones.

        Args:
            chunks: List of chunks to deduplicate.
            seen_hashes: Shared set of already-seen content hashes.

        Returns:
            List of unique chunks.
        """
        unique = []
        for chunk in chunks:
            content_hash = hash_text(chunk.page_content)

            if content_hash in seen_hashes:
                logger.debug("Deduplicated: hash=%s...", content_hash[:12])
                continue

            seen_hashes.add(content_hash)
            unique.append(chunk)

        return unique

    def _enrich_metadata(
        self,
        chunks: List[Document],
        source_doc: Document,
    ) -> List[Document]:
        """Add computed metadata fields to each chunk.

        Fields added:
            chunk_index  → position within the source page (0-based).
            word_count   → word count of the chunk content.
            token_count  → exact tiktoken token count.
            doc_type     → file extension derived from the source path.
            chunk_id     → SHA-256 hash of the content for deduplication.

        Args:
            chunks: List of chunks to enrich.
            source_doc: Original source document (used to derive doc_type).

        Returns:
            Same chunks with enriched metadata in-place.
        """
        source = source_doc.metadata.get("source", "")
        doc_type = source.split(".")[-1].lower() if "." in source else "unknown"

        for idx, chunk in enumerate(chunks):
            content = chunk.page_content
            chunk.metadata.update({
                "chunk_index": idx,
                "word_count": len(content.split()),
                "token_count": self._count_tokens(content),
                "doc_type": doc_type,
                "chunk_id": hash_text(content),
            })

        return chunks

    def _prepend_context(self, chunks: List[Document]) -> List[Document]:
        """Prepend title and section context for richer embedding vectors.

        Writes to metadata['embed_content'] — used by VectorStore for
        embedding. The original page_content is kept unchanged for LLM
        answer generation, preventing context bleed into the answer text.

        Result format:
            embed_content = "Title: report.pdf | Section: Introduction\\n{content}"
            page_content  = original clean text (unchanged)

        Args:
            chunks: List of chunks to add context to.

        Returns:
            Same chunks with embed_content added to metadata.
        """
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            section = chunk.metadata.get("section", "unknown")
            # Extract filename only — handles both forward-slash and backslash paths
            title = source.split("/")[-1].split("\\")[-1]

            chunk.metadata["embed_content"] = (
                f"Title: {title} | Section: {section}\n"
                f"{chunk.page_content}"
            )

        return chunks

    def _add_total_chunks(self, all_chunks: List[Document]) -> List[Document]:
        """Add a total_chunks count per source document to every chunk's metadata.

        Runs after all pages are processed so the full per-source count is
        available. Groups chunks by source path, counts them, then writes
        total_chunks back to each chunk.

        Args:
            all_chunks: Complete list of all chunks from all pages.

        Returns:
            Same chunks with total_chunks added to metadata.
        """
        source_counts: Dict[str, int] = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["total_chunks"] = source_counts[source]

        return all_chunks

    # Token counting

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the cl100k_base tiktoken encoder.

        Args:
            text: Text to tokenize.

        Returns:
            Token count as an integer.
        """
        return len(_ENCODER.encode(text))

    # Public utility methods

    def split_by_character(self, text: str) -> List[str]:
        """Split raw text using the standard recursive splitter (Strategy B).

        Suitable for general RAG ingestion of PDFs, DOCX, and plain text.

        Args:
            text: Raw text to split.

        Returns:
            List of chunk strings.
        """
        if not text or not text.strip():
            logger.warning("split_by_character received empty text")
            return []

        try:
            chunks = self._splitter.split_text(text)
            logger.debug("Character split: chunks=%d", len(chunks))
            return chunks
        except Exception as e:
            logger.exception("split_by_character failed: %s", e)
            return []

    def split_for_rlm(self, text: str) -> List[str]:
        """Split raw text for RLM recursive processing (Strategy D).

        Uses smaller chunks with minimal overlap to support recursive
        summarization without accumulating duplicate context.

        Args:
            text: Raw text to split.

        Returns:
            List of chunk strings.
        """
        if not text or not text.strip():
            logger.warning("split_for_rlm received empty text")
            return []

        try:
            chunks = self._rlm_splitter.split_text(text)
            logger.debug("RLM split: chunks=%d", len(chunks))
            return chunks
        except Exception as e:
            logger.exception("split_for_rlm failed: %s", e)
            return []

    def chunk_stats(self, chunks: list) -> dict:
        """Return summary statistics for a list of chunks.

        Accepts both List[Document] and List[str]. Useful for debugging
        and pipeline validation after splitting.

        Args:
            chunks: List of Document objects or raw strings.

        Returns:
            Dict with count, min/max/avg chars, min/max/avg tokens,
            and a list of unique structure_type values (Documents only).
        """
        if not chunks:
            return {
                "count": 0,
                "total_chunks": 0,
                "min_chars": 0,
                "max_chars": 0,
                "avg_chars": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "avg_tokens": 0,
                "structure_types": [],
            }

        # Support both Document objects and plain strings
        is_documents = hasattr(chunks[0], "metadata")

        if is_documents:
            char_counts = [len(c.page_content) for c in chunks]
            token_counts = [c.metadata.get("token_count", 0) for c in chunks]
        else:
            char_counts = [len(c) for c in chunks]
            token_counts = [self._count_tokens(c) for c in chunks]

        structure_types: list[str] = []
        if is_documents:
            structure_types = list({
                c.metadata.get("structure_type", "unknown") for c in chunks
            })

        return {
            "count": len(chunks),
            "total_chunks": len(chunks),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_chars": round(sum(char_counts) / len(char_counts)),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts)),
            "structure_types": structure_types,
        }
