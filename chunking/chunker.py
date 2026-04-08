"""
Chunker — splits structure-tagged documents into chunks for embedding.

Pipeline position:
    DocumentCleaner.load_and_clean()
        → StructurePreserver.preserve()
            → Chunker.split_documents()     ← here
                → VectorStore.add_documents()

Features:
    - Structure-aware splitting (reads structure_type from metadata)
    - Token-based chunk size (matches BGE-small 512 token limit)
    - Chunk filtering (removes short, empty, boilerplate chunks)
    - Deduplication (SHA-256 hash — consistent with rest of project)
    - Context prepending (title + section for embedding only)
    - Metadata enrichment (word_count, token_count, doc_type, chunk_index)

Splitting strategies by structure type:
    paragraph/heading → RecursiveCharacterTextSplitter (standard)
    table             → row-group splitting with header repeat
    code              → function-boundary splitting (higher overlap)
    list              → item-group splitting with 1-item overlap

Sync — CPU only, zero I/O. Wrapped in asyncio.to_thread() from async.
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

# Module-level encoder — tiktoken caches internally, this makes intent clear
_ENCODER = tiktoken.get_encoding("cl100k_base")


class Chunker:
    """Splits cleaned, structure-tagged documents into chunks for embedding.

    Attributes:
        _chunk_size: Target chunk size in tokens (default 512).
        _chunk_overlap: Overlap in tokens for standard splits (default 100).
        _code_chunk_overlap: Higher overlap for code splits (default 150).
        _min_chunk_tokens: Minimum tokens to keep a chunk (default 20).
        _splitter: Standard RecursiveCharacterTextSplitter for paragraphs.
        _code_splitter: Code-aware splitter with function boundary separators.
        _rlm_splitter: RLM recursive processing splitter.
    """

    def __init__(self) -> None:
        self._chunk_size: int = settings.chunk_size
        self._chunk_overlap: int = settings.chunk_overlap
        self._code_chunk_overlap: int = settings.code_chunk_overlap
        self._min_chunk_tokens: int = settings.min_chunk_tokens

        # Standard splitter for paragraph/heading pages
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Code splitter with function/class boundary separators
        self._code_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_chunk_overlap,
            separators=["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
        )

        # Re-split splitter — zero overlap to prevent duplicate content in sub-chunks.
        # Used ONLY in _filter_chunks() when an oversized chunk needs splitting.
        # Must NOT use self._splitter (overlap=100) there — adjacent sub-chunks would
        # share 100 tokens, causing the LLM to see the same content twice.
        self._resplit_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # RLM splitter for recursive processing (Strategy D)
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

        Processes each document page through structure-aware splitting,
        then applies filtering, deduplication, metadata enrichment,
        and context prepending.

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
        """Route document to the correct splitter based on structure_type.

        Routing:
            table     → _split_table()    (row-group, header repeat)
            code      → _split_code()     (function-boundary, 150 overlap)
            list      → _split_list()     (item-group, 1-item overlap)
            heading   → _standard_split() (standard recursive)
            paragraph → _standard_split() (standard recursive)

        Args:
            doc: Single Document with structure_type in metadata.

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
        """Standard recursive split for paragraphs and headings.

        Args:
            doc: Document to split.

        Returns:
            List of chunk Documents.
        """
        return self._splitter.split_documents([doc])

    # Table splitting

    def _split_table(self, doc: Document) -> List[Document]:
        """Table-aware splitting with header row repetition.

        Small table (fits in chunk_size) → keep as single chunk.
        Large table (exceeds chunk_size) → split by row groups,
        header row repeated at the top of each chunk for context.
        No overlap between row groups — header repeat is better
        than row duplication.

        Args:
            doc: Document tagged as structure_type='table'.

        Returns:
            List of chunk Documents.
        """
        token_count = self._count_tokens(doc.page_content)

        # Small table — keep intact
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

            # Current group is full — save it and start new group
            if current_tokens + row_tokens > self._chunk_size and len(current_rows) > 1:
                chunks.append(Document(
                    page_content="\n".join(current_rows),
                    metadata=doc.metadata.copy(),
                ))
                # New group starts with header for context
                current_rows = [header, row]
                current_tokens = self._count_tokens(header) + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        # Save last group
        if len(current_rows) > 1:
            chunks.append(Document(
                page_content="\n".join(current_rows),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    # Code splitting

    def _split_code(self, doc: Document) -> List[Document]:
        """Code-aware splitting at function/class boundaries.

        Small code (fits in chunk_size) → keep intact.
        Large code (exceeds chunk_size) → split at function/class
        boundaries with 150 token overlap for context.

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
        """List-aware splitting with 1-item overlap between groups.

        Small list (fits in chunk_size) → keep as single chunk.
        Large list (exceeds chunk_size) → split by item groups,
        last item of each group carried into the next for context.

        Args:
            doc: Document tagged as structure_type='list'.

        Returns:
            List of chunk Documents.
        """
        token_count = self._count_tokens(doc.page_content)

        # Small list — keep intact
        if token_count <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large list (%d tokens) — splitting by item groups: page=%s",
            token_count,
            doc.metadata.get("page", "?"),
        )

        # Split on bullet or numbered list markers
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

            # Current group is full — save it
            if current_tokens + item_tokens > self._chunk_size and current_items:
                chunks.append(Document(
                    page_content="".join(current_items),
                    metadata=doc.metadata.copy(),
                ))
                # 1-item overlap: carry last item into next group for context
                last_item = current_items[-1]
                current_items = [last_item, item]
                current_tokens = self._count_tokens(last_item) + item_tokens
            else:
                current_items.append(item)
                current_tokens += item_tokens

        # Save last group
        if current_items:
            chunks.append(Document(
                page_content="".join(current_items),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    # Post-split processing

    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove low-quality chunks after splitting.

        Filters:
            empty       → blank content after strip
            too short   → token_count < min_chunk_tokens (default 20)
            boilerplate → standalone page numbers, copyright lines

        Oversized chunks (> chunk_size) are re-split with the standard
        recursive splitter before being kept. This recovers chunks that
        the list/table splitters couldn't break (single item > chunk_size).
        If re-splitting produces no improvement, the original is kept with
        a warning to avoid silent data loss.

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

            # Skip empty chunks
            if not content:
                continue

            # Skip chunks below minimum token threshold
            if token_count < self._min_chunk_tokens:
                logger.debug("Filtered: too short (%d tokens)", token_count)
                continue

            # Skip boilerplate content
            if _BOILERPLATE.match(content):
                logger.debug("Filtered: boilerplate '%s'", content[:40])
                continue

            # Oversized: attempt re-split with zero-overlap splitter.
            # _resplit_splitter uses chunk_overlap=0 — prevents adjacent sub-chunks
            # sharing content that would make the LLM see the same text twice.
            if token_count > self._chunk_size:
                sub_chunks = self._resplit_splitter.split_documents([chunk])
                if len(sub_chunks) > 1:
                    # Re-split worked — apply basic quality filter to sub-chunks
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
                # Re-split ineffective (single indivisible block) — keep as-is
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
        """Remove duplicate chunks using SHA-256 content hash.

        Uses hash_text() from utils/helpers.py for consistency with
        the rest of the project (cache keys, fingerprints all use SHA-256).

        seen_hashes is shared across all pages in a single
        split_documents() call — catches cross-page duplicates.

        Args:
            chunks: List of chunks to deduplicate.
            seen_hashes: Shared set of seen content hashes.

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
        """Add computed metadata to each chunk.

        Adds:
            chunk_index  → position within this page (0-based)
            word_count   → word count of chunk content
            token_count  → exact tiktoken token count
            doc_type     → file extension from source metadata

        Args:
            chunks: List of chunks to enrich.
            source_doc: Original source document (for doc_type).

        Returns:
            Same chunks with enriched metadata.
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
        """Prepend title and section to each chunk for embedding only.

        Stored in metadata['embed_content'] — used by VectorStore for
        embedding richer semantic vectors. Original page_content is
        preserved unchanged — used by LLM for answer generation.

        Result:
            embed_content = "Title: report.pdf | Section: Introduction\\n{content}"
            page_content  = original clean text (unchanged)

        Args:
            chunks: List of chunks to add context to.

        Returns:
            Same chunks with embed_content in metadata.
        """
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            section = chunk.metadata.get("section", "unknown")
            # Extract filename only (handles both / and \\ paths)
            title = source.split("/")[-1].split("\\")[-1]

            chunk.metadata["embed_content"] = (
                f"Title: {title} | Section: {section}\n"
                f"{chunk.page_content}"
            )

        return chunks

    def _add_total_chunks(self, all_chunks: List[Document]) -> List[Document]:
        """Add total_chunks count per source document.

        Runs after all pages are processed — needs the complete chunk list
        to count per source. Groups by source file, counts, writes back.

        Args:
            all_chunks: Complete list of all chunks from all pages.

        Returns:
            Same chunks with total_chunks added to metadata.
        """
        # Count chunks per source
        source_counts: Dict[str, int] = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        # Write total_chunks back
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["total_chunks"] = source_counts[source]

        return all_chunks

    # Token counting

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoder.

        Used for chunk size validation and structure-aware splitting.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        return len(_ENCODER.encode(text))

    # Public utility methods

    def split_by_character(self, text: str) -> List[str]:
        """Split text using Strategy B — general RAG splitting.

        Best for PDFs, docs, general text ingestion.

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
        """Split text for RLM recursive processing — Strategy D.

        Smaller chunks with minimal overlap for recursive summarization.

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

        Accepts both List[Document] and List[str].
        Useful for debugging and pipeline validation.

        Args:
            chunks: List of Document objects or raw strings.

        Returns:
            Dict with count, min/max/avg chars, min/max/avg tokens.
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

        # Handle both Document and str types
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