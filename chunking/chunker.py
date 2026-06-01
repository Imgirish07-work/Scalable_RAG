"""Routes structure-tagged document pages to splitters, then filters and enriches chunks."""

import re
from typing import Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger
from utils.helpers import hash_text

logger = get_logger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


class Chunker:
    """Splits cleaned, structure-tagged documents into chunks for embedding."""

    def __init__(self) -> None:
        self._chunk_size: int = settings.chunk_size
        self._chunk_overlap: int = settings.chunk_overlap
        self._code_chunk_overlap: int = settings.code_chunk_overlap
        self._min_chunk_tokens: int = settings.min_chunk_tokens

        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._code_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._code_chunk_overlap,
            separators=["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
        )

        # zero overlap prevents duplicate text across adjacent sub-chunks
        self._resplit_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

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

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split structure-tagged documents into chunks ready for embedding."""
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

    def _split_by_structure(self, doc: Document) -> List[Document]:
        """Route a document page to the correct splitter based on structure_type."""
        structure_type = doc.metadata.get("structure_type", "paragraph")

        if structure_type == "table":
            return self._split_table(doc)
        if structure_type == "code":
            return self._split_code(doc)
        if structure_type == "list":
            return self._split_list(doc)

        return self._standard_split(doc)

    def _standard_split(self, doc: Document) -> List[Document]:
        """Standard recursive character split for paragraphs and headings."""
        return self._splitter.split_documents([doc])

    def _split_table(self, doc: Document) -> List[Document]:
        """Split a table page by row groups, repeating the header in each chunk."""
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

        if len(current_rows) > 1:
            chunks.append(Document(
                page_content="\n".join(current_rows),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    def _split_code(self, doc: Document) -> List[Document]:
        """Split a code page at function and class boundaries."""
        if self._count_tokens(doc.page_content) <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large code block — splitting by function boundary: page=%s",
            doc.metadata.get("page", "?"),
        )
        return self._code_splitter.split_documents([doc])

    def _split_list(self, doc: Document) -> List[Document]:
        """Split a list page by item groups with 1-item overlap between groups."""
        token_count = self._count_tokens(doc.page_content)

        if token_count <= self._chunk_size:
            return [doc]

        logger.debug(
            "Large list (%d tokens) — splitting by item groups: page=%s",
            token_count,
            doc.metadata.get("page", "?"),
        )

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

            if current_tokens + item_tokens > self._chunk_size and current_items:
                chunks.append(Document(
                    page_content="".join(current_items),
                    metadata=doc.metadata.copy(),
                ))
                last_item = current_items[-1]
                current_items = [last_item, item]
                current_tokens = self._count_tokens(last_item) + item_tokens
            else:
                current_items.append(item)
                current_tokens += item_tokens

        if current_items:
            chunks.append(Document(
                page_content="".join(current_items),
                metadata=doc.metadata.copy(),
            ))

        return chunks if chunks else [doc]

    _MIN_ALPHA_RATIO: float = 0.40
    # alpha gate only applied to short chunks; long low-alpha chunks are likely code/data tables
    _ALPHA_GATE_MAX_TOKENS: int = 60

    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove low-quality chunks and attempt to break oversized ones."""
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

            if token_count < self._ALPHA_GATE_MAX_TOKENS and content:
                alpha_chars = sum(1 for c in content if c.isalpha())
                alpha_ratio = alpha_chars / len(content)
                if alpha_ratio < self._MIN_ALPHA_RATIO:
                    logger.debug(
                        "Filtered: low alpha ratio (%.2f < %.2f, %d tokens) '%s'",
                        alpha_ratio,
                        self._MIN_ALPHA_RATIO,
                        token_count,
                        content[:60],
                    )
                    continue

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
        """Remove duplicate chunks using SHA-256 content hashing."""
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
        """Add computed metadata fields to each chunk."""
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
        """Prepend title and section context to embed_content for richer embedding vectors."""
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            section = chunk.metadata.get("section", "unknown")
            title = source.split("/")[-1].split("\\")[-1]

            chunk.metadata["embed_content"] = (
                f"Title: {title} | Section: {section}\n"
                f"{chunk.page_content}"
            )

        return chunks

    def _add_total_chunks(self, all_chunks: List[Document]) -> List[Document]:
        """Add a total_chunks count per source document to every chunk's metadata."""
        source_counts: Dict[str, int] = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["total_chunks"] = source_counts[source]

        return all_chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the cl100k_base tiktoken encoder."""
        return len(_ENCODER.encode(text))

    def split_by_character(self, text: str) -> List[str]:
        """Split raw text using the standard recursive splitter."""
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
        """Split raw text for RLM recursive processing."""
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
        """Return summary statistics for a list of chunks (accepts Documents or strings)."""
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
