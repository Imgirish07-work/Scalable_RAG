"""
Structure preserver — detects and tags structural elements in documents.

Runs AFTER DocumentCleaner, BEFORE Chunker. Analyzes each document page
and tags it with structure metadata that the Chunker uses to decide
splitting strategy (table pages stay intact, code pages get function-boundary
splits, etc.).

Pipeline position:
    DocumentCleaner.load_and_clean()
        → StructurePreserver.preserve()       ← here
            → Chunker.split_documents()

Metadata added per Document:
    section        : str  → current heading (e.g. "Introduction")
    heading_level  : int  → 1, 2, 3 (0 = no heading)
    structure_type : str  → "heading" | "table" | "list" | "code" | "paragraph"
    has_table      : bool → True if page contains a table
    has_list       : bool → True if page contains a list
    has_code       : bool → True if page contains a code block

Sync — 100% regex + CPU, zero I/O.
"""

import re
from typing import List

from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class StructurePreserver:
    """Detects and tags structural elements in documents.

    Reads raw cleaned text, applies regex patterns to detect headings,
    tables, lists, and code blocks. Tags each page with metadata that
    the Chunker reads to select the right splitting strategy.

    Attributes:
        _MARKDOWN_HEADING: Pattern for # H1, ## H2, ### H3.
        _PLAIN_HEADING: Pattern for title-case lines (tightened to avoid false positives).
        _MARKDOWN_TABLE: Pattern for |col|col| markdown tables.
        _PLAIN_TABLE: Pattern for PDF-extracted column-aligned tables.
        _BULLET_LIST: Pattern for - or * bullet items.
        _NUMBERED_LIST: Pattern for 1. or 1) numbered items.
        _CODE_BLOCK: Pattern for ``` fenced code blocks.
        _INDENTED_CODE: Pattern for 3+ consecutive indented lines (avoids false positives).
    """

    # Heading patterns
    _MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    # Tightened: requires 2+ words, no trailing punctuation, no common non-heading phrases
    _PLAIN_HEADING = re.compile(
        r"^([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){1,8})$",
        re.MULTILINE,
    )

    # Table patterns
    _MARKDOWN_TABLE = re.compile(r"^\|.+\|$", re.MULTILINE)
    _PLAIN_TABLE = re.compile(r"(\w+\s{3,}\w+.*\n){2,}", re.MULTILINE)

    # List patterns
    _BULLET_LIST = re.compile(r"^[\-\•\*]\s+.+$", re.MULTILINE)
    _NUMBERED_LIST = re.compile(r"^\d+[\.\)]\s+.+$", re.MULTILINE)

    # Code block patterns
    _CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    # Requires 3+ consecutive indented lines to avoid false positives
    # from indented paragraphs in PDFs/DOCX
    _INDENTED_CODE = re.compile(r"(^(    |\t).+\n){3,}", re.MULTILINE)

    def preserve(self, documents: List[Document]) -> List[Document]:
        """Tag every document page with structure metadata.

        Args:
            documents: List of cleaned Documents from DocumentCleaner.

        Returns:
            Same pages with enriched structure metadata.
        """
        if not documents:
            logger.warning("StructurePreserver received empty document list")
            return documents

        logger.info("StructurePreserver processing %d page(s)", len(documents))

        preserved = []
        current_section = "unknown"

        for doc in documents:
            tagged_doc, current_section = self._tag_document(doc, current_section)
            preserved.append(tagged_doc)

        self._log_summary(preserved)
        return preserved

    def _tag_document(
        self, doc: Document, current_section: str
    ) -> tuple[Document, str]:
        """Detect structure in a single page and enrich its metadata.

        Args:
            doc: Document page to analyze.
            current_section: Section heading carried from previous page.

        Returns:
            Tuple of (tagged Document, updated current_section).
        """
        text = doc.page_content

        # Detect all structural elements
        heading, heading_level = self._detect_heading(text)
        has_table = self._detect_table(text)
        has_list = self._detect_list(text)
        has_code = self._detect_code(text)

        # Update running section if a heading is found
        if heading:
            current_section = heading

        # Determine dominant structure type
        structure_type = self._resolve_structure_type(
            heading, has_table, has_list, has_code
        )

        # Preserve all existing metadata, add structure fields
        enriched_metadata = {
            **doc.metadata,
            "section": current_section,
            "heading_level": heading_level,
            "structure_type": structure_type,
            "has_table": has_table,
            "has_list": has_list,
            "has_code": has_code,
        }

        logger.debug(
            "page=%s, section='%s', type=%s, table=%s, list=%s, code=%s",
            doc.metadata.get("page", "?"),
            current_section,
            structure_type,
            has_table,
            has_list,
            has_code,
        )

        return Document(page_content=text, metadata=enriched_metadata), current_section

    # Structure detection helpers

    def _detect_heading(self, text: str) -> tuple[str, int]:
        """Detect the first heading in the text.

        Priority:
            1. Markdown heading → # H1, ## H2, ### H3 (most reliable)
            2. Plain heading → multi-word title-case line (PDF/DOCX text)

        Args:
            text: Page content to scan.

        Returns:
            (heading_text, heading_level) or ("", 0) if no heading found.
        """
        # Check markdown headings first — most reliable
        match = self._MARKDOWN_HEADING.search(text)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            return heading, level

        # Check plain text headings — PDF/DOCX extracted text
        match = self._PLAIN_HEADING.search(text)
        if match:
            heading = match.group(0).strip()
            # Plain headings default to level 2 — exact level is unknown
            return heading, 2

        return "", 0

    def _detect_table(self, text: str) -> bool:
        """Detect if page contains a table (markdown or column-aligned).

        Args:
            text: Page content to scan.

        Returns:
            True if a table pattern is found.
        """
        return bool(
            self._MARKDOWN_TABLE.search(text)
            or self._PLAIN_TABLE.search(text)
        )

    def _detect_list(self, text: str) -> bool:
        """Detect if page contains a bullet or numbered list.

        Args:
            text: Page content to scan.

        Returns:
            True if a list pattern is found.
        """
        return bool(
            self._BULLET_LIST.search(text)
            or self._NUMBERED_LIST.search(text)
        )

    def _detect_code(self, text: str) -> bool:
        """Detect if page contains a code block.

        Checks fenced blocks (```...```) and indented code (3+ consecutive
        indented lines — avoids false positives from indented paragraphs).

        Args:
            text: Page content to scan.

        Returns:
            True if a code pattern is found.
        """
        return bool(
            self._CODE_BLOCK.search(text)
            or self._INDENTED_CODE.search(text)
        )

    def _resolve_structure_type(
        self,
        heading: str,
        has_table: bool,
        has_list: bool,
        has_code: bool,
    ) -> str:
        """Determine the dominant structure type for the page.

        Priority order (matters for Chunker splitting decisions):
            table     → keep intact, header-repeat on split
            code      → keep intact, function-boundary split
            list      → keep together if possible
            heading   → split point marker
            paragraph → standard recursive split

        Args:
            heading: Detected heading text (empty if none).
            has_table: Whether a table was detected.
            has_list: Whether a list was detected.
            has_code: Whether code was detected.

        Returns:
            Structure type string.
        """
        if has_table:
            return "table"
        if has_code:
            return "code"
        if has_list:
            return "list"
        if heading:
            return "heading"
        return "paragraph"

    def _log_summary(self, documents: List[Document]) -> None:
        """Log a summary of structure detection results.

        Args:
            documents: Tagged documents to summarize.
        """
        total = len(documents)
        headings = sum(1 for d in documents if d.metadata.get("heading_level", 0) > 0)
        tables = sum(1 for d in documents if d.metadata.get("has_table"))
        lists = sum(1 for d in documents if d.metadata.get("has_list"))
        codes = sum(1 for d in documents if d.metadata.get("has_code"))

        logger.info(
            "StructurePreserver complete: pages=%d, headings=%d, "
            "tables=%d, lists=%d, code=%d",
            total, headings, tables, lists, codes,
        )