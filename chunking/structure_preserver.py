"""
Detects structural elements in cleaned documents and enriches their metadata.

Design:
    Single-responsibility class (StructurePreserver) that applies a prioritized
    set of compiled regex patterns to each document page. Results are written
    to Document.metadata so the downstream Chunker can select the appropriate
    splitting strategy without re-scanning the text.

Chain of Responsibility:
    Receives List[Document] from DocumentCleaner.load_and_clean() →
    enriches metadata → passes enriched List[Document] to
    Chunker.split_documents().

Dependencies:
    langchain_core.documents.Document, utils.logger.
"""

import re
from typing import List

from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class StructurePreserver:
    """Detects and tags structural elements in document pages.

    Reads cleaned text, applies regex patterns to detect headings, tables,
    lists, and code blocks, then writes structure fields to each Document's
    metadata. The Chunker reads these fields to choose the right splitter.

    Attributes:
        _MARKDOWN_HEADING: Pattern for # H1, ## H2, ### H3 headings.
        _PLAIN_HEADING: Title-case multi-word lines from PDF/DOCX text.
        _MARKDOWN_TABLE: Pipe-delimited markdown table rows.
        _PLAIN_TABLE: Column-aligned text tables from PDF extraction.
        _BULLET_LIST: Dash, bullet, or asterisk list items.
        _NUMBERED_LIST: Decimal-numbered list items (1. or 1)).
        _CODE_BLOCK: Fenced code blocks (```...```).
        _INDENTED_CODE: Three or more consecutive indented lines.
    """

    # Markdown headings are the most reliable — checked first
    _MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    # Tightened to require 2+ words with no trailing punctuation,
    # reducing false positives from capitalized sentences in body text
    _PLAIN_HEADING = re.compile(
        r"^([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){1,8})$",
        re.MULTILINE,
    )

    _MARKDOWN_TABLE = re.compile(r"^\|.+\|$", re.MULTILINE)
    _PLAIN_TABLE = re.compile(r"(\w+\s{3,}\w+.*\n){2,}", re.MULTILINE)

    _BULLET_LIST = re.compile(r"^[\-\•\*]\s+.+$", re.MULTILINE)
    _NUMBERED_LIST = re.compile(r"^\d+[\.\)]\s+.+$", re.MULTILINE)

    _CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    # Requires 3+ consecutive indented lines to avoid mistaking indented
    # body paragraphs in PDFs and DOCX for code blocks
    _INDENTED_CODE = re.compile(r"(^(    |\t).+\n){3,}", re.MULTILINE)

    def preserve(self, documents: List[Document]) -> List[Document]:
        """Tag every document page with structure metadata.

        Args:
            documents: List of cleaned Documents from DocumentCleaner.

        Returns:
            Same pages with structure metadata added to each Document.
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
        """Detect structure in one page and write metadata fields.

        Args:
            doc: Document page to analyze.
            current_section: Section heading carried forward from the previous page.

        Returns:
            Tuple of (tagged Document, updated current_section string).
        """
        text = doc.page_content

        heading, heading_level = self._detect_heading(text)
        has_table = self._detect_table(text)
        has_list = self._detect_list(text)
        has_code = self._detect_code(text)

        # Propagate the new heading as the running section label
        if heading:
            current_section = heading

        structure_type = self._resolve_structure_type(
            heading, has_table, has_list, has_code
        )

        # Merge structure fields into a copy of the existing metadata
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

    def _detect_heading(self, text: str) -> tuple[str, int]:
        """Return the first heading found and its level.

        Priority:
            1. Markdown heading (# / ## / ###) — most reliable signal.
            2. Plain title-case line — fallback for PDF/DOCX text.

        Args:
            text: Page content to scan.

        Returns:
            (heading_text, heading_level) or ("", 0) when no heading is found.
        """
        # Markdown headings carry an explicit level from the # prefix count
        match = self._MARKDOWN_HEADING.search(text)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            return heading, level

        # Plain text headings have no explicit level — default to 2
        match = self._PLAIN_HEADING.search(text)
        if match:
            heading = match.group(0).strip()
            return heading, 2

        return "", 0

    def _detect_table(self, text: str) -> bool:
        """Return True if the page contains a markdown or column-aligned table.

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
        """Return True if the page contains a bullet or numbered list.

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
        """Return True if the page contains a code block.

        Checks fenced blocks (```...```) and 3+ consecutive indented lines.
        The indented-line threshold avoids flagging indented body paragraphs.

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
        """Return the dominant structure type for the page.

        Priority order (highest to lowest) drives Chunker splitting decisions:
            table     → keep intact, repeat header row on split
            code      → keep intact, split at function boundaries
            list      → keep items together with 1-item overlap
            heading   → marks a split point for standard splitter
            paragraph → standard recursive character split

        Args:
            heading: Detected heading text, or empty string if none.
            has_table: True if a table was detected.
            has_list: True if a list was detected.
            has_code: True if code was detected.

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
        """Log a summary count of each detected structure type.

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
