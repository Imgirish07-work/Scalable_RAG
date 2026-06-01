"""Detects structural elements in cleaned documents and enriches their metadata."""

import re
from typing import List

from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class StructurePreserver:
    """Detects and tags structural elements (headings, tables, lists, code) in document pages."""

    _MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    _PLAIN_HEADING = re.compile(
        r"^([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){1,8})$",
        re.MULTILINE,
    )

    _MARKDOWN_TABLE = re.compile(r"^\|.+\|$", re.MULTILINE)
    _PLAIN_TABLE = re.compile(r"(\w+\s{3,}\w+.*\n){2,}", re.MULTILINE)

    _BULLET_LIST = re.compile(r"^[\-\•\*]\s+.+$", re.MULTILINE)
    _NUMBERED_LIST = re.compile(r"^\d+[\.\)]\s+.+$", re.MULTILINE)

    _CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    _INDENTED_CODE = re.compile(r"(^(    |\t).+\n){3,}", re.MULTILINE)

    def preserve(self, documents: List[Document]) -> List[Document]:
        """Tag every document page with structure metadata."""
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
        """Detect structure in one page and write metadata fields."""
        text = doc.page_content

        heading, heading_level = self._detect_heading(text)
        has_table = self._detect_table(text)
        has_list = self._detect_list(text)
        has_code = self._detect_code(text)

        if heading:
            current_section = heading

        structure_type = self._resolve_structure_type(
            heading, has_table, has_list, has_code
        )

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
        """Return the first heading found and its level."""
        match = self._MARKDOWN_HEADING.search(text)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            return heading, level

        match = self._PLAIN_HEADING.search(text)
        if match:
            heading = match.group(0).strip()
            return heading, 2

        return "", 0

    def _detect_table(self, text: str) -> bool:
        return bool(
            self._MARKDOWN_TABLE.search(text)
            or self._PLAIN_TABLE.search(text)
        )

    def _detect_list(self, text: str) -> bool:
        return bool(
            self._BULLET_LIST.search(text)
            or self._NUMBERED_LIST.search(text)
        )

    def _detect_code(self, text: str) -> bool:
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
        """Return the dominant structure type for the page (table > code > list > heading > paragraph)."""
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
        """Log a summary count of each detected structure type."""
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
