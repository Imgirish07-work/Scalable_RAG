"""
Loads and cleans raw documents as the first step in the ingestion pipeline.

Design:
    Single-responsibility class (DocumentCleaner) that handles format detection,
    loader selection, and multi-step text cleaning. Loader choice is driven by
    a settings flag (prefer_pdfplumber) with automatic fallback for PDFs.

Chain of Responsibility:
    Called by the ingestion pipeline → passes cleaned List[Document] to
    StructurePreserver.preserve() → then Chunker.split_documents() →
    then VectorStore.add_documents().

Dependencies:
    ftfy, langchain_community (PyMuPDFLoader, PDFPlumberLoader, Docx2txtLoader,
    TextLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader),
    langchain_core.documents.Document, config.settings, utils.logger.
"""

import re
import os
from pathlib import Path
from typing import List

import ftfy
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PDFPlumberLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentCleaner:
    """Loads documents from disk and cleans extracted text.

    Attributes:
        _min_chars_per_page: Minimum characters required to keep a page.
        _prefer_pdfplumber: If True, use PDFPlumberLoader instead of PyMuPDF.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

    # Only standalone URL lines are stripped — inline URLs within content are kept.
    _BOILERPLATE_PATTERNS = [
        r"all rights reserved",
        r"confidential",
        r"do not distribute",
        r"page\s+\d+\s+of\s+\d+",       # "Page 1 of 10"
        r"^\s*\d+\s*$",                   # standalone page numbers
        r"^\s*http[s]?://[^\s]+\s*$",     # standalone URL lines
        r"^\s*www\.[^\s]+\s*$",           # standalone www lines
    ]

    _BOILERPLATE_REGEX = re.compile(
        "|".join(_BOILERPLATE_PATTERNS),
        re.IGNORECASE | re.MULTILINE,
    )

    # Running header/footer detection thresholds.
    # A line seen verbatim on >= max(MIN_PAGES, PAGE_FRACTION * total_pages) pages
    # is classified as a running header and stripped from all pages before chunking.
    # MAX_WORDS and MAX_CHARS cap what can qualify — real content lines are longer.
    _RUNNING_HEADER_MIN_PAGES: int = 3
    _RUNNING_HEADER_PAGE_FRACTION: float = 0.15
    _RUNNING_HEADER_MAX_WORDS: int = 6
    _RUNNING_HEADER_MAX_CHARS: int = 80

    # Backslash between two word characters is always an OCR artifact in natural prose.
    # PDF text never contains escape sequences; markdown is absent post-extraction.
    _OCR_BACKSLASH_RE = re.compile(r"(?<=[a-zA-Z0-9])\\(?=[a-zA-Z0-9])")

    # Characters outside standard prose, digits, whitespace, and common punctuation.
    # Used to measure symbol density per line for noisy-line filtering.
    _NOISY_SYMBOL_RE = re.compile(r"[^a-zA-Z0-9\s.,!?;:\-'\"()\u00C0-\u024F]")

    # Lines shorter than this skip the noise check — too short for a reliable ratio.
    _NOISE_CHECK_MIN_LEN: int = 5
    # Lines longer than this also skip the noise check — long lines with high symbol
    # density are usually mathematical notation or code, not garbage.
    _NOISE_CHECK_MAX_LEN: int = 100
    # Maximum allowed noisy-symbol fraction before a short line is dropped.
    _MAX_LINE_NOISE_RATIO: float = 0.45

    def __init__(self) -> None:
        self._min_chars_per_page: int = settings.min_chars_per_page
        self._prefer_pdfplumber: bool = settings.prefer_pdfplumber

        logger.debug(
            "DocumentCleaner initialized: min_chars=%d, prefer_pdfplumber=%s",
            self._min_chars_per_page,
            self._prefer_pdfplumber,
        )

    def load_and_clean(self, file_path: str) -> List[Document]:
        """Load a document from disk and return cleaned pages.

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            List of cleaned Document objects with preserved metadata.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file type is not supported.
        """
        file_name = Path(file_path).name
        logger.info("Loading document: %s", file_name)

        docs = self._load_document(file_path)
        cleaned_docs = self._clean_documents(docs)

        logger.info(
            "Loaded %d pages, cleaned to %d pages", len(docs), len(cleaned_docs)
        )
        return cleaned_docs

    def _detect_type(self, file_path: str) -> str:
        """Return the lowercase file extension, raising if unsupported.

        Args:
            file_path: Path to the file.

        Returns:
            Lowercase file extension e.g. '.pdf'.

        Raises:
            ValueError: If the extension is not in SUPPORTED_EXTENSIONS.
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.debug("Detected file type: %s for %s", ext, file_path)
        return ext

    def _load_document(self, file_path: str) -> List[Document]:
        """Select the correct LangChain loader and load the document.

        Loader selection:
            .pdf   → PyMuPDFLoader (default) or PDFPlumberLoader (settings flag)
            .docx  → Docx2txtLoader
            .txt   → TextLoader (utf-8)
            .md    → UnstructuredMarkdownLoader
            .html  → UnstructuredHTMLLoader

        Args:
            file_path: Path to the document.

        Returns:
            List of raw Document objects, one per page or section.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no loader matches the extension.
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = self._detect_type(file_path)
        file_name = Path(file_path).name

        try:
            logger.info("Loading: file=%s, type=%s", file_name, ext)

            if ext == ".pdf":
                docs = self._load_pdf(file_path)
            elif ext == ".docx":
                docs = Docx2txtLoader(file_path).load()
            elif ext == ".txt":
                docs = TextLoader(file_path, encoding="utf-8").load()
            elif ext == ".md":
                docs = UnstructuredMarkdownLoader(file_path).load()
            elif ext in {".html", ".htm"}:
                docs = UnstructuredHTMLLoader(file_path).load()
            else:
                # _detect_type already validates — this branch is unreachable
                raise ValueError(f"No loader for extension: {ext}")

            logger.info("Loaded %d page(s): file=%s", len(docs), file_name)
            return docs

        except Exception as e:
            logger.exception("Failed to load: file=%s, error=%s", file_name, e)
            raise

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF using the preferred loader, with automatic fallback.

        Primary:   PyMuPDFLoader (fast, layout-aware).
        Alternate: PDFPlumberLoader (better table extraction, MIT license).
        Fallback:  PDFPlumberLoader when PyMuPDF raises any exception.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.
        """
        file_name = Path(file_path).name

        if self._prefer_pdfplumber:
            logger.debug("Using PDFPlumberLoader for %s (settings preference)", file_name)
            return PDFPlumberLoader(file_path).load()

        try:
            logger.debug("Attempting PyMuPDFLoader for %s", file_name)
            return PyMuPDFLoader(file_path).load()
        except Exception as e:
            logger.warning(
                "PyMuPDFLoader failed for %s: %s — falling back to PDFPlumberLoader",
                file_name, e,
            )
            return PDFPlumberLoader(file_path).load()

    def _clean_text(self, text: str) -> str:
        """Apply multi-step cleaning to raw extracted text from a single page.

        Steps applied in order:
            1. ftfy           — fix broken unicode and mojibake encoding issues.
            2. OCR artifacts  — remove backslashes between word characters.
            3. Hyphens        — rejoin hyphenated line breaks ('retriev-\\nal' → 'retrieval').
            4. Boilerplate    — remove page numbers, standalone URLs, copyright lines.
            5. Noisy lines    — drop lines whose symbol-noise ratio exceeds the threshold.
            6. Whitespace     — collapse 3+ newlines to 2, collapse repeated spaces.
            7. Strip          — remove leading and trailing whitespace.

        Args:
            text: Raw text extracted from a document page.

        Returns:
            Cleaned text string. Empty string if input was empty or whitespace-only.
        """
        if not text or not text.strip():
            logger.warning("Received empty or whitespace-only text for cleaning")
            return ""

        try:
            # Fix encoding issues such as mojibake and broken unicode
            cleaned = ftfy.fix_text(text)

            # Remove inter-word backslashes — always OCR artifacts in PDF prose
            cleaned = self._remove_ocr_artifacts(cleaned)

            # Rejoin hyphenated line breaks introduced by PDF line wrapping
            cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

            # Remove boilerplate: page numbers, standalone URLs, copyright notices
            cleaned = self._BOILERPLATE_REGEX.sub("", cleaned)

            # Drop lines that are predominantly non-prose symbols (OCR garbage)
            cleaned = self._filter_noisy_lines(cleaned)

            # Normalize whitespace: collapse excess newlines and spaces
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)

            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.exception("Error cleaning text: %s", e)
            return ""

    def _remove_ocr_artifacts(self, text: str) -> str:
        """Remove backslash OCR artifacts between word characters.

        PDF text extraction never produces real backslash escape sequences
        within prose. A backslash between two word characters (e.g., 'retriev\\al')
        is always an artifact from a ligature or font-encoding mapping error.
        The conservative lookbehind/lookahead restricts the match to
        [a-zA-Z0-9] pairs, leaving mathematical notation and code paths untouched.

        Args:
            text: Text after ftfy encoding fixes.

        Returns:
            Text with inter-word backslashes removed.
        """
        return self._OCR_BACKSLASH_RE.sub("", text)

    def _filter_noisy_lines(self, text: str) -> str:
        """Drop lines whose noisy-symbol ratio exceeds the configured threshold.

        Applied per-line after boilerplate removal. Lines shorter than
        _NOISE_CHECK_MIN_LEN or longer than _NOISE_CHECK_MAX_LEN are
        always kept — very short lines lack statistical reliability, and
        long lines with high symbol density are more likely math or code
        than garbage.

        Noisy symbols: any character outside standard prose letters, digits,
        whitespace, and common punctuation (defined by _NOISY_SYMBOL_RE).

        Args:
            text: Text after boilerplate removal.

        Returns:
            Text with high-noise lines removed; blank lines preserved.
        """
        lines = text.split("\n")
        filtered = []
        for line in lines:
            length = len(line)
            # Short or long lines are exempt — see class docstring for rationale
            if length < self._NOISE_CHECK_MIN_LEN or length > self._NOISE_CHECK_MAX_LEN:
                filtered.append(line)
                continue
            noisy_count = len(self._NOISY_SYMBOL_RE.findall(line))
            ratio = noisy_count / length
            if ratio > self._MAX_LINE_NOISE_RATIO:
                logger.debug(
                    "Noisy line dropped: ratio=%.2f '%s'",
                    ratio,
                    line[:60],
                )
                continue
            filtered.append(line)
        return "\n".join(filtered)

    def _detect_running_headers(self, documents: List[Document]) -> frozenset:
        """Identify running headers and footers that repeat verbatim across pages.

        A line is classified as a running header when it appears on at least
        max(MIN_PAGES, PAGE_FRACTION * total_pages) distinct pages AND is
        short enough not to be real content (MAX_WORDS words, MAX_CHARS chars).

        Common examples in large books: chapter titles ("BOOK ONE: 1805"),
        document titles ("Attention Is All You Need"), section markers
        ("CHAPTER V"), and repeating footers. Detecting these cross-document
        prevents every chunk from being polluted with the same header text,
        which would skew embedding similarity for unrelated queries.

        Args:
            documents: Raw Document list from the loader, one per page.

        Returns:
            Frozenset of stripped line strings classified as running headers.
        """
        if not documents:
            return frozenset()

        total_pages = len(documents)
        threshold = max(
            self._RUNNING_HEADER_MIN_PAGES,
            int(self._RUNNING_HEADER_PAGE_FRACTION * total_pages),
        )

        # Count distinct pages each candidate line appears on.
        # Using a set per page ensures a line repeated within one page counts once.
        line_page_count: dict = {}
        for doc in documents:
            page_lines: set = set()
            for line in doc.page_content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if len(stripped) > self._RUNNING_HEADER_MAX_CHARS:
                    continue
                if len(stripped.split()) > self._RUNNING_HEADER_MAX_WORDS:
                    continue
                page_lines.add(stripped)
            for line in page_lines:
                line_page_count[line] = line_page_count.get(line, 0) + 1

        headers = frozenset(
            line for line, count in line_page_count.items()
            if count >= threshold
        )

        if headers:
            logger.info(
                "Detected %d running header(s) across %d pages (threshold=%d): %s",
                len(headers),
                total_pages,
                threshold,
                list(headers)[:5],
            )

        return headers

    def _strip_running_headers(self, text: str, headers: frozenset) -> str:
        """Remove running header lines from a page's cleaned text.

        Lines whose stripped form is in the headers set are removed.
        Called per-page after _clean_text() in _clean_documents(), so
        it operates on already-cleaned text rather than raw extraction output.

        Args:
            text: Cleaned page text.
            headers: Frozenset of running header strings from _detect_running_headers.

        Returns:
            Text with running header lines removed.
        """
        if not headers:
            return text
        lines = text.split("\n")
        return "\n".join(line for line in lines if line.strip() not in headers)

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Apply cleaning to every page and return cleaned documents with no
        silent content loss.

        Pre-pass — running header detection:
            Before any per-page cleaning, scan all pages to identify lines that
            repeat verbatim across >= threshold pages (e.g. chapter titles,
            document headers). These are stripped from every page after cleaning
            to prevent them polluting embedding vectors for unrelated queries.

        Phase 1 — noise rejection (before length check):
            _clean_text() strips boilerplate (page numbers, standalone URLs,
            copyright lines) before we measure length. Pages that become empty
            after stripping are pure noise and are dropped silently.

        Phase 2 — short-page merging (replaces dropping):
            Pages that survive stripping but are below min_chars_per_page are NOT
            dropped. They are buffered and stitched into the next qualifying page.
            This preserves critical single-sentence content — a story climax, a
            chapter conclusion, a scene break — that would be silently lost by a
            naive length filter.

            Buffer flush rules:
                - On the next qualifying page: prepend buffer → clear buffer.
                - At end of document with buffered content: append to last kept page.
                - If the document has NO qualifying pages at all: keep short pages
                  as-is rather than lose them.

        Chapter/section headers ("Chapter 5", "BOOK FIVE: 1806–1807") also hit
        the short-page path. Merging them into the following page is beneficial —
        the chunk gains section context that improves retrieval precision.

        Args:
            documents: Raw Document objects from the loader, one per page.

        Returns:
            Cleaned Documents with all meaningful content preserved.
        """
        # Detect running headers cross-document before cleaning any page.
        # Headers are stripped after per-page cleaning so they don't pollute chunks.
        running_headers = self._detect_running_headers(documents)

        # Buffer stores (text, metadata) pairs so page provenance is never lost,
        # even in the edge case where the entire document consists of short pages.
        cleaned: List[Document] = []
        short_buffer: List[tuple] = []  # (cleaned_text, original_metadata)

        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)

            # Strip running headers identified across the full document
            if running_headers:
                cleaned_text = self._strip_running_headers(cleaned_text, running_headers)
                cleaned_text = cleaned_text.strip()

            # Phase 1: pure noise — empty after boilerplate stripping → drop
            if not cleaned_text:
                logger.info(
                    "Page empty after cleaning — dropped: source=%s, page=%s",
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                continue

            # Phase 2: real but short content → buffer for merge, never drop
            if len(cleaned_text) < self._min_chars_per_page:
                logger.info(
                    "Page too short (%d chars) — buffering for merge: source=%s, page=%s",
                    len(cleaned_text),
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                short_buffer.append((cleaned_text, doc.metadata))
                continue

            # Qualifying page — flush buffered short content by prepending it.
            # The merged page keeps the qualifying page's metadata (source, page number)
            # since that is the anchor page readers would navigate to.
            if short_buffer:
                buffered_text = "\n\n".join(text for text, _ in short_buffer)
                cleaned_text = buffered_text + "\n\n" + cleaned_text
                logger.debug(
                    "Merged %d buffered short page(s) into page=%s",
                    len(short_buffer),
                    doc.metadata.get("page", "?"),
                )
                short_buffer = []

            cleaned.append(Document(page_content=cleaned_text, metadata=doc.metadata))

        # Flush trailing buffer into the last kept page so end-of-document
        # content (climax lines, epilogue fragments) is never silently lost.
        if short_buffer:
            if cleaned:
                trailing_text = "\n\n".join(text for text, _ in short_buffer)
                cleaned[-1] = Document(
                    page_content=cleaned[-1].page_content + "\n\n" + trailing_text,
                    metadata=cleaned[-1].metadata,
                )
                logger.debug(
                    "Flushed %d trailing short page(s) into last kept page",
                    len(short_buffer),
                )
            else:
                # Entire document consists of short pages — keep each with its own
                # metadata rather than lose everything or collapse into one blob.
                logger.warning(
                    "All %d page(s) were short — keeping as individual documents "
                    "to avoid data loss",
                    len(short_buffer),
                )
                cleaned = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in short_buffer
                ]

        return cleaned
