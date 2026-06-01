"""Loads documents from disk and cleans extracted text for the ingestion pipeline."""

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
    """Loads documents from disk and cleans extracted text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

    _BOILERPLATE_PATTERNS = [
        r"all rights reserved",
        r"confidential",
        r"do not distribute",
        r"page\s+\d+\s+of\s+\d+",
        r"^\s*\d+\s*$",
        r"^\s*http[s]?://[^\s]+\s*$",
        r"^\s*www\.[^\s]+\s*$",
        r"^\s*[a-zA-Z]\s*$",
        r"^\s*\[[a-zA-Z][\\'\]]*\s*$",
    ]

    _BOILERPLATE_REGEX = re.compile(
        "|".join(_BOILERPLATE_PATTERNS),
        re.IGNORECASE | re.MULTILINE,
    )

    _RUNNING_HEADER_MIN_PAGES: int = 3
    _RUNNING_HEADER_PAGE_FRACTION: float = 0.15
    _RUNNING_HEADER_MIN_CHARS: int = 6
    _RUNNING_HEADER_MIN_WORDS: int = 2
    _RUNNING_HEADER_MAX_WORDS: int = 6
    _RUNNING_HEADER_MAX_CHARS: int = 80

    _OCR_BACKSLASH_RE = re.compile(r"(?<=[a-zA-Z0-9])\\(?=[a-zA-Z0-9])")

    _NOISY_SYMBOL_RE = re.compile(r"[^a-zA-Z0-9\s.,!?;:\-'\"()\u00C0-\u024F]")

    _NOISE_CHECK_MIN_LEN: int = 5
    _NOISE_CHECK_MAX_LEN: int = 100
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
        """Load a document from disk and return cleaned pages."""
        file_name = Path(file_path).name
        logger.info("Loading document: %s", file_name)

        docs = self._load_document(file_path)
        cleaned_docs = self._clean_documents(docs)

        logger.info(
            "Loaded %d pages, cleaned to %d pages", len(docs), len(cleaned_docs)
        )
        return cleaned_docs

    def _detect_type(self, file_path: str) -> str:
        """Return the lowercase file extension, raising if unsupported."""
        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.debug("Detected file type: %s for %s", ext, file_path)
        return ext

    def _load_document(self, file_path: str) -> List[Document]:
        """Select the correct LangChain loader and load the document."""
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
                raise ValueError(f"No loader for extension: {ext}")

            logger.info("Loaded %d page(s): file=%s", len(docs), file_name)
            return docs

        except Exception as e:
            logger.exception("Failed to load: file=%s, error=%s", file_name, e)
            raise

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF using the preferred loader, with automatic fallback to PDFPlumber."""
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
        """Apply multi-step cleaning to raw extracted text from a single page."""
        if not text or not text.strip():
            logger.warning("Received empty or whitespace-only text for cleaning")
            return ""

        try:
            cleaned = ftfy.fix_text(text)

            cleaned = self._remove_ocr_artifacts(cleaned)

            cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

            cleaned = self._BOILERPLATE_REGEX.sub("", cleaned)

            cleaned = self._filter_noisy_lines(cleaned)

            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)

            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.exception("Error cleaning text: %s", e)
            return ""

    def _remove_ocr_artifacts(self, text: str) -> str:
        """Remove backslash OCR artifacts between word characters."""
        return self._OCR_BACKSLASH_RE.sub("", text)

    def _filter_noisy_lines(self, text: str) -> str:
        """Drop lines whose noisy-symbol ratio exceeds the configured threshold."""
        lines = text.split("\n")
        filtered = []
        for line in lines:
            length = len(line)
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
        """Identify running headers and footers that repeat verbatim across pages."""
        if not documents:
            return frozenset()

        total_pages = len(documents)
        threshold = max(
            self._RUNNING_HEADER_MIN_PAGES,
            int(self._RUNNING_HEADER_PAGE_FRACTION * total_pages),
        )

        line_page_count: dict = {}
        for doc in documents:
            page_lines: set = set()
            for line in doc.page_content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if len(stripped) < self._RUNNING_HEADER_MIN_CHARS:
                    continue
                if len(stripped.split()) < self._RUNNING_HEADER_MIN_WORDS:
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
        """Remove running header lines from a page's cleaned text."""
        if not headers:
            return text
        lines = text.split("\n")
        return "\n".join(line for line in lines if line.strip() not in headers)

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Apply cleaning to every page and return cleaned documents with no silent content loss."""
        running_headers = self._detect_running_headers(documents)
        cleaned: List[Document] = []
        short_buffer: List[tuple] = []

        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)

            if running_headers:
                cleaned_text = self._strip_running_headers(cleaned_text, running_headers)
                cleaned_text = cleaned_text.strip()

            if not cleaned_text:
                logger.info(
                    "Page empty after cleaning — dropped: source=%s, page=%s",
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                continue

            if len(cleaned_text) < self._min_chars_per_page:
                logger.info(
                    "Page too short (%d chars) — buffering for merge: source=%s, page=%s",
                    len(cleaned_text),
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                short_buffer.append((cleaned_text, doc.metadata))
                continue

            # flush buffered short pages by prepending; keep this page's metadata as anchor
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
