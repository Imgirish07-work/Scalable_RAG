"""
Document loader and cleaner — first step in the ingestion pipeline.

Loads documents from disk (PDF, DOCX, TXT, MD, HTML), cleans raw text
by fixing encoding issues, removing boilerplate, and normalizing whitespace.
Filters out pages that are too short or empty after cleaning.

Pipeline position:
    DocumentCleaner.load_and_clean()         ← here
        → StructurePreserver.preserve()
            → Chunker.split_documents()
                → VectorStore.add_documents()

Supported formats:
    .pdf  → PyMuPDFLoader (default) or PDFPlumberLoader (settings toggle)
    .docx → Docx2txtLoader
    .txt  → TextLoader (utf-8)
    .md   → UnstructuredMarkdownLoader
    .html → UnstructuredHTMLLoader

Sync — file I/O only, LangChain loaders are sync-only.
Wrapped in asyncio.to_thread() when called from async context.
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
        _min_chars_per_page: Minimum characters to keep a page (filters noise).
        _prefer_pdfplumber: If True, use PDFPlumberLoader instead of PyMuPDF.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

    # Boilerplate patterns for removal during cleaning
    # Only standalone URL lines are stripped (not inline URLs in content)
    _BOILERPLATE_PATTERNS = [
        r"all rights reserved",
        r"confidential",
        r"do not distribute",
        r"page\s+\d+\s+of\s+\d+",       # "Page 1 of 10"
        r"^\s*\d+\s*$",                   # standalone page numbers
        r"^\s*http[s]?://[^\s]+\s*$",     # standalone URL lines (not inline)
        r"^\s*www\.[^\s]+\s*$",           # standalone www lines (not inline)
    ]

    _BOILERPLATE_REGEX = re.compile(
        "|".join(_BOILERPLATE_PATTERNS),
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self) -> None:
        self._min_chars_per_page: int = settings.min_chars_per_page
        self._prefer_pdfplumber: bool = settings.prefer_pdfplumber

        logger.debug(
            "DocumentCleaner initialized: min_chars=%d, prefer_pdfplumber=%s",
            self._min_chars_per_page,
            self._prefer_pdfplumber,
        )

    def load_and_clean(self, file_path: str) -> List[Document]:
        """Load a document from disk and clean all pages.

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
        """Detect file type from extension.

        Args:
            file_path: Path to the file.

        Returns:
            Lowercase file extension (e.g. '.pdf').

        Raises:
            ValueError: If extension is not in SUPPORTED_EXTENSIONS.
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.debug("Detected file type: %s for %s", ext, file_path)
        return ext

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document using the correct LangChain loader per file type.

        Loader selection:
            .pdf   → PyMuPDFLoader (default, fastest, best layout)
                     PDFPlumberLoader (if prefer_pdfplumber=True)
                     PDFPlumberLoader (fallback if PyMuPDF fails)
            .docx  → Docx2txtLoader
            .txt   → TextLoader (utf-8)
            .md    → UnstructuredMarkdownLoader
            .html  → UnstructuredHTMLLoader

        Args:
            file_path: Path to the document.

        Returns:
            List of raw Document objects (one per page/section).

        Raises:
            FileNotFoundError: If file does not exist.
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
                # Defensive guard — _detect_type already validates
                raise ValueError(f"No loader for extension: {ext}")

            logger.info("Loaded %d page(s): file=%s", len(docs), file_name)
            return docs

        except Exception as e:
            logger.exception("Failed to load: file=%s, error=%s", file_name, e)
            raise

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF with loader preference from settings.

        Primary:   PyMuPDFLoader (fast, layout-aware)
        Alternate: PDFPlumberLoader (MIT license, better tables)
        Fallback:  PDFPlumberLoader (if primary fails)

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects (one per page).
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
        """Clean raw extracted text from a single page.

        Steps (in order):
            1. ftfy — fix broken unicode, mojibake, encoding issues
            2. Hyphens — fix hyphenated line breaks "retriev-\\nal" → "retrieval"
            3. Boilerplate — remove standalone page numbers, URLs, copyright
            4. Whitespace — normalize multiple newlines, collapse spaces
            5. Strip — remove leading/trailing whitespace

        Args:
            text: Raw text extracted from a document page.

        Returns:
            Cleaned text string. Empty string if input was empty/whitespace.
        """
        if not text or not text.strip():
            logger.warning("Received empty or whitespace-only text for cleaning")
            return ""

        try:
            # Fix encoding issues (mojibake, broken unicode)
            cleaned = ftfy.fix_text(text)

            # Fix hyphenated line breaks: "retriev-\nal" → "retrieval"
            cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

            # Remove boilerplate (standalone URLs, page numbers, copyright)
            cleaned = self._BOILERPLATE_REGEX.sub("", cleaned)

            # Normalize whitespace: collapse 3+ newlines to 2, collapse spaces
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)

            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.exception("Error cleaning text: %s", e)
            return ""

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Apply _clean_text() to every document page.

        Filters:
            - Skips pages that are empty after cleaning
            - Skips pages shorter than min_chars_per_page
            - Preserves all original metadata

        Args:
            documents: List of raw Document objects from loader.

        Returns:
            List of cleaned Document objects.
        """
        cleaned = []

        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)

            # Skip empty pages
            if not cleaned_text:
                logger.warning(
                    "Page empty after cleaning — skipping: source=%s, page=%s",
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                continue

            # Skip pages below minimum length
            if len(cleaned_text) < self._min_chars_per_page:
                logger.warning(
                    "Page too short (%d chars) — skipping: source=%s, page=%s",
                    len(cleaned_text),
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                continue

            cleaned.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )

        return cleaned