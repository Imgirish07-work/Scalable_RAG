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
            1. ftfy — fix broken unicode and mojibake encoding issues.
            2. Hyphens — rejoin hyphenated line breaks ('retriev-\\nal' → 'retrieval').
            3. Boilerplate — remove page numbers, standalone URLs, copyright lines.
            4. Whitespace — collapse 3+ newlines to 2, collapse repeated spaces.
            5. Strip — remove leading and trailing whitespace.

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

            # Rejoin hyphenated line breaks introduced by PDF line wrapping
            cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

            # Remove boilerplate: page numbers, standalone URLs, copyright notices
            cleaned = self._BOILERPLATE_REGEX.sub("", cleaned)

            # Normalize whitespace: collapse excess newlines and spaces
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)

            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.exception("Error cleaning text: %s", e)
            return ""

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Apply _clean_text() to every page and filter low-quality results.

        Filters applied:
            - Skips pages that are empty after cleaning.
            - Skips pages shorter than min_chars_per_page.
            - Preserves all original metadata on kept pages.

        Args:
            documents: List of raw Document objects from the loader.

        Returns:
            List of cleaned Document objects.
        """
        cleaned = []

        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)

            if not cleaned_text:
                logger.info(
                    "Page empty after cleaning — skipping: source=%s, page=%s",
                    doc.metadata.get("source", "unknown"),
                    doc.metadata.get("page", "?"),
                )
                continue

            if len(cleaned_text) < self._min_chars_per_page:
                logger.info(
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
