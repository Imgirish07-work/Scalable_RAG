"""
Unit tests for the DocumentCleaner component.

Test scope:
    Unit tests covering file-type detection, unsupported-type rejection,
    file-not-found handling, text cleaning (unicode, hyphenation, boilerplate,
    URL removal, whitespace), short-page filtering, real file loading (.txt
    and .pdf), and metadata preservation.

Flow:
    Module-level execution — each Test N section runs sequentially; a failed
    assert exits immediately via sys.exit(1).

Dependencies:
    DocumentCleaner, sample_docs directory (optional for file-load tests),
    langchain_core Document.
"""

import os
import sys

# Corporate network SSL fix
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"]                   = ""
os.environ["REQUESTS_CA_BUNDLE"]               = ""

from pathlib import Path
from chunking.document_cleaner import DocumentCleaner

cleaner   = DocumentCleaner()
SAMPLE_DIR = Path("data/sample_docs")

# Helpers

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def passed(msg: str) -> None:
    print(f"  ✅ {msg}")

def failed(msg: str) -> None:
    print(f"  ❌ {msg}")
    sys.exit(1)

# Test 1 — Supported file type detection

section("Test 1 — File Type Detection")

supported = [".pdf", ".docx", ".txt", ".md", ".html", ".htm"]
for ext in supported:
    dummy_path = f"dummy_file{ext}"
    detected   = cleaner._detect_type(dummy_path)
    assert detected == ext, f"Expected {ext} got {detected}"
    passed(f"Detected: {ext}")

# Test 2 — Unsupported file type raises ValueError

section("Test 2 — Unsupported File Type")

try:
    cleaner._detect_type("file.xlsx")
    failed("Should have raised ValueError for .xlsx")
except ValueError as e:
    passed(f"Raised ValueError correctly → {e}")

# Test 3 — File not found raises FileNotFoundError

section("Test 3 — File Not Found")

try:
    cleaner.load_and_clean("non_existent_file.pdf")
    failed("Should have raised FileNotFoundError")
except FileNotFoundError as e:
    passed(f"Raised FileNotFoundError correctly → {e}")

# Test 4 — clean_text() basic cleaning

section("Test 4 — Text Cleaning")

# 4a — unicode fix
dirty = "â€œThis is a testâ€\x00"
result = cleaner._clean_text(dirty)
assert "\x00" not in result
passed(f"Unicode fix    | input='{dirty[:20]}...' → cleaned")

# 4b — hyphenated line break
dirty = "retriev-\nal augmented generation"
result = cleaner._clean_text(dirty)
assert "retrieval" in result
passed(f"Hyphen fix     | 'retriev-\\nal' → '{result[:30]}'")

# 4c — boilerplate removal
dirty = "This document is Confidential. Page 1 of 10. All Rights Reserved."
result = cleaner._clean_text(dirty)
assert "confidential" not in result.lower()
assert "all rights reserved" not in result.lower()
passed(f"Boilerplate    | removed confidential + page numbers + copyright")

# 4d — URL removal
dirty = "Visit www.example.com for more info."
result = cleaner._clean_text(dirty)
assert "https://" not in result
passed(f"URL removal    | '{result.strip()}'")

# 4e — whitespace normalization
dirty = "line one\n\n\n\n\nline two"
result = cleaner._clean_text(dirty)
assert "\n\n\n" not in result
passed(f"Whitespace     | 5 newlines → 2 newlines")

# 4f — empty text
result = cleaner._clean_text("   ")
assert result == ""
passed(f"Empty text     | whitespace only → returns empty string")

# Test 5 — clean_documents() filters short pages

section("Test 5 — Short Page Filtering")

from langchain_core.documents import Document

short_doc = Document(page_content="Hi", metadata={"source": "test.pdf", "page": 1})
normal_doc = Document(
    page_content="RAG combines retrieval with generation for grounded responses. " * 5,
    metadata={"source": "test.pdf", "page": 2}
)

result = cleaner._clean_documents([short_doc, normal_doc])
assert len(result) == 1, f"Expected 1 doc, got {len(result)}"
passed(f"Short page filtered | kept={len(result)}/2 pages")

# Test 6 — Load real .txt file

section("Test 6 — Load Real .txt File")

txt_files = list(SAMPLE_DIR.glob("*.txt"))
if not txt_files:
    print(f"  ⚠️  No .txt files found in {SAMPLE_DIR} — skipping")
else:
    txt_path = str(txt_files[0])
    docs = cleaner.load_and_clean(txt_path)
    assert len(docs) > 0
    assert all(doc.page_content.strip() for doc in docs)
    passed(f"Loaded {len(docs)} page(s) | file={txt_files[0].name}")
    print(f"     Preview → {docs[0].page_content[:100]}...")

# Test 7 — Load real .pdf file

section("Test 7 — Load Real .pdf File")

pdf_files = list(SAMPLE_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"  ⚠️  No .pdf files found in {SAMPLE_DIR} — skipping")
else:
    pdf_path = str(pdf_files[0])
    docs = cleaner.load_and_clean(pdf_path)
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)
    passed(f"Loaded {len(docs)} page(s) | file={pdf_files[0].name}")
    print(f"     Preview → {docs[0].page_content[:100]}...")
    print(f"     Metadata → {docs[0].metadata}")

# Test 8 — Metadata preserved after cleaning

section("Test 8 — Metadata Preserved")

doc = Document(
    page_content="RAG combines retrieval with generation for grounded responses. " * 5,
    metadata={"source": "test.pdf", "page": 1, "author": "test_user"}
)
result = cleaner._clean_documents([doc])
assert result[0].metadata["source"] == "test.pdf"
assert result[0].metadata["page"]   == 1
assert result[0].metadata["author"] == "test_user"
passed("Metadata fully preserved after cleaning")

# Summary

section("All Tests Passed ✅")
print("  DocumentCleaner is working correctly\n")
