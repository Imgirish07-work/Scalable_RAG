"""
Pipeline smoke test: chunking and embedding for all files in data/sample_docs.

Test scope:
    Smoke test (no pytest) for the document ingestion pipeline applied to real
    files. Processes each supported file (PDF, DOCX, TXT, MD, HTML) one at a
    time through DocumentCleaner → StructurePreserver → Chunker → Embeddings.

Flow:
    For each file: load/clean → structure detection → chunking → embed all
    chunks one by one. Unsupported extensions (e.g. .xlsx) are skipped.

Dependencies:
    BGE-small-en-v1.5 embedding model; files in data/sample_docs/; no external
    services (Qdrant, Redis, LLM API).
"""

import time
from pathlib import Path

from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from chunking.chunker import Chunker
from vectorstore.embeddings import get_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)

DOCS_DIR = Path("data/sample_docs")

# Extensions supported by DocumentCleaner
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def test_cleaner(filepath: Path, filename: str) -> list:
    """Load and clean the file via DocumentCleaner, returning a List[Document]."""
    separator(f"[1] DocumentCleaner — {filename}")

    cleaner = DocumentCleaner()
    start = time.monotonic()

    docs = cleaner.load_and_clean(str(filepath))

    elapsed = (time.monotonic() - start) * 1000

    total_chars = sum(len(d.page_content) for d in docs)

    print(f"  Pages loaded   : {len(docs)}")
    print(f"  Total chars    : {total_chars:,}")
    print(f"  Time           : {elapsed:.1f} ms")
    print(f"  Preview        : {docs[0].page_content[:200]!r}" if docs else "  No content")

    assert docs,        "DocumentCleaner returned no documents"
    assert total_chars, "DocumentCleaner returned empty content"

    logger.info("[PASS] DocumentCleaner | file=%s | pages=%d", filename, len(docs))
    return docs


def test_structure(docs: list, filename: str) -> list:
    """Run StructurePreserver on cleaned docs, returning List[Document] with enriched metadata."""
    separator(f"[2] StructurePreserver — {filename}")

    preserver = StructurePreserver()
    start = time.monotonic()

    structured_docs = preserver.preserve(docs)

    elapsed = (time.monotonic() - start) * 1000

    types = {}
    for d in structured_docs:
        t = d.metadata.get("structure_type", "unknown")
        types[t] = types.get(t, 0) + 1

    print(f"  Pages          : {len(structured_docs)}")
    print(f"  Structure types: {types}")
    print(f"  Time           : {elapsed:.1f} ms")

    assert structured_docs, "StructurePreserver returned no documents"

    logger.info("[PASS] StructurePreserver | file=%s | pages=%d", filename, len(structured_docs))
    return structured_docs


def test_chunker(structured_docs: list, filename: str) -> list:
    """Split structured docs into chunks via Chunker, returning List[Document]."""
    separator(f"[3] Chunker — {filename}")

    chunker = Chunker()
    start = time.monotonic()

    chunks = chunker.split_documents(structured_docs)

    elapsed = (time.monotonic() - start) * 1000
    stats = chunker.chunk_stats(chunks)

    print(f"  Total chunks   : {stats['count']}")
    print(f"  Avg chars      : {stats['avg_chars']}")
    print(f"  Min/Max chars  : {stats['min_chars']} / {stats['max_chars']}")
    print(f"  Avg tokens     : {stats['avg_tokens']}")
    print(f"  Min/Max tokens : {stats['min_tokens']} / {stats['max_tokens']}")
    print(f"  Time           : {elapsed:.1f} ms")

    assert chunks,           "Chunker returned no chunks"
    assert stats["count"] > 0, "chunk_stats count is 0"

    logger.info("[PASS] Chunker | file=%s | chunks=%d", filename, len(chunks))
    return chunks


# ...existing code...

def test_embeddings(chunks: list, filename: str) -> None:
    """Embed each chunk using the BGE model and assert non-empty vectors."""
    separator(f"[4] Embeddings — {filename}")

    embedder = get_embeddings()
    total = len(chunks)

    print(f"  Embedding {total} chunks one by one...")

    start_all = time.monotonic()

    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else chunk
        start = time.monotonic()

        embedding = embedder.embed_query(text)

        elapsed = (time.monotonic() - start) * 1000

        if i % 10 == 0 or i == total - 1:
            print(
                f"  [{i+1}/{total}] "
                f"dim={len(embedding)} | "
                f"time={elapsed:.1f}ms | "
                f"preview={text[:50]!r}"
            )
            print(
                f"  embedding[:10] : "
                f"{[round(v, 6) for v in embedding[:10]]}"
            )

        assert len(embedding) > 0, f"Empty embedding at chunk {i}"

    total_elapsed = (time.monotonic() - start_all) * 1000
    print(f"\n  Total embedding time : {total_elapsed:.1f} ms")

    logger.info(
        "[PASS] Embeddings | file=%s | chunks=%d | total_ms=%.1f",
        filename,
        total,
        total_elapsed,
    )

# ...existing code...


def process_document(filepath: Path) -> None:
    """Run all four pipeline steps for a single document and report pass/fail."""
    filename = filepath.name

    separator(f"DOCUMENT: {filename}")
    print(f"  Path : {filepath}")
    print(f"  Size : {filepath.stat().st_size / 1024:.1f} KB")

    try:
        docs          = test_cleaner(filepath, filename)
        structured_docs = test_structure(docs, filename)
        chunks        = test_chunker(structured_docs, filename)
        test_embeddings(chunks, filename)

        separator(f"✅ PASSED — {filename}")
        logger.info("All steps passed | file=%s", filename)

    except AssertionError as e:
        logger.error("[FAIL] Assertion | file=%s | error=%s", filename, e)
        print(f"\n  ❌ FAILED: {e}")

    except Exception as e:
        logger.error("[FAIL] Error | file=%s | error=%s", filename, e)
        print(f"\n  ❌ ERROR: {e}")


def main() -> None:
    docs = sorted(DOCS_DIR.glob("*.*"))

    if not docs:
        print(f"No files found in {DOCS_DIR}")
        return

    print(f"\nFound {len(docs)} document(s) in {DOCS_DIR}")
    for d in docs:
        print(f"  - {d.name} ({d.stat().st_size / 1024:.1f} KB)")

    for filepath in docs:
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"\n  ⚠️  SKIPPED — unsupported format: {filepath.name}")
            logger.warning("Skipping unsupported file | file=%s", filepath.name)
            continue

        process_document(filepath)
        print("\n")


if __name__ == "__main__":
    main()