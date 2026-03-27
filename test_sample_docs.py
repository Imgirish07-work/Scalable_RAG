"""
Minimal pipeline test — chunking to embeddings.
Processes one document at a time (large files).

Flow:
    raw file
        → DocumentCleaner   (load + clean)
        → StructurePreserver (detect structure)
        → Chunker            (split into chunks)
        → Embeddings         (embed each chunk)
"""

import time
from pathlib import Path

from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from chunking.chunker import Chunker
from vectorstore.embeddings import get_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

DOCS_DIR = Path("data/sample_docs")

# Supported by DocumentCleaner only
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


# ------------------------------------------------------------------ #
#  Step 1 — DocumentCleaner                                           #
# ------------------------------------------------------------------ #

def test_cleaner(filepath: Path, filename: str) -> list:
    """
    DocumentCleaner loads AND cleans the file.
    Pass filepath — NOT raw text.
    Returns List[Document].
    """
    separator(f"[1] DocumentCleaner — {filename}")

    cleaner = DocumentCleaner()
    start = time.monotonic()

    # ✅ correct method — takes file path, returns List[Document]
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


# ------------------------------------------------------------------ #
#  Step 2 — StructurePreserver                                        #
# ------------------------------------------------------------------ #

def test_structure(docs: list, filename: str) -> list:
    """
    StructurePreserver takes List[Document] — NOT raw text.
    Returns List[Document] with enriched metadata.
    """
    separator(f"[2] StructurePreserver — {filename}")

    preserver = StructurePreserver()
    start = time.monotonic()

    # ✅ correct — takes List[Document], returns List[Document]
    structured_docs = preserver.preserve(docs)

    elapsed = (time.monotonic() - start) * 1000

    # Count structure types found
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


# ------------------------------------------------------------------ #
#  Step 3 — Chunker                                                   #
# ------------------------------------------------------------------ #

def test_chunker(structured_docs: list, filename: str) -> list:
    """
    Chunker takes List[Document] — NOT raw text.
    Returns List[Document] chunks ready for embedding.
    """
    separator(f"[3] Chunker — {filename}")

    chunker = Chunker()
    start = time.monotonic()

    # ✅ correct — takes List[Document], returns List[Document]
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


# ------------------------------------------------------------------ #
#  Step 4 — Embeddings                                                #
# ------------------------------------------------------------------ #

# ...existing code...

def test_embeddings(chunks: list, filename: str) -> None:
    separator(f"[4] Embeddings — {filename}")

    embedder = get_embeddings()
    total = len(chunks)

    print(f"  Embedding {total} chunks one by one...")

    start_all = time.monotonic()

    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else chunk
        start = time.monotonic()

        # ✅ correct method on HuggingFaceEmbeddings
        embedding = embedder.embed_query(text)

        elapsed = (time.monotonic() - start) * 1000

        # Print progress every 10 chunks
        if i % 10 == 0 or i == total - 1:
            print(
                f"  [{i+1}/{total}] "
                f"dim={len(embedding)} | "
                f"time={elapsed:.1f}ms | "
                f"preview={text[:50]!r}"
            )
            # ✅ Print first 10 embedding values only
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

# ------------------------------------------------------------------ #
#  Main — Process one doc at a time                                   #
# ------------------------------------------------------------------ #

def process_document(filepath: Path) -> None:
    """Full pipeline for a single document."""
    filename = filepath.name

    separator(f"DOCUMENT: {filename}")
    print(f"  Path : {filepath}")
    print(f"  Size : {filepath.stat().st_size / 1024:.1f} KB")

    try:
        # Step 1 — load + clean (pass filepath not raw text)
        docs = test_cleaner(filepath, filename)

        # Step 2 — structure detection (pass List[Document])
        structured_docs = test_structure(docs, filename)

        # Step 3 — chunking (pass List[Document])
        chunks = test_chunker(structured_docs, filename)

        # Step 4 — embeddings (pass List[Document])
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
        # ✅ Skip unsupported files (xlsx etc.)
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"\n  ⚠️  SKIPPED — unsupported format: {filepath.name}")
            logger.warning("Skipping unsupported file | file=%s", filepath.name)
            continue

        process_document(filepath)
        print("\n")


if __name__ == "__main__":
    main()