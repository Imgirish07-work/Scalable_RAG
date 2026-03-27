"""
Vector Store Pipeline Test — full ingestion + retrieval validation.

Run from project root:
    python test_vectorstore_pipeline.py

Prerequisites:
    - BGE-small-en-v1.5 model configured in .env
    - qdrant-client installed
    - No Qdrant server needed (uses in-memory mode)

Tests the complete pipeline:
    Document creation → Chunker metadata → QdrantStore write → search → restore

Sections:
    1.  Embedding model + dimension check
    2.  QdrantStore initialization (in-memory)
    3.  Document ingestion — embed_content swap
    4.  Similarity search — basic retrieval
    5.  Similarity search — score threshold filtering
    6.  Similarity search — relevance score in metadata
    7.  Original content restoration (no embed prefix in results)
    8.  User ID filtering
    9.  Multiple documents — correct retrieval
    10. Collection stats
    11. Empty query and edge cases
    12. Delete collection
    13. Graceful close
"""
import os
import ssl

# Corporate network SSL Fix — disables SSL verification for HuggingFace downloads
# (SPLADE model for hybrid mode). Remove once corporate CA cert is installed.
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context


import asyncio
import time
import sys
from datetime import datetime

from colorama import Fore, Style, init as colorama_init
from langchain_core.documents import Document

from vectorstore.embeddings import get_embeddings, get_embedding_dimension
from vectorstore.qdrant_store import QdrantStore

colorama_init(autoreset=True)

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
MODEL_AVAILABLE = False


def header(title: str, section: int) -> None:
    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"  SECTION {section} — {title}")
    print(f"{'═' * 70}{Style.RESET_ALL}")


def sub_header(title: str) -> None:
    print(f"\n{Fore.YELLOW}  ▸ {title}{Style.RESET_ALL}")


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
    else:
        FAIL_COUNT += 1
        status = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
    msg = f"    {status}  {label}"
    if detail:
        msg += f"  {Fore.WHITE}({detail}){Style.RESET_ALL}"
    print(msg)


def skip(label: str, reason: str = "Model not available") -> None:
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"    {Fore.YELLOW}⊘ SKIP{Style.RESET_ALL}  {label}  {Fore.WHITE}({reason}){Style.RESET_ALL}")


def info(msg: str) -> None:
    print(f"    {Fore.WHITE}ℹ {msg}{Style.RESET_ALL}")


# Sample documents that simulate Chunker output
def make_sample_documents() -> list[Document]:
    """Create sample documents with metadata matching Chunker output.

    Each document has embed_content (title + section + text) and
    page_content (clean text). This mimics what the Chunker produces
    after _prepend_context().
    """
    docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with language model generation to ground answers in real documents.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 1,
                "section": "Introduction",
                "structure_type": "paragraph",
                "chunk_index": 0,
                "total_chunks": 5,
                "word_count": 22,
                "token_count": 28,
                "doc_type": "pdf",
                "doc_id": "",
                "user_id": "user_001",
                "embed_content": "Title: ml_guide.pdf | Section: Introduction\nRAG stands for Retrieval-Augmented Generation. It combines retrieval with language model generation to ground answers in real documents.",
            },
        ),
        Document(
            page_content="Transformers use self-attention mechanisms to process sequences in parallel. The attention mechanism computes weighted sums of value vectors based on query-key similarity.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 2,
                "section": "Transformers",
                "structure_type": "paragraph",
                "chunk_index": 1,
                "total_chunks": 5,
                "word_count": 26,
                "token_count": 32,
                "doc_type": "pdf",
                "doc_id": "",
                "user_id": "user_001",
                "embed_content": "Title: ml_guide.pdf | Section: Transformers\nTransformers use self-attention mechanisms to process sequences in parallel. The attention mechanism computes weighted sums of value vectors based on query-key similarity.",
            },
        ),
        Document(
            page_content="Vector databases store high-dimensional embeddings and support fast approximate nearest neighbor search. Qdrant uses HNSW graphs for efficient similarity lookup.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 3,
                "section": "Vector Databases",
                "structure_type": "paragraph",
                "chunk_index": 2,
                "total_chunks": 5,
                "word_count": 24,
                "token_count": 30,
                "doc_type": "pdf",
                "doc_id": "",
                "user_id": "user_001",
                "embed_content": "Title: ml_guide.pdf | Section: Vector Databases\nVector databases store high-dimensional embeddings and support fast approximate nearest neighbor search. Qdrant uses HNSW graphs for efficient similarity lookup.",
            },
        ),
        Document(
            page_content="Fine-tuning adapts a pre-trained model to a specific task by training on a small labeled dataset. LoRA reduces memory usage by training low-rank adapter matrices.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 4,
                "section": "Fine-tuning",
                "structure_type": "paragraph",
                "chunk_index": 3,
                "total_chunks": 5,
                "word_count": 26,
                "token_count": 34,
                "doc_type": "pdf",
                "doc_id": "",
                "user_id": "user_002",
                "embed_content": "Title: ml_guide.pdf | Section: Fine-tuning\nFine-tuning adapts a pre-trained model to a specific task by training on a small labeled dataset. LoRA reduces memory usage by training low-rank adapter matrices.",
            },
        ),
        Document(
            page_content="Python is a high-level programming language known for its readability and extensive library ecosystem. It is widely used in data science, web development, and machine learning.",
            metadata={
                "source": "python_guide.pdf",
                "page": 1,
                "section": "Overview",
                "structure_type": "paragraph",
                "chunk_index": 0,
                "total_chunks": 3,
                "word_count": 28,
                "token_count": 30,
                "doc_type": "pdf",
                "doc_id": "",
                "user_id": "user_002",
                "embed_content": "Title: python_guide.pdf | Section: Overview\nPython is a high-level programming language known for its readability and extensive library ecosystem. It is widely used in data science, web development, and machine learning.",
            },
        ),
    ]
    return docs


# ══════════════════════════════════════════════════
# SECTION 1 — Embedding model check
# ══════════════════════════════════════════════════


def test_embedding_model() -> None:
    global MODEL_AVAILABLE
    header("Embedding Model + Dimension", 1)

    sub_header("Load BGE model via get_embeddings()")

    try:
        start = time.perf_counter()
        embeddings = get_embeddings()
        elapsed = time.perf_counter() - start

        MODEL_AVAILABLE = True
        check("Model loaded", True, f"{elapsed:.2f}s")

        dim = get_embedding_dimension()
        check("Dimension is 384", dim == 384, f"dim={dim}")

        # Verify embedding works
        vector = embeddings.embed_query("test query")
        check("embed_query returns list", isinstance(vector, list))
        check("Vector has correct dimension", len(vector) == dim)
        check("Vector is non-zero", any(v != 0.0 for v in vector))

    except Exception as e:
        MODEL_AVAILABLE = False
        check("Model loaded", False, str(e))


# ══════════════════════════════════════════════════
# SECTION 2 — QdrantStore initialization
# ══════════════════════════════════════════════════


async def test_initialization() -> None:
    header("QdrantStore Initialization (in-memory)", 2)

    if not MODEL_AVAILABLE:
        skip("Initialization tests")
        return

    sub_header("Dense mode")

    store = QdrantStore(
        collection_name="test_dense",
        in_memory=True,
        search_mode="dense",
    )
    await store.initialize()

    check("Dense store initialized", store._client is not None)
    check("Dense store built", store._store is not None)
    check("Collection name correct", store.collection_name == "test_dense")
    check("Search mode is dense", store.search_mode == "dense")

    await store.close()

    sub_header("Hybrid mode")

    store_hybrid = QdrantStore(
        collection_name="test_hybrid",
        in_memory=True,
        search_mode="hybrid",
    )
    try:
        await store_hybrid.initialize()
        check("Hybrid store initialized", store_hybrid._client is not None)
        check("Hybrid search mode", store_hybrid.search_mode == "hybrid")
        await store_hybrid.close()
    except Exception as e:
        skip("Hybrid store initialized", f"SPLADE model unavailable: {e.__class__.__name__}")
        skip("Hybrid search mode", "skipped — hybrid init failed")

    sub_header("Double initialize is safe")

    store2 = QdrantStore(collection_name="test_idempotent", in_memory=True)
    await store2.initialize()
    await store2.initialize()

    check("Double initialize — no error", True)
    await store2.close()


# ══════════════════════════════════════════════════
# SECTION 3 — Document ingestion + embed_content
# ══════════════════════════════════════════════════


async def test_ingestion() -> None:
    header("Document Ingestion — embed_content Swap", 3)

    if not MODEL_AVAILABLE:
        skip("Ingestion tests")
        return

    store = QdrantStore(collection_name="test_ingestion", in_memory=True)
    await store.initialize()

    docs = make_sample_documents()

    sub_header("Add documents")

    ids = await store.add_documents(docs)
    check("Returns IDs", len(ids) == len(docs), f"ids={len(ids)}")
    check("IDs are strings", all(isinstance(i, str) for i in ids))

    sub_header("Verify embed_content was used for embedding")

    # If embed_content swap works, searching for "Title: ml_guide.pdf"
    # should return results (because embed_content includes the title)
    results = await store.similarity_search("ml_guide Introduction RAG", k=1)
    check("Search returns results", len(results) > 0)

    if results:
        # page_content should be the ORIGINAL text, not embed_content
        first = results[0]
        has_prefix = first.page_content.startswith("Title:")
        check(
            "page_content is clean (no Title: prefix)",
            not has_prefix,
            f"starts_with='Title:': {has_prefix}",
        )

    sub_header("Verify original_content in metadata")

    # Retrieve raw from Qdrant to check metadata
    raw_results = await asyncio.to_thread(
        store._store.similarity_search, "RAG retrieval augmented", k=1
    )
    if raw_results:
        meta = raw_results[0].metadata
        check(
            "original_content stored in metadata",
            "original_content" in meta,
        )
        if "original_content" in meta:
            check(
                "original_content matches source doc",
                meta["original_content"] == docs[0].page_content,
            )

    sub_header("Verify enriched metadata")

    if results:
        meta = results[0].metadata
        check("Has doc_id", "doc_id" in meta)
        check("Has user_id", "user_id" in meta)
        check("Has source", "source" in meta)
        check("Has ingested_at", "ingested_at" in meta)
        check("Has char_count", "char_count" in meta and meta["char_count"] > 0)
        info(f"Metadata keys: {list(meta.keys())}")

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 4 — Basic similarity search
# ══════════════════════════════════════════════════


async def test_basic_search() -> None:
    header("Similarity Search — Basic Retrieval", 4)

    if not MODEL_AVAILABLE:
        skip("Search tests")
        return

    store = QdrantStore(collection_name="test_search", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Semantic search — relevant query")

    results = await store.similarity_search("What is RAG?", k=3)
    check("Returns results", len(results) > 0, f"count={len(results)}")
    check("Returns <= k results", len(results) <= 3)

    if results:
        # RAG document should be most relevant
        top_content = results[0].page_content
        check(
            "Top result is about RAG",
            "RAG" in top_content or "retrieval" in top_content.lower(),
            f"content='{top_content[:60]}...'",
        )

    sub_header("Semantic search — different topic")

    results2 = await store.similarity_search("How does attention work in transformers?", k=2)
    check("Returns results for transformer query", len(results2) > 0)

    if results2:
        top = results2[0].page_content
        check(
            "Top result is about transformers/attention",
            "attention" in top.lower() or "transformer" in top.lower(),
            f"content='{top[:60]}...'",
        )

    sub_header("k parameter respected")

    results_k1 = await store.similarity_search("machine learning", k=1)
    results_k5 = await store.similarity_search("machine learning", k=5)
    check("k=1 returns 1 result", len(results_k1) == 1)
    check("k=5 returns all 5 docs", len(results_k5) == 5)

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 5 — Score threshold filtering
# ══════════════════════════════════════════════════


async def test_score_threshold() -> None:
    header("Similarity Search — Score Threshold", 5)

    if not MODEL_AVAILABLE:
        skip("Threshold tests")
        return

    store = QdrantStore(collection_name="test_threshold", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Low threshold returns more results")

    results_low = await store.similarity_search(
        "machine learning techniques", k=5, score_threshold=0.3
    )
    check(
        "Low threshold (0.3) returns results",
        len(results_low) > 0,
        f"count={len(results_low)}",
    )

    sub_header("High threshold returns fewer results")

    results_high = await store.similarity_search(
        "machine learning techniques", k=5, score_threshold=0.8
    )
    check(
        "High threshold (0.8) returns fewer/no results",
        len(results_high) <= len(results_low),
        f"low={len(results_low)}, high={len(results_high)}",
    )

    sub_header("Very high threshold may return empty")

    results_extreme = await store.similarity_search(
        "completely unrelated query about cooking pasta", k=5, score_threshold=0.9
    )
    check(
        "Extreme threshold (0.9) on unrelated query → few/no results",
        len(results_extreme) <= 1,
        f"count={len(results_extreme)}",
    )

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 6 — Relevance score in metadata
# ══════════════════════════════════════════════════


async def test_relevance_scores() -> None:
    header("Relevance Score in Metadata", 6)

    if not MODEL_AVAILABLE:
        skip("Score metadata tests")
        return

    store = QdrantStore(collection_name="test_scores", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Scores attached when threshold is provided")

    results = await store.similarity_search(
        "What is RAG?", k=3, score_threshold=0.3
    )

    if results:
        first = results[0]
        has_score = "relevance_score" in first.metadata
        check("relevance_score in metadata", has_score)

        if has_score:
            score = first.metadata["relevance_score"]
            check("Score is a float", isinstance(score, float))
            check("Score is between 0 and 1", 0.0 <= score <= 1.0, f"score={score}")
            info(f"Top result score: {score}")

        # Scores should be descending
        if len(results) >= 2:
            scores = [r.metadata.get("relevance_score", 0) for r in results]
            is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            check("Scores are in descending order", is_sorted, f"scores={scores}")
    else:
        check("Results returned for score test", False, "no results")

    sub_header("No scores when threshold is None")

    results_no_threshold = await store.similarity_search("What is RAG?", k=1)
    if results_no_threshold:
        has_score = "relevance_score" in results_no_threshold[0].metadata
        check(
            "No relevance_score when threshold=None",
            not has_score,
        )

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 7 — Original content restoration
# ══════════════════════════════════════════════════


async def test_content_restoration() -> None:
    header("Original Content Restoration", 7)

    if not MODEL_AVAILABLE:
        skip("Content restoration tests")
        return

    store = QdrantStore(collection_name="test_restore", in_memory=True)
    await store.initialize()

    docs = make_sample_documents()
    await store.add_documents(docs)

    sub_header("Retrieved content is clean (no embed prefix)")

    results = await store.similarity_search("RAG retrieval augmented", k=1)

    if results:
        content = results[0].page_content

        # Should NOT start with "Title:" prefix
        check(
            "No 'Title:' prefix in page_content",
            not content.startswith("Title:"),
            f"starts='{content[:30]}...'",
        )

        # Should match one of the original documents
        original_contents = [d.page_content for d in docs]
        check(
            "Content matches an original document",
            content in original_contents,
        )

        # embed_content should still be in metadata (for reference)
        meta = results[0].metadata
        check(
            "embed_content still in metadata",
            "embed_content" in meta,
        )

        if "embed_content" in meta:
            check(
                "embed_content starts with Title:",
                meta["embed_content"].startswith("Title:"),
            )
    else:
        check("Results returned", False, "no results")

    sub_header("All results have clean content")

    all_results = await store.similarity_search("machine learning", k=5)
    clean_count = sum(
        1 for r in all_results if not r.page_content.startswith("Title:")
    )
    check(
        "All results have clean page_content",
        clean_count == len(all_results),
        f"clean={clean_count}/{len(all_results)}",
    )

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 8 — User ID filtering
# ══════════════════════════════════════════════════


async def test_user_filtering() -> None:
    header("User ID Filtering", 8)

    if not MODEL_AVAILABLE:
        skip("User filtering tests")
        return

    store = QdrantStore(collection_name="test_filter", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Filter by user_001")

    results_u1 = await store.similarity_search(
        "machine learning", k=10, filter_user_id="user_001"
    )
    all_u1 = all(r.metadata.get("user_id") == "user_001" for r in results_u1)
    check(
        "All results belong to user_001",
        all_u1 and len(results_u1) > 0,
        f"count={len(results_u1)}",
    )

    sub_header("Filter by user_002")

    results_u2 = await store.similarity_search(
        "machine learning", k=10, filter_user_id="user_002"
    )
    all_u2 = all(r.metadata.get("user_id") == "user_002" for r in results_u2)
    check(
        "All results belong to user_002",
        all_u2 and len(results_u2) > 0,
        f"count={len(results_u2)}",
    )

    sub_header("Users see different result counts")

    check(
        "user_001 and user_002 have different counts",
        len(results_u1) != len(results_u2),
        f"u1={len(results_u1)}, u2={len(results_u2)}",
    )

    sub_header("Nonexistent user returns empty")

    results_none = await store.similarity_search(
        "machine learning", k=10, filter_user_id="user_999"
    )
    check("Unknown user → empty results", len(results_none) == 0)

    sub_header("No filter returns all documents")

    results_all = await store.similarity_search("machine learning", k=10)
    check(
        "No filter returns more than either user alone",
        len(results_all) >= max(len(results_u1), len(results_u2)),
        f"all={len(results_all)}",
    )

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 9 — Multiple documents, correct retrieval
# ══════════════════════════════════════════════════


async def test_correct_retrieval() -> None:
    header("Multiple Documents — Correct Retrieval", 9)

    if not MODEL_AVAILABLE:
        skip("Retrieval correctness tests")
        return

    store = QdrantStore(collection_name="test_correct", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Query-specific retrieval")

    test_cases = [
        ("What is RAG?", "RAG", "retrieval"),
        ("How does attention work?", "attention", "transformer"),
        ("What are vector databases?", "vector", "qdrant"),
        ("How does fine-tuning work?", "fine-tun", "lora"),
        ("Tell me about Python", "python", "programming"),
    ]

    for query, keyword1, keyword2 in test_cases:
        results = await store.similarity_search(query, k=1)
        if results:
            content = results[0].page_content.lower()
            matches = keyword1.lower() in content or keyword2.lower() in content
            check(
                f"'{query[:35]}' → relevant result",
                matches,
                f"found={'yes' if matches else 'no'}, content='{content[:50]}...'",
            )
        else:
            check(f"'{query[:35]}' → relevant result", False, "no results")

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 10 — Collection stats
# ══════════════════════════════════════════════════


async def test_collection_stats() -> None:
    header("Collection Stats", 10)

    if not MODEL_AVAILABLE:
        skip("Stats tests")
        return

    store = QdrantStore(collection_name="test_stats", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Stats structure")

    stats = await store.get_collection_stats()
    check("Stats is a dict", isinstance(stats, dict))
    check("Has backend", stats.get("backend") == "qdrant")
    check("Has collection_name", stats.get("collection_name") == "test_stats")
    check("Has document_count", "document_count" in stats)
    check("Document count = 5", stats.get("document_count") == 5, f"count={stats.get('document_count')}")
    check("Has search_mode", "search_mode" in stats)
    check("Has mode", stats.get("mode") == "memory")
    info(f"Full stats: {stats}")

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 11 — Edge cases
# ══════════════════════════════════════════════════


async def test_edge_cases() -> None:
    header("Empty Query and Edge Cases", 11)

    if not MODEL_AVAILABLE:
        skip("Edge case tests")
        return

    store = QdrantStore(collection_name="test_edge", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Empty query")

    results = await store.similarity_search("", k=3)
    check("Empty query → empty results", len(results) == 0)

    results2 = await store.similarity_search("   \t\n  ", k=3)
    check("Whitespace query → empty results", len(results2) == 0)

    sub_header("Empty document list")

    ids = await store.add_documents([])
    check("Empty doc list → empty IDs", len(ids) == 0)

    sub_header("k=0")

    results_k0 = await store.similarity_search("test", k=0)
    check("k=0 → empty results", len(results_k0) == 0)

    sub_header("Very long query")

    long_query = "explain " * 500 + "RAG"
    results_long = await store.similarity_search(long_query, k=1)
    check("Long query doesn't crash", True, f"results={len(results_long)}")

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 12 — Delete collection
# ══════════════════════════════════════════════════


async def test_delete_collection() -> None:
    header("Delete Collection", 12)

    if not MODEL_AVAILABLE:
        skip("Delete tests")
        return

    store = QdrantStore(collection_name="test_delete", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents())

    sub_header("Verify data exists before delete")

    stats = await store.get_collection_stats()
    check("Data exists", stats.get("document_count", 0) == 5)

    sub_header("Delete collection")

    await store.delete_collection()

    # Stats should fail or return empty after deletion
    stats_after = await store.get_collection_stats()
    check(
        "Collection gone after delete",
        stats_after.get("document_count") is None or stats_after == {},
    )

    await store.close()


# ══════════════════════════════════════════════════
# SECTION 13 — Graceful close
# ══════════════════════════════════════════════════


async def test_graceful_close() -> None:
    header("Graceful Close", 13)

    if not MODEL_AVAILABLE:
        skip("Close tests")
        return

    store = QdrantStore(collection_name="test_close", in_memory=True)
    await store.initialize()
    await store.add_documents(make_sample_documents()[:2])

    sub_header("Close releases resources")

    await store.close()
    check("Client set to None", store._client is None)
    check("Store set to None", store._store is None)

    sub_header("Double close is safe")

    await store.close()
    check("Double close — no error", True)


# ══════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════


async def run_all() -> None:
    pipeline_start = time.perf_counter()

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   VECTOR STORE PIPELINE TEST")
    print(f"   QdrantStore — in-memory mode")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}{Style.RESET_ALL}")

    # Sync test
    test_embedding_model()

    # Async tests
    await test_initialization()
    await test_ingestion()
    await test_basic_search()
    await test_score_threshold()
    await test_relevance_scores()
    await test_content_restoration()
    await test_user_filtering()
    await test_correct_retrieval()
    await test_collection_stats()
    await test_edge_cases()
    await test_delete_collection()
    await test_graceful_close()

    elapsed = time.perf_counter() - pipeline_start

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   FINAL RESULTS — Vector Store Pipeline")
    print(f"{'═' * 70}{Style.RESET_ALL}")
    print()
    print(f"    {Fore.GREEN}Passed  : {PASS_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.RED}Failed  : {FAIL_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.YELLOW}Skipped : {SKIP_COUNT}{Style.RESET_ALL}")
    print(f"    Model   : {'LOADED' if MODEL_AVAILABLE else 'NOT FOUND'}")
    print(f"    Time    : {elapsed:.2f}s")
    print()

    if FAIL_COUNT == 0:
        print(f"    {Fore.GREEN}{'═' * 50}")
        if MODEL_AVAILABLE:
            print(f"    ✓ ALL TESTS PASSED — Vector store verified")
        else:
            print(f"    ✓ ALL TESTS PASSED — Skipped (no model)")
            print(f"    ℹ Configure BGE model in .env for full tests")
        print(f"    {'═' * 50}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.RED}{'═' * 50}")
        print(f"    ✗ {FAIL_COUNT} TEST(S) FAILED — Review above")
        print(f"    {'═' * 50}{Style.RESET_ALL}")

    print()
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all())