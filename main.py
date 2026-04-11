"""
Top-level entry point for layer-by-layer validation.

Design:
    Script runner that exercises the foundation (Layer 1) and vector
    store (Layer 2) subsystems in isolation. Not production traffic —
    used to validate that each layer works correctly after a fresh
    install or environment change.

Chain of Responsibility:
    __main__ → main() (Layer 1 sync) or main_layer2() (Layer 2 async).
    Neither function is wired to the FastAPI app — they are standalone
    smoke tests.

Dependencies:
    config.settings, utils.helpers, chunking.chunker, vectorstore.qdrant_store
"""

import asyncio
import os

# Corporate network SSL fix — must be set before any HTTP library is imported.
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import time
from config.settings import settings
from utils.logger import get_logger
from utils.helpers import generate_unique_id, hash_text, truncate_text, chunk_list, retry
from chunking.chunker import Chunker

from langchain_core.documents import Document
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def main():
    try:
        t0 = time.perf_counter()

        # Settings Validation
        logger.info("=" * 60)
        logger.info("LAYER 1 — FOUNDATION VALIDATION")
        logger.info("=" * 60)

        logger.info(f"App               : {settings.app_name} v{settings.app_version}")
        logger.info(f"Provider          : {settings.default_provider}")
        logger.info(f"OpenAI Model      : {settings.openai_model}")
        logger.info(f"Gemini Model      : {settings.gemini_model}")
        logger.info(f"Embedding Model   : {settings.embedding_model}")
        # logger.info(f"Vector Store      : {settings.vector_store_type}")
        logger.info(f"Chunk Size        : {settings.chunk_size}")
        logger.info(f"Chunk Overlap     : {settings.chunk_overlap}")
        logger.info(f"Context Limit     : {settings.effective_context_limit}")
        logger.info(f"Cache Enabled     : {settings.cache_enabled}")
        logger.info(f"Debug             : {settings.debug}")
        logger.info(f"Log Level         : {settings.log_level}")
        logger.info(f"⏱ Settings        : {time.perf_counter() - t0:.3f}s")

        # Helpers Validation
        logger.info("-" * 60)
        logger.info("HELPERS VALIDATION")
        logger.info("-" * 60)

        t1 = time.perf_counter()
        uid = generate_unique_id()
        logger.info(f"UUID v7            : {uid}")

        hashed = hash_text("what is machine learning?")
        logger.info(f"Hash (cache key)   : {hashed}")

        truncated = truncate_text("This is a very long text " * 10, max_length=50)
        logger.info(f"Truncated text     : {truncated}")

        batches = chunk_list(list(range(10)), chunk_size=3)
        logger.info(f"Chunked list       : {batches}")
        logger.info(f"⏱ Helpers          : {time.perf_counter() - t1:.3f}s")

        # Retry Decorator Validation
        t2 = time.perf_counter()
        attempt_count = {"count": 0}

        @retry(max_retries=3, base_delay=0.1)
        def flaky_function():
            attempt_count["count"] += 1
            if attempt_count["count"] < 3:
                raise ValueError("Simulated failure")
            return "success"

        result = flaky_function()
        logger.info(f"Retry decorator    : {result} after {attempt_count['count']} attempts")
        logger.info(f"⏱ Retry            : {time.perf_counter() - t2:.3f}s")

        # Text Splitter Validation
        logger.info("-" * 60)
        logger.info("TEXT SPLITTER VALIDATION")
        logger.info("-" * 60)

        t3 = time.perf_counter()
        splitter = Chunker()
        logger.info(f"⏱ Splitter init    : {time.perf_counter() - t3:.3f}s")

        sample_text = (
            "Artificial intelligence is transforming industries worldwide. "
            "Machine learning models are deployed in healthcare, finance, "
            "and education. Natural language processing enables computers to "
            "understand human language. Large language models like GPT and "
            "Gemini are revolutionizing how we interact with technology. "
        ) * 20

        # Strategy B — character-based splitting
        t4 = time.perf_counter()
        char_chunks = splitter.split_by_character(sample_text)
        stats_b = splitter.chunk_stats(char_chunks)
        logger.info(f"Character Split    : {stats_b}")
        logger.info(f"⏱ Char split       : {time.perf_counter() - t4:.3f}s")

        # Strategy D — RLM-optimised splitting
        t5 = time.perf_counter()
        rlm_chunks = splitter.split_for_rlm(sample_text)
        stats_rlm = splitter.chunk_stats(rlm_chunks)
        logger.info(f"RLM Split          : {stats_rlm}")
        logger.info(f"⏱ RLM split        : {time.perf_counter() - t5:.3f}s")

        # Edge case — empty input should return empty list, not raise.
        empty_result = splitter.split_by_character("")
        logger.info(f"Empty input test   : {empty_result}")

        logger.info("=" * 60)
        logger.info(f"⏱ TOTAL TIME       : {time.perf_counter() - t0:.3f}s")
        logger.info("Layer 1 — Foundation validation complete ✅")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Layer 1 validation failed: {e}")


async def main_layer2():
    # Layer 2 — Vector Store
    logger.info("=" * 55)
    logger.info("LAYER 2 — VECTOR STORE")
    logger.info("=" * 55)

    # Simulate documents coming from Chunker.split_documents().
    # page_content → will be embedded
    # metadata     → stored as payload, never embedded
    sample_docs = [
        Document(
            page_content="RAG combines retrieval with generation for grounded responses.",
            metadata={"source": "rag_paper.pdf", "page": 1, "doc_id": "", "user_id": ""},
        ),
        Document(
            page_content="LLMs hallucinate less when given retrieved context.",
            metadata={"source": "llm_paper.pdf", "page": 3, "doc_id": "", "user_id": ""},
        ),
        Document(
            page_content="Qdrant is a production-ready vector database.",
            metadata={"source": "qdrant_docs.pdf", "page": 1, "doc_id": "", "user_id": "user_A"},
        ),
        Document(
            page_content="BGE embeddings are free, local, top-ranked on MTEB.",
            metadata={"source": "embedding_blog.pdf", "page": 2, "doc_id": "", "user_id": "user_A"},
        ),
        Document(
            page_content="Score thresholds filter irrelevant chunks before LLM sees them.",
            metadata={"source": "best_practices.pdf", "page": 5, "doc_id": "", "user_id": "user_B"},
        ),
    ]

    t = time.perf_counter()
    store = QdrantStore(in_memory=True)
    await store.initialize()

    # Add documents
    ids = await store.add_documents(sample_docs)
    logger.info(f"Stored  | {len(ids)} chunks")

    # Search — no filter (all documents)
    results = await store.similarity_search("What is RAG?", k=2)
    logger.info(f"Search (no filter)    | {len(results)} results")
    for doc in results:
        logger.info(f"  → [{doc.metadata['source']}] {doc.page_content[:60]}")

    # Search — with score threshold
    results_thresh = await store.similarity_search("What is RAG?", k=3, score_threshold=0.5)
    logger.info(f"Search (thresh=0.5)   | {len(results_thresh)} results")

    # Search — filter by user_id (private mode)
    results_user = await store.similarity_search("vector database", k=3, filter_user_id="user_A")
    logger.info(f"Search (user_A only)  | {len(results_user)} results")
    for doc in results_user:
        logger.info(f"  → [{doc.metadata['user_id']}] {doc.page_content[:60]}")

    # Stats
    logger.info(f"Stats   | {await store.get_collection_stats()}")

    await store.delete_collection()
    await store.close()
    logger.info(f"⏱ Layer 2 | {time.perf_counter() - t:.3f}s")
    logger.info("Layer 2 ✅")


if __name__ == "__main__":
    # main()
    asyncio.run(main_layer2())
