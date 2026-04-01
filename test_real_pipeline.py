"""
Real end-to-end RAG pipeline test with a real PDF.

Uses:
    - redis_commands_reference.pdf  (real document, no mocks)
    - DocumentCleaner + StructurePreserver + Chunker (full ingestion pipeline)
    - QdrantStore      (in-memory, no server needed)
    - DenseRetriever   (BGE embeddings)
    - GeminiProvider   (real LLM)
    - CacheManager     (L1 memory + L2 Redis)

Run:
    python test_real_pipeline.py
"""

import asyncio

from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from chunking.chunker import Chunker
from vectorstore.qdrant_store import QdrantStore
from rag.retrieval.dense_retriever import DenseRetriever
from llm.providers.gemini_provider import GeminiProvider
from rag.models.rag_request import RAGRequest, RAGConfig
from rag.rag_factory import RAGFactory
from cache.cache_manager import CacheManager
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

PDF_PATH = "./data/sample_docs/redis_commands_reference.pdf"
QUERY    = "what does redis-server.exe command do?"


def _print_response(response, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"[ {label} ]")
    print(f"CACHE HIT  : {response.cache_hit} | layer={response.cache_layer}")
    print(f"ANSWER:\n{response.answer}")
    print("\nSOURCES:")
    for chunk in response.sources:
        print(f"  [{chunk.source_file}] score={chunk.relevance_score:.2f}")
        print(f"    {chunk.content[:150]}...")
    print(f"\nCONFIDENCE : {response.confidence.value:.2f}")
    print(f"LATENCY    : {response.timings.total_ms:.1f} ms")
    print(f"MODEL      : {response.model_name}")
    print("=" * 60 + "\n")


async def run():
    # ── 1. Load + clean PDF ──────────────────────────────────────────────────
    logger.info("Loading PDF: %s", PDF_PATH)
    cleaner   = DocumentCleaner()
    preserver = StructurePreserver()
    chunker   = Chunker()

    raw_docs    = await asyncio.to_thread(cleaner.load_and_clean, PDF_PATH)
    logger.info("Loaded %d pages from PDF", len(raw_docs))

    structured  = await asyncio.to_thread(preserver.preserve, raw_docs)
    chunks      = await asyncio.to_thread(chunker.split_documents, structured)
    logger.info("Split into %d chunks", len(chunks))

    # ── 2. Vector store ───────────────────────────────────────────────────────
    store = QdrantStore(
        collection_name="redis_docs",
        in_memory=False,
        search_mode="dense",
    )
    await store.initialize()

    ids = await store.add_documents(chunks)
    logger.info("Ingested %d chunks into Qdrant", len(ids))

    # ── 3. Cache ──────────────────────────────────────────────────────────────
    cache = CacheManager(settings)
    await cache.initialize()

    # ── 4. RAG pipeline ───────────────────────────────────────────────────────
    retriever = DenseRetriever(store)
    llm       = GeminiProvider()

    rag = RAGFactory.create(
        "simple",
        retriever=retriever,
        llm=llm,
        cache=cache,
    )

    request = RAGRequest(
        query=QUERY,
        collection_name="redis_docs",
        config=RAGConfig(top_k=5, rerank_strategy="mmr"),
    )

    # ── 5. Query 1 — cache MISS ───────────────────────────────────────────────
    logger.info("=== QUERY 1 (expect cache MISS) ===")
    response1 = await rag.query(request)
    _print_response(response1, "QUERY 1 — cache miss, calls Gemini")

    # ── 6. Query 2 — cache HIT ────────────────────────────────────────────────
    logger.info("=== QUERY 2 (expect cache HIT) ===")
    response2 = await rag.query(request)
    _print_response(response2, "QUERY 2 — should be cache HIT")


if __name__ == "__main__":
    asyncio.run(run())
