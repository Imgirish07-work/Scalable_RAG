"""
Integration test for the ChainRAG (CoRAG) multi-hop retrieval pipeline.

Test scope:
    End-to-end integration comparing SimpleRAG (single-hop baseline) against
    CoRAG (multi-hop chain retrieval) using a real PDF document. No mocks —
    exercises the full ingestion stack and real LLM calls.

Flow:
    Load PDF → chunk → ingest into in-memory QdrantStore → SimpleRAG baseline
    query → CoRAG multi-hop query → side-by-side comparison printout.

Dependencies:
    redis_commands_reference.pdf in data/sample_docs/, GeminiProvider API key,
    BGE embedding model, in-memory QdrantStore, CacheManager.
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

# SimpleRAG baseline — straightforward single-hop query
SIMPLE_QUERY = "What is the redis-cli.exe scan command and when should I use it?"

# CoRAG query — multi-part question spanning multiple document sections:
# Hop 1 retrieves hash creation/set commands
# CoRAG detects missing: how to READ back fields and increment values
# Hop 2 follows up with targeted retrieval for those operations
CORAG_QUERY = (
    "Explain how to work with Redis hashes end-to-end: "
    "how to create them, add fields, increment numeric fields, "
    "and read back specific fields?"
)


def _print_response(response, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"[ {label} ]")
    print(f"CACHE HIT   : {response.cache_hit} | layer={response.cache_layer}")
    print(f"CONFIDENCE  : {response.confidence.value:.2f} "
          f"(method={response.confidence.method})")
    print(f"LATENCY     : {response.timings.total_ms:.1f} ms")
    print(f"MODEL       : {response.model_name}")
    print(f"SOURCES     : {len(response.sources)} chunks")
    print(f"\nANSWER:\n{response.answer}")
    print("\nSOURCES:")
    for chunk in response.sources:
        print(f"  [{chunk.source_file}] score={chunk.relevance_score:.2f}")
        print(f"    {chunk.content[:120]}...")
    print("=" * 60 + "\n")


async def run():
    # 1. Load + chunk PDF
    logger.info("Loading PDF: %s", PDF_PATH)
    cleaner   = DocumentCleaner()
    preserver = StructurePreserver()
    chunker   = Chunker()

    raw_docs   = await asyncio.to_thread(cleaner.load_and_clean, PDF_PATH)
    structured = await asyncio.to_thread(preserver.preserve, raw_docs)
    chunks     = await asyncio.to_thread(chunker.split_documents, structured)
    logger.info("Loaded %d pages -> %d chunks", len(raw_docs), len(chunks))

    # 2. Vector store
    store = QdrantStore(
        collection_name="redis_corag_test",
        in_memory=True,
        search_mode="dense",
    )
    await store.initialize()
    await store.add_documents(chunks)
    logger.info("Ingested %d chunks into Qdrant", len(chunks))

    # 3. Shared dependencies
    llm   = GeminiProvider()
    cache = CacheManager(settings)
    await cache.initialize()

    # 4. SimpleRAG — baseline
    logger.info("=" * 50)
    logger.info("SIMPLE RAG — baseline (1 retrieval, 1 LLM call)")
    logger.info("=" * 50)

    simple_rag = RAGFactory.create(
        "simple",
        retriever=DenseRetriever(store),
        llm=llm,
        cache=cache,
    )

    simple_request = RAGRequest(
        query=SIMPLE_QUERY,
        collection_name="redis_corag_test",
        config=RAGConfig(top_k=5, rerank_strategy="mmr"),
    )

    simple_response = await simple_rag.query(simple_request)
    _print_response(simple_response, "SIMPLE RAG — single retrieval pass")

    # 5. CoRAG — multi-hop retrieval
    logger.info("=" * 50)
    logger.info("CoRAG — multi-hop retrieval (up to 3 hops)")
    logger.info("LLM calls: 2 per hop (draft + completeness) + 1 final generation")
    logger.info("=" * 50)

    corag = RAGFactory.create(
        "chain",
        retriever=DenseRetriever(store),
        llm=llm,
        cache=cache,
    )

    corag_request = RAGRequest(
        query=CORAG_QUERY,
        collection_name="redis_corag_test",
        config=RAGConfig(
            top_k=3,            # fewer chunks per hop — CoRAG accumulates across hops
            rerank_strategy="mmr",
            max_hops=3,         # max retrieval iterations
        ),
    )

    corag_response = await corag.query(corag_request)
    _print_response(corag_response, "CoRAG — multi-hop chain retrieval")

    # 6. Side-by-side comparison
    print("\n" + "=" * 60)
    print("COMPARISON: SimpleRAG vs CoRAG")
    print("=" * 60)
    print(f"{'Metric':<25} {'SimpleRAG':>15} {'CoRAG':>15}")
    print("-" * 55)
    print(f"{'Latency (ms)':<25} {simple_response.timings.total_ms:>15.1f} "
          f"{corag_response.timings.total_ms:>15.1f}")
    print(f"{'Confidence':<25} {simple_response.confidence.value:>15.2f} "
          f"{corag_response.confidence.value:>15.2f}")
    print(f"{'Confidence method':<25} {simple_response.confidence.method:>15} "
          f"{corag_response.confidence.method:>15}")
    print(f"{'Sources retrieved':<25} {len(simple_response.sources):>15} "
          f"{len(corag_response.sources):>15}")
    print("=" * 60)
    print()
    print("NOTE: CoRAG uses MORE LLM calls (2 per hop + 1 final).")
    print("      Use it when answer quality matters more than latency/cost.")
    print("      SimpleRAG is faster and cheaper for straightforward queries.")


if __name__ == "__main__":
    asyncio.run(run())
