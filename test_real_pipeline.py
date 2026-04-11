"""
End-to-end RAG pipeline test using a real PDF with no mocks.

Test scope:
    End-to-end integration test covering the full ingestion and query pipeline:
    DocumentCleaner → StructurePreserver → Chunker → QdrantStore (in-memory,
    hybrid mode) → HybridRetriever → SimpleRAG with CacheManager (L1+L2).
    Each query is run twice: first call is a cache miss, second is a cache hit.

Flow:
    Load PDF → chunk → upsert to Qdrant → initialize cache → pre-warm LLM
    → run queries (miss + hit pairs) → print structured response summaries.

Dependencies:
    GEMINI_API_KEY (and optionally GROQ_API_KEY) in .env; BGE embedding model;
    sample PDF at data/sample_docs/Attention is all you need.pdf.
"""

import asyncio

from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from chunking.chunker import Chunker
from vectorstore.qdrant_store import QdrantStore
from rag.retrieval.hybrid_retriever import HybridRetriever
from llm.llm_factory import LLMFactory
from llm.exceptions.llm_exceptions import LLMProviderError
from rag.models.rag_request import RAGRequest, RAGConfig
from rag.rag_factory import RAGFactory
from cache.cache_manager import CacheManager
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

PDF_PATH = "./data/sample_docs/Attention is all you need.pdf"
# Original queries (Run 1): populate the semantic cache with canonical forms.
# Paraphrases (Run 2): verify cross-restart semantic cache hits on startup.
# QUERIES = [
#     "what is the transformer architecture?",
#     "how does multi-head attention work?",
#     "what are the results on WMT translation tasks?",
# ]

# Paraphrases for Run 2 semantic cache validation
QUERIES = [
    "explain the transformer model architecture",
    "describe how multi-headed attention works",
    "what BLEU scores did the transformer get on WMT?",
]

def _print_response(response, query_num: int) -> None:
    """Print a structured summary of a RAGResponse for manual inspection."""
    # Derive the execution path label from actual response fields
    if response.cache_hit:
        execution_path = f"cache HIT ({response.cache_layer})"
        model_label = f"[cached] {response.model_name}"
    elif response.timings.generation_ms == 0.0 and response.low_confidence:
        execution_path = "low-confidence guard (no LLM call)"
        model_label = response.model_name
    else:
        execution_path = f"cache MISS — called {response.model_name}"
        model_label = response.model_name

    print(f"\n{'=' * 60}")
    print(f"[ QUERY {query_num} — {execution_path} ]")
    print(f"CACHE HIT  : {response.cache_hit} | layer={response.cache_layer}")
    if response.low_confidence:
        print("LOW CONFIDENCE: retrieval threshold not met — answer may be empty")
    print(f"ANSWER:\n{response.answer}")
    print("\nSOURCES:")
    for chunk in response.sources:
        print(f"  [{chunk.source_file}] score={chunk.relevance_score:.2f}")
        print(f"    {chunk.content[:150]}...")
    print(f"\nCONFIDENCE : {response.confidence.value:.4f} (method={response.confidence.method})")
    print(f"LATENCY    : {response.timings.total_ms:.1f} ms  "
          f"[retrieval={response.timings.retrieval_ms:.0f}ms | "
          f"ranking={response.timings.ranking_ms:.0f}ms | "
          f"generation={response.timings.generation_ms:.0f}ms]")
    print(f"MODEL      : {model_label}")
    print("=" * 60 + "\n")


async def run():
    """Load the PDF, ingest it into Qdrant, initialize the cache, and run all queries."""
    # Step 1: Load and clean the PDF
    logger.info("Loading PDF: %s", PDF_PATH)
    cleaner   = DocumentCleaner()
    preserver = StructurePreserver()
    chunker   = Chunker()

    raw_docs    = await asyncio.to_thread(cleaner.load_and_clean, PDF_PATH)
    logger.info("Loaded %d pages from PDF", len(raw_docs))

    structured  = await asyncio.to_thread(preserver.preserve, raw_docs)
    chunks      = await asyncio.to_thread(chunker.split_documents, structured)
    logger.info("Split into %d chunks", len(chunks))

    # Step 2: Build in-memory Qdrant store and ingest chunks
    store = QdrantStore(
        collection_name="attention_paper",
        in_memory=False,
        search_mode="hybrid",
    )
    await store.initialize()

    ids = await store.add_documents(chunks)
    logger.info("Ingested %d chunks into Qdrant", len(ids))

    # Warm up HNSW index in RAM; skip if all chunks were already present (duplicates)
    if ids:
        try:
            await store.similarity_search_with_vectors("warmup", k=1)
            logger.info("Post-ingest HNSW warmup complete")
        except Exception:
            pass

    # Step 3: Initialize the cache
    cache = CacheManager(settings)
    await cache.initialize()

    # Step 4: Build the RAG pipeline
    retriever = HybridRetriever(store)
    try:
        llm = LLMFactory.create_rate_limited("groq")
        fallback_llm = LLMFactory.create_rate_limited("gemini")
        logger.info("LLM: %s | fallback: %s", llm.model_name, fallback_llm.model_name)
    except Exception as e:
        logger.warning("Groq unavailable (%s) — using Gemini as primary", e)
        llm = LLMFactory.create_rate_limited("gemini")
        fallback_llm = None

    # Pre-warm the LLM to open TCP+TLS before Q1 (no RAGPipeline.initialize() here)
    async def _pre_warm_llm(provider, name):
        try:
            await provider.generate("Reply with: OK", max_tokens=2)
            logger.info("LLM pre-warm complete: %s", name)
        except Exception as exc:
            err = str(exc)
            if "<html" in err.lower() or "<!doctype" in err.lower():
                # Zscaler/proxy returned an HTML block page — log cleanly
                logger.warning(
                    "LLM pre-warm skipped (%s): blocked by corporate proxy/firewall "
                    "(request to provider API was intercepted)", name,
                )
            else:
                logger.warning("LLM pre-warm failed (%s): %s", name, err[:200])

    # Only warm Gemini; Groq is Zscaler-blocked on corp network and will
    # self-attempt on first query then fall back to the warm Gemini provider.
    if fallback_llm:
        await _pre_warm_llm(fallback_llm, fallback_llm.model_name)

    rag = RAGFactory.create(
        "simple",
        retriever=retriever,
        llm=llm,
        cache=cache,
        fallback_llm=fallback_llm,
    )

    # Step 5: Run each query twice — first call is a miss, second is a cache hit
    for q_idx, query_text in enumerate(QUERIES, start=1):
        request = RAGRequest(
            query=query_text,
            collection_name="attention_paper",
            config=RAGConfig(top_k=5, rerank_strategy="cross_encoder"),
        )

        logger.info("=== QUERY %d/%d (first call): %s ===", q_idx, len(QUERIES), query_text)
        resp_miss = await rag.query(request)
        _print_response(resp_miss, q_idx * 2 - 1)

        logger.info("=== QUERY %d/%d (second call): %s ===", q_idx, len(QUERIES), query_text)
        resp_hit = await rag.query(request)
        _print_response(resp_hit, q_idx * 2)


if __name__ == "__main__":
    asyncio.run(run())
