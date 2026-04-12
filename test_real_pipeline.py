"""
End-to-end RAG pipeline test using a real PDF with no mocks.

Test scope:
    End-to-end integration test covering the full ingestion and query pipeline:
    DocumentCleaner → StructurePreserver → Chunker → QdrantStore →
    HybridRetriever → SimpleRAG / ChainRAG with CacheManager.
    Each query is run twice: first call is a cache miss, second is a cache hit.

Flow:
    Load PDF → chunk → upsert to Qdrant → initialize cache → pre-warm LLM
    → run queries (miss + hit pairs) → print structured response summaries.

Configuration (change these two lines to switch modes):
    SEARCH_MODE  : "hybrid" | "dense" | "sparse"
    RAG_VARIANT  : "simple" | "chain"

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
from rag.models.rag_request import RAGRequest, RAGConfig
from rag.rag_factory import RAGFactory
from cache.cache_manager import CacheManager
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Configuration — change these two lines to switch modes
SEARCH_MODE = "hybrid"       # "hybrid" | "dense" | "sparse"
RAG_VARIANT = "simple"       # "simple" | "chain"

PDF_PATH = "./data/sample_docs/Attention is all you need.pdf"

# Query sets — each variant gets queries matched to its strengths

# SimpleRAG: direct factual, single-hop questions
_SIMPLE_QUERIES = [
    "explain the transformer model architecture",
    "describe how multi-headed attention works",
    "what BLEU scores did the transformer get on WMT?",
]

# ChainRAG: multi-hop questions that require connecting multiple sections
_CHAIN_QUERIES = [
    (
        "How does positional encoding work in the transformer, and why is it necessary "
        "given that the architecture uses no recurrence or convolution?"
    ),
    (
        "Explain the full training procedure of the transformer: what optimizer was used, "
        "what learning rate schedule was applied, and what regularisation techniques "
        "were used to prevent overfitting?"
    ),
    (
        "How does the transformer's multi-head attention differ from single-head attention, "
        "and what is the computational cost tradeoff between the number of heads "
        "and the dimension per head?"
    ),
]

QUERIES = {
    "simple": _SIMPLE_QUERIES,
    "chain":  _CHAIN_QUERIES,
}[RAG_VARIANT]


def _print_response(response, query_num: int, query_text: str) -> None:
    """Print a structured summary of a RAGResponse for manual inspection."""
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
    print(f"[ QUERY {query_num} | {RAG_VARIANT.upper()} | {SEARCH_MODE} | {execution_path} ]")
    print(f"QUERY      : {query_text}")
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


async def run() -> None:
    """Load the PDF, ingest it into Qdrant, initialize the cache, and run all queries."""
    logger.info("Pipeline config | variant=%s | search_mode=%s", RAG_VARIANT, SEARCH_MODE)

    # Step 1: Load and clean the PDF
    logger.info("Loading PDF: %s", PDF_PATH)
    cleaner   = DocumentCleaner()
    preserver = StructurePreserver()
    chunker   = Chunker()

    raw_docs   = await asyncio.to_thread(cleaner.load_and_clean, PDF_PATH)
    logger.info("Loaded %d pages from PDF", len(raw_docs))

    structured = await asyncio.to_thread(preserver.preserve, raw_docs)
    chunks     = await asyncio.to_thread(chunker.split_documents, structured)
    logger.info("Split into %d chunks", len(chunks))

    # Step 2: Build Qdrant store and ingest chunks
    store = QdrantStore(
        collection_name="attention_paper",
        in_memory=False,
        search_mode=SEARCH_MODE,
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
        llm = LLMFactory.create_groq_pool()
        fallback_llm = LLMFactory.create_rate_limited("gemini")
        logger.info("LLM: %s | fallback: %s", llm.model_name, fallback_llm.model_name)
    except Exception as e:
        logger.warning("Groq unavailable (%s) — using Gemini as primary", e)
        llm = LLMFactory.create_rate_limited("gemini")
        fallback_llm = None

    # Pre-warm Gemini to open TCP+TLS before Q1
    async def _pre_warm_llm(provider, name):
        try:
            await provider.generate("Reply with: OK", max_tokens=2)
            logger.info("LLM pre-warm complete: %s", name)
        except Exception as exc:
            err = str(exc)
            if "<html" in err.lower() or "<!doctype" in err.lower():
                logger.warning(
                    "LLM pre-warm skipped (%s): blocked by corporate proxy/firewall "
                    "(request to provider API was intercepted)", name,
                )
            else:
                logger.warning("LLM pre-warm failed (%s): %s", name, err[:200])

    # Pre-warm whichever LLM is Gemini — Groq is Zscaler-blocked on corp network.
    # Normal path: Gemini is fallback_llm. Groq-unavailable path: Gemini is llm.
    _gemini_llm = fallback_llm if fallback_llm is not None else llm
    await _pre_warm_llm(_gemini_llm, _gemini_llm.model_name)

    # ChainRAG config: tighter top_k per hop — it accumulates across hops
    rag_config = RAGConfig(
        top_k=3 if RAG_VARIANT == "chain" else 5,
        rerank_strategy="cross_encoder",
        max_hops=3 if RAG_VARIANT == "chain" else 1,
    )

    rag = RAGFactory.create(
        RAG_VARIANT,
        retriever=retriever,
        llm=llm,
        cache=cache,
        fallback_llm=fallback_llm,
    )

    logger.info(
        "RAG pipeline ready | variant=%s | search_mode=%s | queries=%d",
        RAG_VARIANT, SEARCH_MODE, len(QUERIES),
    )

    # Step 5: Run each query twice — first call is a miss, second is a cache hit
    for q_idx, query_text in enumerate(QUERIES, start=1):
        request = RAGRequest(
            query=query_text,
            collection_name="attention_paper",
            config=rag_config,
        )

        logger.info("=== QUERY %d/%d (first call): %s ===", q_idx, len(QUERIES), query_text[:80])
        resp_miss = await rag.query(request)
        _print_response(resp_miss, q_idx * 2 - 1, query_text)

        logger.info("=== QUERY %d/%d (second call — cache): %s ===", q_idx, len(QUERIES), query_text[:80])
        resp_hit = await rag.query(request)
        _print_response(resp_hit, q_idx * 2, query_text)


if __name__ == "__main__":
    asyncio.run(run())
