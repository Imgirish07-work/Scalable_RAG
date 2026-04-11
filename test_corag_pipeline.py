"""
Integration test for ChainRAG and CorrectiveRAG with real LLM calls.

Test scope:
    End-to-end comparison of SimpleRAG (baseline), ChainRAG (multi-hop), and
    CorrectiveRAG (self-correcting) on the same document and queries. No mocks —
    exercises the full ingestion stack and real LLM calls.

    ChainRAG is designed for queries that require connecting information across
    multiple document sections. It performs up to max_hops retrieval iterations,
    each time checking whether the accumulated context is sufficient to answer.

    CorrectiveRAG is designed for borderline or ambiguous queries. It evaluates
    retrieved context and re-retrieves with a rewritten query when confidence is low.

Flow:
    Load PDF → chunk → ingest into QdrantStore → run 3 variants side-by-side
    → print per-variant responses and a comparison table.

Configuration:
    PDF_PATH   : document to test against (default: Attention is all you need)
    SEARCH_MODE: "hybrid" | "dense" (applied to all variants equally)

Dependencies:
    GEMINI_API_KEY (and optionally GROQ_API_KEY) in .env; BGE embedding model.
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

# Configuration
SEARCH_MODE = "hybrid"   # "hybrid" | "dense"
PDF_PATH    = "./data/sample_docs/Attention is all you need.pdf"
COLLECTION  = "attention_corag_test"

# Query sets

# Multi-hop queries: answer requires connecting ≥2 separate document sections
CHAIN_QUERIES = [
    (
        "How does positional encoding work in the transformer, and why is it necessary "
        "given that the architecture uses no recurrence or convolution?",
        "multi-hop: positional encoding + no-recurrence design decision"
    ),
    (
        "Explain the full training procedure of the transformer: what optimizer was used, "
        "what learning rate schedule was applied, what regularisation techniques "
        "were used, and how long was training on WMT English-to-German?",
        "multi-hop: spans optimizer, lr schedule, dropout, training time sections"
    ),
    (
        "How does the transformer's multi-head attention mechanism reduce the path length "
        "between long-range dependencies compared to RNNs and CNNs, and what is the "
        "computational complexity tradeoff per layer?",
        "multi-hop: attention → path length table → complexity comparison table"
    ),
]

# Corrective queries: borderline or partially answerable — should trigger re-retrieval
CORRECTIVE_QUERIES = [
    (
        "Does the transformer completely outperform all previous architectures on every "
        "benchmark, or are there tasks or settings where RNNs and CNNs remain competitive?",
        "corrective: ambiguous — paper makes specific claims, not blanket superiority"
    ),
    (
        "What are the limitations, failure modes, or open questions acknowledged by the "
        "authors in the attention is all you need paper?",
        "corrective: borderline — limitations section may be sparse"
    ),
    (
        "Is scaled dot-product attention always more efficient than additive attention, "
        "or does the relationship depend on the dimension of the queries and keys?",
        "corrective: nuanced — answer depends on specific dimension threshold"
    ),
]


def _print_response(response, label: str, query_text: str, note: str) -> None:
    """Print a structured summary of a RAGResponse."""
    if response.cache_hit:
        path = f"cache HIT ({response.cache_layer})"
    elif response.timings.generation_ms == 0.0 and response.low_confidence:
        path = "low-confidence guard (no LLM call)"
    else:
        path = f"LLM call → {response.model_name}"

    print(f"\n{'=' * 65}")
    print(f"[ {label} | {path} ]")
    print(f"QUERY : {query_text}")
    print(f"NOTE  : {note}")
    print(f"CACHE : {response.cache_hit} | layer={response.cache_layer}")
    if response.low_confidence:
        print(">>> LOW CONFIDENCE: re-retrieval triggered or answer withheld")
    print(f"ANSWER:\n{response.answer}")
    print(f"\nSOURCES ({len(response.sources)} chunks):")
    for chunk in response.sources:
        print(f"  [{chunk.source_file}] score={chunk.relevance_score:.2f}")
        print(f"    {chunk.content[:120]}...")
    print(f"\nCONFIDENCE : {response.confidence.value:.4f} (method={response.confidence.method})")
    print(f"LATENCY    : {response.timings.total_ms:.1f} ms  "
          f"[retrieval={response.timings.retrieval_ms:.0f}ms | "
          f"ranking={response.timings.ranking_ms:.0f}ms | "
          f"generation={response.timings.generation_ms:.0f}ms]")
    print("=" * 65 + "\n")


def _print_comparison(results: list[tuple]) -> None:
    """Print a side-by-side comparison table for all variants on one query."""
    print(f"\n{'─' * 75}")
    print(f"{'Metric':<22} {'SimpleRAG':>16} {'ChainRAG':>16} {'CorrectiveRAG':>16}")
    print(f"{'─' * 75}")
    for label, simple, chain, corrective in results:
        print(f"{label:<22} {simple:>16} {chain:>16} {corrective:>16}")
    print(f"{'─' * 75}\n")


async def run() -> None:
    """Ingest document, build 3 RAG variants, run all queries, print comparison."""
    logger.info("Test config | search_mode=%s | pdf=%s", SEARCH_MODE, PDF_PATH)

    # Step 1: Ingest
    logger.info("Loading and chunking PDF...")
    cleaner   = DocumentCleaner()
    preserver = StructurePreserver()
    chunker   = Chunker()

    raw_docs   = await asyncio.to_thread(cleaner.load_and_clean, PDF_PATH)
    structured = await asyncio.to_thread(preserver.preserve, raw_docs)
    chunks     = await asyncio.to_thread(chunker.split_documents, structured)
    logger.info("Loaded %d pages → %d chunks", len(raw_docs), len(chunks))

    # Step 2: Vector store
    store = QdrantStore(
        collection_name=COLLECTION,
        in_memory=True,
        search_mode=SEARCH_MODE,
    )
    await store.initialize()
    await store.add_documents(chunks)
    logger.info("Ingested %d chunks | search_mode=%s", len(chunks), SEARCH_MODE)

    # Step 3: Shared dependencies
    cache = CacheManager(settings)
    await cache.initialize()

    retriever = HybridRetriever(store)

    try:
        llm = LLMFactory.create_groq_pool()
        fallback_llm = LLMFactory.create_rate_limited("gemini")
        logger.info("LLM: %s | fallback: %s", llm.model_name, fallback_llm.model_name)
    except Exception as e:
        logger.warning("Groq unavailable (%s) — using Gemini as primary", e)
        llm = LLMFactory.create_rate_limited("gemini")
        fallback_llm = None

    # Step 4: Build all three variants sharing the same retriever + cache
    simple_rag = RAGFactory.create(
        "simple",
        retriever=retriever,
        llm=llm,
        cache=cache,
        fallback_llm=fallback_llm,
    )
    chain_rag = RAGFactory.create(
        "chain",
        retriever=retriever,
        llm=llm,
        cache=cache,
        fallback_llm=fallback_llm,
    )
    corrective_rag = RAGFactory.create(
        "corrective",
        retriever=retriever,
        llm=llm,
        cache=cache,
        fallback_llm=fallback_llm,
    )

    # Step 5: Run ChainRAG queries
    print("\n" + "=" * 65)
    print("SECTION 1 — ChainRAG: multi-hop queries")
    print("=" * 65)

    chain_config = RAGConfig(
        top_k=3,             # fewer per hop — accumulates across hops
        rerank_strategy="cross_encoder",
        max_hops=3,
    )
    simple_config = RAGConfig(top_k=5, rerank_strategy="cross_encoder")

    for q_idx, (query_text, note) in enumerate(CHAIN_QUERIES, start=1):
        logger.info("CHAIN Q%d: %s", q_idx, query_text[:80])

        simple_resp = await simple_rag.query(RAGRequest(
            query=query_text, collection_name=COLLECTION, config=simple_config,
        ))
        chain_resp = await chain_rag.query(RAGRequest(
            query=query_text, collection_name=COLLECTION, config=chain_config,
        ))

        _print_response(simple_resp, f"CHAIN Q{q_idx} — SimpleRAG baseline", query_text, note)
        _print_response(chain_resp,  f"CHAIN Q{q_idx} — ChainRAG multi-hop", query_text, note)

        _print_comparison([
            ("Latency (ms)",
             f"{simple_resp.timings.total_ms:.0f}",
             f"{chain_resp.timings.total_ms:.0f}",
             "—"),
            ("Confidence",
             f"{simple_resp.confidence.value:.3f}",
             f"{chain_resp.confidence.value:.3f}",
             "—"),
            ("Sources retrieved",
             str(len(simple_resp.sources)),
             str(len(chain_resp.sources)),
             "—"),
            ("Low confidence",
             str(simple_resp.low_confidence),
             str(chain_resp.low_confidence),
             "—"),
        ])

    # Step 6: Run CorrectiveRAG queries
    print("\n" + "=" * 65)
    print("SECTION 2 — CorrectiveRAG: borderline / ambiguous queries")
    print("=" * 65)

    for q_idx, (query_text, note) in enumerate(CORRECTIVE_QUERIES, start=1):
        logger.info("CORRECTIVE Q%d: %s", q_idx, query_text[:80])

        simple_resp     = await simple_rag.query(RAGRequest(
            query=query_text, collection_name=COLLECTION, config=simple_config,
        ))
        corrective_resp = await corrective_rag.query(RAGRequest(
            query=query_text, collection_name=COLLECTION, config=simple_config,
        ))

        _print_response(simple_resp,     f"CORRECTIVE Q{q_idx} — SimpleRAG baseline",   query_text, note)
        _print_response(corrective_resp, f"CORRECTIVE Q{q_idx} — CorrectiveRAG",         query_text, note)

        _print_comparison([
            ("Latency (ms)",
             f"{simple_resp.timings.total_ms:.0f}",
             "—",
             f"{corrective_resp.timings.total_ms:.0f}"),
            ("Confidence",
             f"{simple_resp.confidence.value:.3f}",
             "—",
             f"{corrective_resp.confidence.value:.3f}"),
            ("Sources retrieved",
             str(len(simple_resp.sources)),
             "—",
             str(len(corrective_resp.sources))),
            ("Low confidence",
             str(simple_resp.low_confidence),
             "—",
             str(corrective_resp.low_confidence)),
        ])

    print("\nTest complete.")
    print(f"  Chain queries    : {len(CHAIN_QUERIES)}")
    print(f"  Corrective queries: {len(CORRECTIVE_QUERIES)}")
    print(f"  Search mode      : {SEARCH_MODE}")
    print(f"  Document         : {PDF_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
