"""
Integration test for the agent layer using real infrastructure — no mocks.

Flow:
    Inject GroqModelPool + Gemini fallback + real Qdrant (hybrid) + cache
    → initialize → health check → ingest DDAI PDF → configure agents
    → routing check → agent queries → cache hit test → teardown.

LLM stack:
    Primary  : GroqModelPool (6 models — llama-3.1-8b, gpt-oss-20b, kimi-k2,
               llama-3.3-70b, qwen3-32b, llama-4-scout; auto FAST/STRONG routing)
    Fallback : Gemini 2.5 Flash (activated when all Groq exhausted or hard-blocked)

Dependencies:
    GROQ_API_KEY + GEMINI_API_KEY in .env; BGE + SPLADE models;
    Qdrant on localhost:6333; Designing Data Intensive Applications.pdf in data/sample_docs/.
"""

import asyncio
import time

from cache.cache_manager import CacheManager
from config.settings import settings
from llm.llm_factory import LLMFactory
from pipeline.models.pipeline_request import PipelineQuery
from pipeline.rag_pipeline import RAGPipeline
from agents.planner.complexity_detector import should_decompose
from vectorstore.qdrant_store import QdrantStore
from utils.logger import get_logger

logger = get_logger(__name__)

PDF_PATH   = "./data/sample_docs/Designing Data Intensive Applications.pdf"
COLLECTION = "DDAI"

COLLECTIONS = {
    "DDAI": (
        "Designing Data-Intensive Applications (DDIA) by Martin Kleppmann. "
        "Covers storage engines (B-Trees, LSM-Trees), replication (leader-based, "
        "leaderless, multi-leader), partitioning, transactions (ACID, isolation "
        "levels, 2PC), distributed consistency (CAP, linearizability, eventual "
        "consistency), batch and stream processing."
    ),
}

# Complex multi-part queries — all should trigger agent decomposition
AGENT_QUERIES = [
    (
        "Compare B-Tree and LSM-Tree storage engines: how does each handle writes "
        "and reads, what is write amplification, and under what workload should "
        "you prefer one over the other?",
        "B-Tree vs LSM-Tree — multi-entity comparison"
    ),
    (
        "How does leader-based replication differ from leaderless replication "
        "in distributed databases? Compare their consistency guarantees, "
        "failover handling, and trade-offs under network partitions.",
        "Leader vs leaderless replication — multi-aspect comparison"
    ),
    (
        "Explain how distributed transactions work: what are the ACID properties, "
        "how does two-phase commit achieve atomicity across nodes, and what "
        "isolation levels exist to control concurrency anomalies?",
        "ACID + 2PC + isolation levels — multi-concept"
    ),
]


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _section(title: str) -> None:
    print(f"\n{'-' * 72}\n  {title}\n{'-' * 72}")


def _print_response(label: str, response, query: str, wall_ms: float) -> None:
    print(f"\n[ {label} ]")
    print(f"QUERY      : {query[:120]}{'...' if len(query) > 120 else ''}")
    print(f"ROUTE      : {'AGENT (decomposed)' if response.rag_variant == 'agent' else 'DIRECT RAG'}")
    print(f"VARIANT    : {response.rag_variant}")
    print(f"MODEL      : {response.model_name}")
    print(f"CACHE HIT  : {response.cache_hit}"
          + (f" (layer={response.cache_layer})" if response.cache_hit else ""))
    print(f"CONFIDENCE : {response.confidence.value:.3f} ({response.confidence.method})")
    print(f"LATENCY    : {wall_ms:.0f} ms  "
          f"[retrieval={response.timings.retrieval_ms:.0f}ms | "
          f"ranking={response.timings.ranking_ms:.0f}ms | "
          f"generation={response.timings.generation_ms:.0f}ms]")
    if response.sources:
        print(f"SOURCES    : {len(response.sources)} chunks")
        for i, chunk in enumerate(response.sources[:3], 1):
            print(f"  [{i}] score={chunk.relevance_score:.3f} | {chunk.content[:100]}...")
    if response.low_confidence:
        print("  [!] LOW CONFIDENCE — retrieval threshold not met")
    print(f"\nANSWER:\n{response.answer}\n")


async def run() -> None:
    _banner("REAL AGENT TEST — Designing Data-Intensive Applications")

    # Step 1: Build LLM providers and inject real infrastructure
    _section("Step 1: Building infrastructure")

    try:
        llm = LLMFactory.create_groq_pool()
        print(f"  Primary LLM  : {llm.model_name} (GroqModelPool — 6 models)")
    except Exception as exc:
        logger.warning("Groq pool unavailable (%s) — using Gemini as primary", exc)
        llm = LLMFactory.create_rate_limited("gemini")
        print(f"  Primary LLM  : {llm.model_name} (Gemini — Groq unavailable)")

    try:
        fallback_llm = LLMFactory.create_rate_limited("gemini")
        print(f"  Fallback LLM : {fallback_llm.model_name}")
    except Exception as exc:
        logger.warning("Gemini fallback unavailable: %s", exc)
        fallback_llm = None
        print("  Fallback LLM : unavailable")

    store = QdrantStore(collection_name=COLLECTION, in_memory=False, search_mode="hybrid")
    print(f"  Vector store : QdrantStore collection='{COLLECTION}' mode=hybrid")

    pipeline = RAGPipeline(
        llm=llm,
        fallback_llm=fallback_llm,
        store=store,
        cache=CacheManager(settings),
    )

    # Step 2: Initialize — boots all subsystems and pre-warms models
    _section("Step 2: Initializing pipeline")
    init_start = time.perf_counter()
    await pipeline.initialize()
    print(f"  Initialized in {(time.perf_counter() - init_start) * 1000:.0f} ms")

    # Step 3: Health check — abort early if LLM or store is down
    _section("Step 3: Health check")
    health = await pipeline.health_check()
    print(f"  {'[OK]' if health.ready else '[WARN]'} ready={health.ready}")
    print(f"  LLM          : {health.llm}")
    print(f"  Vector store : {health.vector_store}")
    print(f"  Cache        : {health.cache}")
    print(f"  Primary LLM  : {health.details.get('primary_llm', '?')}")
    print(f"  Fallback LLM : {health.details.get('fallback_llm', '?')}")

    if not health.ready:
        print("\n  [!] Pipeline not ready — aborting")
        await pipeline.shutdown()
        return

    # Step 4: Ingest DDAI PDF into real Qdrant
    _section("Step 4: Ingesting DDAI PDF")
    ingest_start = time.perf_counter()
    result = await pipeline.ingest(file_path=PDF_PATH, collection=COLLECTION)
    print(f"  Chunks : {result.chunks_stored} stored / {result.total_chunks} total "
          f"({result.duplicates_skipped} duplicates skipped)")
    print(f"  Time   : {(time.perf_counter() - ingest_start) * 1000:.0f} ms")

    # Step 5: Configure agent layer
    _section("Step 5: Configuring agent layer")
    pipeline.configure_agents(
        collections=COLLECTIONS,
        max_concurrent=3,
        use_llm_verification=False,
    )
    print("  Agent layer configured — planner + parallel retriever + synthesizer")

    # Step 6: Routing check — confirm all queries are flagged for decomposition
    _section("Step 6: Routing check (complexity detector)")
    for query, desc in AGENT_QUERIES:
        flag = should_decompose(query)
        print(f"  {'[OK]' if flag else '[WARN] expected True'}  {desc}")

    # Step 7: Agent queries — full decomposition path via AgentOrchestrator
    _section("Step 7: Agent queries")
    for i, (query, desc) in enumerate(AGENT_QUERIES, 1):
        print(f"\n  Query {i}/{len(AGENT_QUERIES)}: {desc}")
        q_start = time.perf_counter()
        response = await pipeline.query(
            PipelineQuery(
                query=query,
                collection=COLLECTION,
                variant="chain",
                top_k=5,
                max_hops=3,
            )
        )
        _print_response(
            label=f"AGENT QUERY {i} — {desc}",
            response=response,
            query=query,
            wall_ms=(time.perf_counter() - q_start) * 1000,
        )

    # Step 8: Cache hit test — re-run first query; expect cache_hit=True
    _section("Step 8: Cache hit test (re-running first agent query)")
    first_query, first_desc = AGENT_QUERIES[0]
    print(f"\n  Re-running: {first_desc}")
    q_start = time.perf_counter()
    response_hit = await pipeline.query(
        PipelineQuery(
            query=first_query,
            collection=COLLECTION,
            variant="chain",
            top_k=5,
            max_hops=3,
        )
    )
    _print_response(
        label=f"CACHE HIT TEST — {first_desc}",
        response=response_hit,
        query=first_query,
        wall_ms=(time.perf_counter() - q_start) * 1000,
    )
    print("  [OK] cache hit confirmed" if response_hit.cache_hit
          else "  [INFO] cache miss (cold cache or cache disabled)")

    await pipeline.shutdown()
    _banner("TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(run())
