"""
Real end-to-end agent test — The Constitution of India.

Uses:
    - THE_CONSTITUTION_OF_INDIA.pdf  (real document, no mocks)
    - Full ingestion pipeline (DocumentCleaner → Chunker → QdrantStore in-memory)
    - GeminiProvider (real LLM)
    - RAGPipeline with configure_agents()
    - 3 agent queries that require decomposition
    - 2 simple queries that bypass the agent (direct RAG)

Run:
    python test_agents_real.py
"""

import asyncio
import time

from chunking.chunker import Chunker
from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from pipeline.rag_pipeline import RAGPipeline
from pipeline.models.pipeline_request import PipelineQuery
from agents.planner.complexity_detector import should_decompose
from utils.logger import get_logger

logger = get_logger(__name__)

PDF_PATH   = "./data/sample_docs/THE_CONSTITUTION_OF_INDIA.pdf"
COLLECTION = "constitution_india"

# ── Collections metadata for the planner ──────────────────────────────────────
COLLECTIONS = {
    "constitution_india": (
        "The Constitution of India — Preamble, Parts, Articles, Schedules. "
        "Covers fundamental rights, directive principles, executive, legislative, "
        "judicial powers, amendment procedures, and emergency provisions."
    ),
}

# ── Queries ───────────────────────────────────────────────────────────────────

# These should trigger agent decomposition (complex, multi-part)
AGENT_QUERIES = [
    (
        "Compare the powers and composition of the Lok Sabha versus the Rajya Sabha, "
        "and explain how both houses differ in terms of election, tenure, and legislative role.",
        "Comparing two houses of Parliament — multi-entity comparison query"
    ),
    (
        "What are the Fundamental Rights guaranteed under Part III, and how do the "
        "Directive Principles of State Policy in Part IV differ from them in terms of "
        "enforceability and purpose?",
        "Comparing Fundamental Rights vs Directive Principles — multi-part query"
    ),
    (
        "Explain the emergency provisions under Articles 352, 356, and 360 — "
        "what triggers each type of emergency, who proclaims it, and what are the "
        "constitutional safeguards against misuse?",
        "Three emergency types with multiple sub-questions"
    ),
]

# These should go directly to RAG (simple, single-focus)
SIMPLE_QUERIES = [
    (
        "What does the Preamble of the Indian Constitution say?",
        "Simple factual — single focus"
    ),
    (
        "How many schedules are there in the Constitution of India?",
        "Simple factual — single focus"
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _section(title: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def _print_response(label: str, response, query: str, routed_to_agent: bool) -> None:
    print(f"\n[ {label} ]")
    print(f"QUERY      : {query[:120]}{'...' if len(query) > 120 else ''}")
    print(f"ROUTE      : {'AGENT (decomposed)' if routed_to_agent else 'DIRECT RAG'}")
    print(f"VARIANT    : {response.rag_variant}")
    print(f"CONFIDENCE : {response.confidence.value:.3f} ({response.confidence.method})")
    print(f"LATENCY    : {response.timings.total_ms:.0f} ms")
    print(f"CACHE HIT  : {response.cache_hit}")
    if response.sources:
        print(f"SOURCES    : {len(response.sources)} chunks retrieved")
        for i, chunk in enumerate(response.sources[:2], 1):
            print(f"  [{i}] score={chunk.relevance_score:.3f} | {chunk.content[:100]}...")
    print(f"\nANSWER:\n{response.answer}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def run() -> None:
    _banner("REAL AGENT TEST — The Constitution of India")

    # ── Step 1: Ingest document ───────────────────────────────────────────────
    _section("Step 1: Ingesting document")

    pipeline = RAGPipeline()
    await pipeline.initialize()

    ingest_start = time.perf_counter()
    result = await pipeline.ingest(
        file_path=PDF_PATH,
        collection=COLLECTION,
    )
    ingest_ms = (time.perf_counter() - ingest_start) * 1000

    print(f"  File       : {result.file_path}")
    print(f"  Collection : {result.collection}")
    print(f"  Chunks     : {result.chunks_stored} stored / {result.total_chunks} total")
    print(f"  Duplicates : {result.duplicates_skipped} skipped")
    print(f"  Time       : {ingest_ms:.0f} ms")

    # ── Step 2: Configure agents ──────────────────────────────────────────────
    _section("Step 2: Configuring agent layer")

    pipeline.configure_agents(
        collections=COLLECTIONS,
        max_concurrent=3,
        use_llm_verification=False,
    )
    print("  Agent layer configured — planner + parallel retriever + verifier + synthesizer")

    # ── Step 3: Routing check — confirm queries route correctly ───────────────
    _section("Step 3: Routing check (complexity detector)")

    print("  AGENT queries (should decompose = True):")
    for query, desc in AGENT_QUERIES:
        result_flag = should_decompose(query)
        status = "[OK]" if result_flag else "[WARN] expected True"
        print(f"    {status} {desc[:60]}")

    print("\n  SIMPLE queries (should decompose = False):")
    for query, desc in SIMPLE_QUERIES:
        result_flag = should_decompose(query)
        status = "[OK]" if not result_flag else "[WARN] expected False"
        print(f"    {status} {desc[:60]}")

    # ── Step 4: Run agent queries ─────────────────────────────────────────────
    _section("Step 4: Agent queries (decomposed)")

    for i, (query, desc) in enumerate(AGENT_QUERIES, 1):
        print(f"\n  Running agent query {i}/3: {desc}")
        print(f"  {'.' * 60}")

        q_start = time.perf_counter()
        response = await pipeline.query(
            PipelineQuery(query=query, collection=COLLECTION)
        )
        q_ms = (time.perf_counter() - q_start) * 1000

        routed_to_agent = (response.rag_variant == "agent")
        _print_response(
            label=f"AGENT QUERY {i} — {desc}",
            response=response,
            query=query,
            routed_to_agent=routed_to_agent,
        )
        print(f"  Wall-clock latency: {q_ms:.0f} ms")

    # ── Step 5: Run simple queries (direct RAG, no decomposition) ─────────────
    _section("Step 5: Simple queries (direct RAG)")

    for i, (query, desc) in enumerate(SIMPLE_QUERIES, 1):
        print(f"\n  Running simple query {i}/2: {desc}")

        q_start = time.perf_counter()
        response = await pipeline.query(
            PipelineQuery(query=query, collection=COLLECTION)
        )
        q_ms = (time.perf_counter() - q_start) * 1000

        routed_to_agent = (response.rag_variant == "agent")
        _print_response(
            label=f"SIMPLE QUERY {i} — {desc}",
            response=response,
            query=query,
            routed_to_agent=routed_to_agent,
        )
        print(f"  Wall-clock latency: {q_ms:.0f} ms")

    # ── Step 6: Teardown ──────────────────────────────────────────────────────
    await pipeline.shutdown()
    _banner("TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(run())
