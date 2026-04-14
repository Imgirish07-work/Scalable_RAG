"""
End-to-end RAG pipeline test using a real PDF with no mocks.

Test scope:
    Full pipeline test covering ingestion and query routing:
    DocumentCleaner → StructurePreserver → Chunker → QdrantStore →
    RAGPipeline → auto-routes: simple queries → SimpleRAG,
    complex queries → AgentOrchestrator (decompose + fuse + synthesize).

Flow:
    Load PDF → chunk → upsert to Qdrant → initialize cache → configure agents
    → run queries (auto-routed by complexity detector).

Configuration:
    SEARCH_MODE : "hybrid" | "dense"
    COLLECTION  : Qdrant collection name
    PDF_PATH    : Path to the source PDF

Dependencies:
    GEMINI_API_KEY (and optionally GROQ_API_KEY) in .env; BGE embedding model.
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

# Configuration

SEARCH_MODE = "hybrid"   # "hybrid" | "dense"

COLLECTION = "A Dance with Dragons"
PDF_PATH   = "./data/sample_docs/A Dance with Dragons.pdf"

# Description is read by the agent planner to route sub-queries — be specific.
COLLECTIONS = {
    COLLECTION: (
        "A Dance with Dragons by George R.R. Martin. Covers Jon Snow commanding "
        "the Night's Watch at Castle Black, Daenerys ruling Meereen and managing "
        "dragons, Tyrion's journey through Essos, Bran's training beyond the Wall, "
        "and political conflicts across Westeros including the Iron Throne."
    )
}

# Mix of simple and complex queries — pipeline auto-routes each one.
QUERIES = [
    # Simple — single-hop, direct factual → SimpleRAG
    "Who is Jon Snow?",
    "Where is Daenerys at the start of the book?",
    "What is the Night's Watch?",

    # Complex — multi-aspect, multi-character → AgentOrchestrator
    (
        "Analyze Jon Snow's leadership at the Wall: what major decisions does he make, "
        "how do they affect the Night's Watch and the wildlings, and what conflicts "
        "arise that ultimately lead to his downfall?"
    ),
    (
        "Examine Daenerys Targaryen's rule in Meereen: what challenges does she face in "
        "governing, how do her decisions impact stability and slavery, and how do these "
        "events influence her control over dragons and her future direction?"
    ),
    (
        "Compare the storylines of Jon Snow, Daenerys, and Tyrion: how do their leadership "
        "styles differ, what challenges do they face, and how do their decisions shape "
        "their respective arcs?"
    ),
    (
        "Analyze the political instability across Westeros: what factions are involved, "
        "how do their conflicts evolve during the story, and how do these power struggles "
        "impact the overall state of the realm?"
    ),
]

# Print helper

def _print_response(response, query_num: int, query_text: str, wall_ms: float) -> None:
    """Print a structured summary of a RAGResponse."""
    if response.cache_hit:
        execution_path = f"CACHE HIT ({response.cache_layer})"
    else:
        execution_path = "CACHE MISS"

    route = "AGENT" if response.rag_variant == "agent" else "SIMPLE RAG"

    print(f"\n{'=' * 70}")
    print(f"[ Q{query_num} | {route} | {SEARCH_MODE.upper()} | {execution_path} ]")
    print(f"QUERY      : {query_text[:120]}{'...' if len(query_text) > 120 else ''}")
    print(f"VARIANT    : {response.rag_variant}")
    print(f"MODEL      : {response.model_name}")
    print(f"CONFIDENCE : {response.confidence.value:.4f} ({response.confidence.method})")
    print(f"LATENCY    : {wall_ms:.0f} ms total  "
          f"[retrieval={response.timings.retrieval_ms:.0f}ms | "
          f"ranking={response.timings.ranking_ms:.0f}ms | "
          f"generation={response.timings.generation_ms:.0f}ms]")
    if response.low_confidence:
        print("  [!] LOW CONFIDENCE — retrieval threshold not met")
    print(f"\nANSWER:\n{response.answer}")
    if response.sources:
        print(f"\nSOURCES ({len(response.sources)} chunks):")
        for chunk in response.sources[:3]:
            print(f"  score={chunk.relevance_score:.3f} | {chunk.content[:120]}...")
    print("=" * 70 + "\n")

# Main 

async def run() -> None:
    logger.info("Pipeline config | search_mode=%s | collection=%s", SEARCH_MODE, COLLECTION)

    # Step 1: Build LLM
    try:
        llm = LLMFactory.create_groq_pool()
        logger.info("Primary LLM: %s (GroqModelPool)", llm.model_name)
    except Exception as exc:
        logger.warning("Groq unavailable (%s) — using Gemini", exc)
        llm = LLMFactory.create_rate_limited("gemini")

    # Step 2: Build pipeline
    store = QdrantStore(
        collection_name=COLLECTION,
        in_memory=False,
        search_mode=SEARCH_MODE,
    )
    pipeline = RAGPipeline(
        llm=llm,
        store=store,
        cache=CacheManager(settings),
    )
    await pipeline.initialize()
    logger.info("Pipeline initialized")

    # Step 3: Health check
    health = await pipeline.health_check()
    if not health.ready:
        logger.error("Pipeline not ready — aborting | llm=%s | store=%s", health.llm, health.vector_store)
        await pipeline.shutdown()
        return

    # Step 4: Ingest PDF
    logger.info("Ingesting PDF: %s", PDF_PATH)
    result = await pipeline.ingest(file_path=PDF_PATH, collection=COLLECTION)
    logger.info(
        "Ingestion complete | stored=%d | total=%d | duplicates=%d",
        result.chunks_stored, result.total_chunks, result.duplicates_skipped,
    )

    # Step 5: Configure agents — enables automatic routing for complex queries
    pipeline.configure_agents(collections=COLLECTIONS, max_concurrent=3)
    logger.info("Agent layer configured | collections=%d", len(COLLECTIONS))

    # Step 6: Show routing preview
    print(f"\n{'─' * 70}")
    print("  ROUTING PREVIEW (complexity detector)")
    print(f"{'─' * 70}")
    for q in QUERIES:
        route = "→ AGENT" if should_decompose(q) else "→ SIMPLE RAG"
        print(f"  {route} | {q[:80]}{'...' if len(q) > 80 else ''}")
    print(f"{'─' * 70}\n")

    # Step 7: Run all queries — auto-routed by pipeline
    for q_idx, query_text in enumerate(QUERIES, start=1):
        logger.info("Running query %d/%d | route=%s | query='%s'",
                    q_idx, len(QUERIES),
                    "AGENT" if should_decompose(query_text) else "SIMPLE",
                    query_text[:80])

        q_start = time.perf_counter()
        response = await pipeline.query(
            PipelineQuery(
                query=query_text,
                collection=COLLECTION,
                top_k=5,
            )
        )
        wall_ms = (time.perf_counter() - q_start) * 1000

        _print_response(response, q_idx, query_text, wall_ms)

    await pipeline.shutdown()
    logger.info("Pipeline shutdown complete")


if __name__ == "__main__":
    asyncio.run(run())
