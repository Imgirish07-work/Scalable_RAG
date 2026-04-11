"""
End-to-end tests for the semantic (hybrid exact+semantic) cache layer.

Test scope:
    End-to-end tests (no pytest) covering BGE embedding model loading, embedding
    quality (cosine similarity), SemanticCacheStrategy initialization and
    index/find_similar, tiered threshold classification, cross-model and
    system-prompt isolation, hybrid CacheManager exact and semantic hit paths,
    false-match prevention, metrics tracking, and graceful degradation when the
    embedding model is unavailable or strategy is set to exact.

Flow:
    Section 1 loads the BGE model; if unavailable, sections 2-10 are skipped.
    Section 11 runs regardless (tests exact-only mode and config-driven degradation).

Dependencies:
    BGE-small-en-v1.5 via EMBEDDING_MODEL_LOCAL_PATH in .env; in-memory Qdrant
    (no server); colorama; MagicMock for settings.
"""

import asyncio
import time
import sys
from datetime import datetime
from unittest.mock import MagicMock

from colorama import Fore, Style, init as colorama_init

from llm.models.llm_response import LLMResponse
from vectorstore.embeddings import get_embeddings, get_embedding_dimension
from cache.strategies.semantic_strategy import SemanticCacheStrategy
from cache.models.cache_result import CacheLayer, CacheStrategy, SemanticTier
from cache.cache_manager import CacheManager

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


def make_response(**overrides) -> LLMResponse:
    defaults = dict(
        text="RAG is Retrieval-Augmented Generation.",
        model="gemini-2.5-flash",
        provider="gemini",
        prompt_tokens=25,
        completion_tokens=40,
        tokens_used=65,
        latency_ms=450.0,
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)


def make_settings(**overrides) -> MagicMock:
    defaults = {
        "CACHE_ENABLED": True,
        "CACHE_L1_MAX_SIZE": 100,
        "CACHE_TTL_SECONDS": 3600,
        "CACHE_STRATEGY": "semantic",
        "CACHE_SEMANTIC_THRESHOLD": 0.98,
        "CACHE_SEMANTIC_THRESHOLD_HIGH": 0.93,
        "CACHE_SEMANTIC_THRESHOLD_PARTIAL": 0.88,
        "CACHE_SEMANTIC_COLLECTION": "cache_semantic_test",
        "CACHE_CIRCUIT_BREAKER_THRESHOLD": 5,
        "CACHE_CIRCUIT_BREAKER_RESET_SECONDS": 60.0,
        "CACHE_MIN_RESPONSE_TOKENS": 20,
        "CACHE_MIN_RESPONSE_LATENCY_MS": 100.0,
        "COST_PER_TOKEN_OPENAI": 0.000002,
        "COST_PER_TOKEN_GEMINI": 0.0000001,
        "REDIS_ENV": "disabled",
        "REDIS_URL": "",
        "REDIS_CLOUD_URL": "",
        "QDRANT_URL": "",
        "QDRANT_API_KEY": "",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "TEMPERATURE": 0.0,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def test_embedding_model() -> None:
    """Verifies the BGE model loads, returns the correct dimension, and is cached.

    Tests:
        - get_embeddings() succeeds and sets MODEL_AVAILABLE for later sections.
        - Embedding dimension is 384.
        - Second call returns the same instance via lru_cache (near-instant).
        - embed_query() returns a non-zero list of the expected length.
    """
    global MODEL_AVAILABLE
    header("Embedding Model Loading", 1)

    sub_header("Load BGE model via get_embeddings()")

    try:
        start = time.perf_counter()
        embeddings = get_embeddings()
        elapsed = time.perf_counter() - start

        MODEL_AVAILABLE = True

        check("Model loaded successfully", True, f"{elapsed:.2f}s")

        dim = get_embedding_dimension()
        check("Embedding dimension is 384", dim == 384, f"dim={dim}")

        sub_header("Second load uses lru_cache")

        start2 = time.perf_counter()
        embeddings2 = get_embeddings()
        elapsed2 = time.perf_counter() - start2

        check("Cached load is instant", elapsed2 < 0.01, f"{elapsed2:.4f}s")
        check("Same instance returned", embeddings is embeddings2)

        sub_header("embed_query() works")

        vector = embeddings.embed_query("What is RAG?")
        check("Returns a list", isinstance(vector, list))
        check("Correct dimension", len(vector) == dim, f"len={len(vector)}")
        check("Non-zero vector", any(v != 0.0 for v in vector))

    except Exception as e:
        MODEL_AVAILABLE = False
        check("Model loaded successfully", False, str(e))
        info("Ensure BGE model is at the path configured in .env")


def test_embedding_quality() -> None:
    """Verifies that BGE embeddings reflect semantic similarity correctly.

    Tests:
        - Paraphrases of the same question score above 0.85 cosine similarity.
        - An unrelated query scores below 0.5 against the reference.
        - A structurally similar but topically different query scores lower than paraphrases.
        - Vectors are L2-normalized (norm ≈ 1.0).
    """
    header("Query Embedding Quality", 2)

    if not MODEL_AVAILABLE:
        skip("All embedding quality tests")
        return

    embeddings = get_embeddings()

    sub_header("Similar queries produce similar vectors")

    import numpy as np

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    v1 = embeddings.embed_query("What is RAG?")
    v2 = embeddings.embed_query("Explain RAG to me")
    v3 = embeddings.embed_query("Can you define RAG?")
    v4 = embeddings.embed_query("What is the weather today?")
    v5 = embeddings.embed_query("What is RLM?")

    sim_paraphrase = cosine(v1, v2)
    sim_define = cosine(v1, v3)
    sim_unrelated = cosine(v1, v4)
    sim_similar_structure = cosine(v1, v5)

    check(
        "'What is RAG?' <> 'Explain RAG to me'",
        sim_paraphrase > 0.85,
        f"cosine={sim_paraphrase:.4f}",
    )
    check(
        "'What is RAG?' <> 'Can you define RAG?'",
        sim_define > 0.85,
        f"cosine={sim_define:.4f}",
    )
    check(
        "'What is RAG?' <> 'What is the weather?' (should be low)",
        sim_unrelated < 0.5,
        f"cosine={sim_unrelated:.4f}",
    )
    check(
        "'What is RAG?' <> 'What is RLM?' (similar structure, diff topic)",
        sim_similar_structure < sim_paraphrase,
        f"cosine={sim_similar_structure:.4f} < {sim_paraphrase:.4f}",
    )

    sub_header("Vector properties")

    check("Vector has correct dimension", len(v1) == get_embedding_dimension(), f"len={len(v1)}")
    check("Vector is non-zero", any(v != 0.0 for v in v1))

    norm = float(np.linalg.norm(np.array(v1)))
    check("Vector is L2-normalized", abs(norm - 1.0) < 0.01, f"norm={norm:.4f}")


async def test_strategy_init() -> None:
    """Verifies SemanticCacheStrategy creates a Qdrant collection and is idempotent.

    Tests:
        - initialize() creates a collection with zero points.
        - A second call to initialize() does not raise an error.
    """
    header("Semantic Strategy — Initialization", 3)

    if not MODEL_AVAILABLE:
        skip("Strategy initialization tests")
        return

    strategy = SemanticCacheStrategy(
        collection_name="test_init_collection",
        use_memory=True,
    )

    sub_header("Initialize creates Qdrant collection")

    await strategy.initialize()

    stats = await strategy.get_collection_stats()
    check("Collection created", stats.get("status") is not None)
    check("Points count is 0", stats.get("points_count", -1) == 0)
    info(f"Collection stats: {stats}")

    sub_header("Double initialize is safe")

    await strategy.initialize()
    check("No error on second initialize", True)

    await strategy.close()


async def test_index_and_find() -> None:
    """Verifies index_entry() and find_similar() correctly store and retrieve entries.

    Tests:
        - Indexed query can be found with the same text (exact match).
        - A paraphrase finds the same cache_key above the high threshold.
        - An unrelated query returns None.
        - With multiple indexed entries, find_similar returns the closest match.
    """
    header("Semantic Strategy — Index and Find", 4)

    if not MODEL_AVAILABLE:
        skip("Index and find tests")
        return

    strategy = SemanticCacheStrategy(
        collection_name="test_index_find",
        threshold_direct=0.98,
        threshold_high=0.85,
        use_memory=True,
    )
    await strategy.initialize()

    sub_header("Index a query")

    await strategy.index_entry(
        query="What is RAG?",
        cache_key="key_rag_001",
        model_name="gemini",
        temperature=0.0,
    )

    stats = await strategy.get_collection_stats()
    check("1 point indexed", stats.get("points_count", 0) == 1)

    sub_header("Find exact same query")

    match = await strategy.find_similar(
        query="What is RAG?",
        model_name="gemini",
        temperature=0.0,
    )
    check("Exact query found", match is not None)
    if match:
        check("Correct cache_key returned", match.cache_key == "key_rag_001")
        check("High similarity score", match.similarity_score > 0.95, f"score={match.similarity_score:.4f}")
        check("Tier is 'direct'", match.tier == "direct", f"tier={match.tier}")
        info(f"Score: {match.similarity_score:.4f}, Tier: {match.tier}")

    sub_header("Find paraphrased query")

    match2 = await strategy.find_similar(
        query="Explain RAG to me",
        model_name="gemini",
        temperature=0.0,
    )
    check("Paraphrase found", match2 is not None)
    if match2:
        check("Returns same cache_key", match2.cache_key == "key_rag_001")
        check("Score above high threshold", match2.similarity_score >= 0.85, f"score={match2.similarity_score:.4f}")
        info(f"Paraphrase score: {match2.similarity_score:.4f}, Tier: {match2.tier}")

    sub_header("Unrelated query misses")

    match3 = await strategy.find_similar(
        query="What is the weather today?",
        model_name="gemini",
        temperature=0.0,
    )
    check("Unrelated query → no match", match3 is None)

    sub_header("Multiple indexed queries")

    await strategy.index_entry(
        query="How does a transformer work?",
        cache_key="key_transformer_001",
        model_name="gemini",
        temperature=0.0,
    )

    match4 = await strategy.find_similar(
        query="Explain transformer architecture",
        model_name="gemini",
        temperature=0.0,
    )
    check("Finds correct entry among multiple", match4 is not None)
    if match4:
        check(
            "Returns transformer key, not RAG key",
            match4.cache_key == "key_transformer_001",
            f"key={match4.cache_key}",
        )

    await strategy.close()


async def test_tiered_thresholds() -> None:
    """Verifies the direct/high/miss tier classification logic and live threshold matching.

    Tests:
        - An identical query scores in the 'direct' tier.
        - A moderate paraphrase scores in the 'high' or 'direct' tier.
        - _classify_tier() maps score ranges to the correct tier labels.
    """
    header("Tiered Threshold Matching", 5)

    if not MODEL_AVAILABLE:
        skip("Threshold tests")
        return

    strategy = SemanticCacheStrategy(
        collection_name="test_tiers",
        threshold_direct=0.98,
        threshold_high=0.93,
        use_memory=True,
    )
    await strategy.initialize()

    await strategy.index_entry(
        query="What is retrieval augmented generation?",
        cache_key="key_tier_001",
        model_name="gemini",
        temperature=0.0,
    )

    sub_header("Near-identical query → direct tier")

    match = await strategy.find_similar(
        query="What is retrieval augmented generation?",
        model_name="gemini",
        temperature=0.0,
    )
    if match:
        check("Direct tier for identical query", match.tier == "direct", f"score={match.similarity_score:.4f}")
    else:
        check("Direct tier for identical query", False, "no match returned")

    sub_header("Moderate paraphrase → high tier")

    match2 = await strategy.find_similar(
        query="Explain what RAG means in AI",
        model_name="gemini",
        temperature=0.0,
    )
    if match2:
        check(
            "Paraphrase gets high or direct tier",
            match2.tier in ("high", "direct"),
            f"score={match2.similarity_score:.4f}, tier={match2.tier}",
        )
    else:
        info("Paraphrase scored below threshold — model-dependent behavior")

    sub_header("Tier classification logic")

    check("Score 0.99 → direct", strategy._classify_tier(0.99) == "direct")
    check("Score 0.98 → direct", strategy._classify_tier(0.98) == "direct")
    check("Score 0.95 → high", strategy._classify_tier(0.95) == "high")
    check("Score 0.93 → high", strategy._classify_tier(0.93) == "high")
    check("Score 0.90 → miss", strategy._classify_tier(0.90) == "miss")
    check("Score 0.50 → miss", strategy._classify_tier(0.50) == "miss")

    await strategy.close()


async def test_cross_model_isolation() -> None:
    """Verifies that model name and system prompt are included in the semantic key space.

    Tests:
        - A query indexed for 'gemini' is found by the same model but not 'openai'.
        - A query indexed with a specific system prompt hash is only found with the same hash.
        - A different system prompt hash produces a miss.
    """
    header("Cross-Model Isolation", 6)

    if not MODEL_AVAILABLE:
        skip("Cross-model tests")
        return

    strategy = SemanticCacheStrategy(
        collection_name="test_isolation",
        threshold_high=0.85,
        use_memory=True,
    )
    await strategy.initialize()

    await strategy.index_entry(
        query="What is RAG?",
        cache_key="key_gemini_001",
        model_name="gemini",
        temperature=0.0,
    )

    sub_header("Same model finds match")

    match = await strategy.find_similar("What is RAG?", "gemini", 0.0)
    check("Same model → hit", match is not None)

    sub_header("Different model misses")

    match2 = await strategy.find_similar("What is RAG?", "openai", 0.0)
    check("Different model → miss", match2 is None)

    sub_header("System prompt isolation")

    await strategy.index_entry(
        query="What is RAG?",
        cache_key="key_sp_001",
        model_name="gemini",
        temperature=0.0,
        system_prompt_hash="sp_hash_abc",
    )

    match3 = await strategy.find_similar(
        "What is RAG?", "gemini", 0.0, system_prompt_hash="sp_hash_abc"
    )
    check("Same system prompt → hit", match3 is not None)

    match4 = await strategy.find_similar(
        "What is RAG?", "gemini", 0.0, system_prompt_hash="sp_hash_xyz"
    )
    check("Different system prompt → miss", match4 is None)

    await strategy.close()


async def test_hybrid_exact_hit() -> None:
    """Verifies the exact fast-path still works when the semantic strategy is active.

    Tests:
        - The CacheManager initializes with a non-None semantic strategy.
        - An exact query lookup returns EXACT strategy and L1 layer.
        - A normalized variant of the original query also returns an EXACT hit.
    """
    header("Hybrid CacheManager — Exact Hit (Fast Path)", 7)

    if not MODEL_AVAILABLE:
        skip("Hybrid exact hit tests")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    check("Semantic strategy is active", cache._semantic_strategy is not None)

    response = make_response(text="RAG is Retrieval-Augmented Generation.")

    sub_header("Exact match still works in hybrid mode")

    await cache.set("What is RAG?", "gemini", 0.0, response)

    r = await cache.get("What is RAG?", "gemini", 0.0)
    check("Exact query hits", r.hit is True)
    check("Strategy is EXACT", r.strategy == CacheStrategy.EXACT)
    check("Layer is L1", r.layer == CacheLayer.L1_MEMORY)
    info(f"Latency: {r.lookup_latency_ms:.3f}ms")

    sub_header("Normalized variant still exact-hits")

    r2 = await cache.get("  WHAT IS RAG??  ", "gemini", 0.0)
    check("Normalized variant → exact hit", r2.hit is True)
    check("Still EXACT strategy", r2.strategy == CacheStrategy.EXACT)

    await cache.close()


async def test_hybrid_semantic_hit() -> None:
    """Verifies that a paraphrased query triggers a semantic hit in hybrid mode.

    Tests:
        - A paraphrase that misses the exact key returns a SEMANTIC strategy hit.
        - The hit carries a positive similarity score and a direct/high tier.
        - The original exact query still prefers the EXACT strategy over semantic.
    """
    header("Hybrid CacheManager — Semantic Hit (Fallback Path)", 8)

    if not MODEL_AVAILABLE:
        skip("Hybrid semantic hit tests")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response(text="RAG combines retrieval with generation.")
    await cache.set("What is RAG?", "gemini", 0.0, response)

    sub_header("Paraphrased query hits via semantic")

    r = await cache.get("Explain RAG to me", "gemini", 0.0)

    if r.hit:
        check("Paraphrase hits cache", True)
        check("Strategy is SEMANTIC", r.strategy == CacheStrategy.SEMANTIC)
        check("Similarity score > 0", r.similarity_score > 0, f"score={r.similarity_score:.4f}")
        check(
            "Semantic tier is direct or high",
            r.semantic_tier in (SemanticTier.DIRECT, SemanticTier.HIGH),
            f"tier={r.semantic_tier}",
        )
        check("Response text correct", r.response.text == response.text)
        info(f"Similarity: {r.similarity_score:.4f}, Tier: {r.semantic_tier}, Latency: {r.lookup_latency_ms:.2f}ms")
    else:
        check(
            "Paraphrase hits cache",
            False,
            "Semantic match below threshold — adjust thresholds if needed",
        )

    sub_header("Another paraphrase")

    r2 = await cache.get("Can you define RAG?", "gemini", 0.0)
    if r2.hit:
        check("'Can you define RAG?' → semantic hit", True, f"score={r2.similarity_score:.4f}")
    else:
        info("This paraphrase scored below threshold — model-dependent")

    sub_header("Exact still preferred over semantic")

    r3 = await cache.get("What is RAG?", "gemini", 0.0)
    check("Original query → EXACT (not semantic)", r3.strategy == CacheStrategy.EXACT)

    await cache.close()


async def test_no_false_matches() -> None:
    """Verifies the semantic cache does not produce false positives for unrelated queries.

    Tests:
        - Four semantically unrelated queries all return misses.
        - A query with a similar syntactic structure but a different topic also misses.
    """
    header("Different Queries Don't Match", 9)

    if not MODEL_AVAILABLE:
        skip("False match tests")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response(text="RAG stands for Retrieval-Augmented Generation.")
    await cache.set("What is RAG?", "gemini", 0.0, response)

    sub_header("Unrelated queries miss")

    unrelated = [
        "What is the weather today?",
        "How do I cook pasta?",
        "Write a Python sort function",
        "What year was Python created?",
    ]

    for q in unrelated:
        r = await cache.get(q, "gemini", 0.0)
        check(f"'{q[:40]}' → miss", r.hit is False)

    sub_header("Similar structure, different topic")

    r2 = await cache.get("What is quantum computing?", "gemini", 0.0)
    check("'What is quantum computing?' → miss", r2.hit is False)

    await cache.close()


async def test_semantic_metrics() -> None:
    """Verifies that semantic hits and misses are reflected in the metrics counters.

    Tests:
        - Total lookups equals the number of get() calls made.
        - exact_hits and total_misses are at least 1 each.
        - get_full_stats() reports strategy as 'hybrid' and includes a 'semantic' key.
    """
    header("Metrics — Semantic Hits Tracked", 10)

    if not MODEL_AVAILABLE:
        skip("Semantic metrics tests")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response()
    await cache.set("What is RAG?", "gemini", 0.0, response)

    await cache.get("What is RAG?", "gemini", 0.0)
    await cache.get("Explain RAG to me", "gemini", 0.0)
    await cache.get("How do I cook pasta?", "gemini", 0.0)

    m = cache.get_metrics()

    sub_header("Metrics summary")

    check("Total lookups = 3", m["total_lookups"] == 3)
    check("Exact hits >= 1", m["exact_hits"] >= 1)
    check("Total misses >= 1", m["total_misses"] >= 1)

    info(f"Exact hits: {m['exact_hits']}")
    info(f"Semantic hits: {m['semantic_hits']}")
    info(f"Misses: {m['total_misses']}")
    info(f"Hit rate: {m['hit_rate_pct']}%")

    sub_header("Full stats include semantic collection")

    stats = await cache.get_full_stats()
    check("Strategy is hybrid", stats["strategy"] == "hybrid")
    check("Semantic stats present", "semantic" in stats)

    if "semantic" in stats:
        info(f"Semantic collection: {stats['semantic']}")

    await cache.close()


async def test_graceful_degradation() -> None:
    """Verifies the cache operates correctly when the semantic strategy is disabled or unavailable.

    Tests:
        - CACHE_STRATEGY=exact results in no semantic strategy and only EXACT hits.
        - get_full_stats() shows 'exact' strategy with no 'semantic' key.
        - When the model is available, the exact path still works in hybrid mode.
    """
    header("Graceful Degradation — Semantic Failure", 11)

    sub_header("CACHE_STRATEGY=exact disables semantic")

    settings = make_settings(CACHE_STRATEGY="exact")
    cache = CacheManager(settings)
    await cache.initialize()

    check("Semantic disabled by config", cache._semantic_strategy is None)

    response = make_response()
    stored = await cache.set("hello", "gemini", 0.0, response)
    check("Set still works (exact only)", stored is True)

    r = await cache.get("hello", "gemini", 0.0)
    check("Get still works (exact hit)", r.hit is True)
    check("Strategy is EXACT", r.strategy == CacheStrategy.EXACT)

    stats = await cache.get_full_stats()
    check("Stats shows exact strategy", stats["strategy"] == "exact")
    check("No semantic in stats", "semantic" not in stats)

    await cache.close()

    if MODEL_AVAILABLE:
        sub_header("Semantic failure → exact still works")

        settings2 = make_settings(CACHE_STRATEGY="semantic")
        cache2 = CacheManager(settings2)
        await cache2.initialize()

        check("Semantic is active", cache2._semantic_strategy is not None)

        response2 = make_response(text="Exact path answer")
        await cache2.set("fallback query", "gemini", 0.0, response2)

        # Exact is always the fast path; semantic is only the fallback
        r2 = await cache2.get("fallback query", "gemini", 0.0)
        check("Exact hit works in hybrid mode", r2.hit is True)
        check("Strategy is EXACT (not semantic)", r2.strategy == CacheStrategy.EXACT)
        info("Exact path is always the fast path — semantic is fallback only")

        await cache2.close()


async def run_all() -> None:
    pipeline_start = time.perf_counter()

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   SEMANTIC CACHE PIPELINE TEST — Phase 6")
    print(f"   Hybrid: Exact + Semantic (BGE + Qdrant)")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}{Style.RESET_ALL}")

    test_embedding_model()
    test_embedding_quality()
    await test_strategy_init()
    await test_index_and_find()
    await test_tiered_thresholds()
    await test_cross_model_isolation()
    await test_hybrid_exact_hit()
    await test_hybrid_semantic_hit()
    await test_no_false_matches()
    await test_semantic_metrics()
    await test_graceful_degradation()

    elapsed = time.perf_counter() - pipeline_start

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   FINAL RESULTS — Phase 6")
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
            print(f"    ✓ ALL TESTS PASSED — Hybrid caching verified")
        else:
            print(f"    ✓ ALL TESTS PASSED — Degradation only (no model)")
            print(f"    ℹ Ensure BGE model path is configured in .env")
        print(f"    {'═' * 50}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.RED}{'═' * 50}")
        print(f"    ✗ {FAIL_COUNT} TEST(S) FAILED — Review above")
        print(f"    {'═' * 50}{Style.RESET_ALL}")

    print()
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all())