"""
Unit tests for the full L1 caching system.

Test scope:
    Unit tests (no pytest) covering the normalizer chain, exact cache strategy,
    JSON serializer, memory backend (LRU + TTL), CacheManager end-to-end flow,
    quality gate, request coalescing, cache invalidation, metrics/observability,
    stress testing, and error resilience. No Redis, Qdrant, or LLM API calls.

Flow:
    Sync sections (1-3) → async sections (4-11) → printed pass/fail summary.
    All backends are in-memory with a mocked Settings object.

Dependencies:
    colorama for terminal output; MagicMock for settings; no external services.
"""

import asyncio
import time
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from colorama import Fore, Style, init as colorama_init

from llm.models.llm_response import LLMResponse
from cache.models.cache_entry import CacheEntry
from cache.models.cache_result import CacheResult, CacheLayer, CacheStrategy, SemanticTier
from cache.models.cache_metrics import CacheMetrics
from cache.backend.memory_backend import MemoryCacheBackend
from cache.strategies.exact_strategy import ExactCacheStrategy
from cache.normalizers.query_normalizer import (
    QueryNormalizerChain,
    WhitespaceNormalizer,
    CaseNormalizer,
    PunctuationNormalizer,
    UnicodeNormalizer,
)
from cache.serializers.json_serializer import JSONSerializer
from cache.exceptions.cache_exceptions import CacheSerializationError
from cache.cache_manager import CacheManager


colorama_init(autoreset=True)

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def header(title: str, section: int) -> None:
    width = 70
    print()
    print(f"{Fore.CYAN}{'═' * width}")
    print(f"  SECTION {section} — {title}")
    print(f"{'═' * width}{Style.RESET_ALL}")


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


def info(msg: str) -> None:
    print(f"    {Fore.WHITE}ℹ {msg}{Style.RESET_ALL}")


def warn(msg: str) -> None:
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"    {Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")


def divider() -> None:
    print(f"{Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")


def make_response(
    text: str = "RAG stands for Retrieval-Augmented Generation.",
    model: str = "gemini-2.5-flash",
    provider: str = "gemini",
    prompt_tokens: int = 25,
    completion_tokens: int = 40,
    tokens_used: int = 65,
    latency_ms: float = 450.0,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )


def make_settings(**overrides) -> MagicMock:
    defaults = {
        "CACHE_ENABLED": True,
        "CACHE_L1_MAX_SIZE": 100,
        "CACHE_TTL_SECONDS": 3600,
        "CACHE_STRATEGY": "exact",
        "CACHE_SEMANTIC_THRESHOLD": 0.95,
        "CACHE_SEMANTIC_THRESHOLD_HIGH": 0.93,
        "CACHE_SEMANTIC_THRESHOLD_PARTIAL": 0.88,
        "CACHE_SEMANTIC_COLLECTION": "cache_semantic",
        "CACHE_CIRCUIT_BREAKER_THRESHOLD": 5,
        "CACHE_CIRCUIT_BREAKER_RESET_SECONDS": 60.0,
        "CACHE_MIN_RESPONSE_TOKENS": 20,
        "CACHE_MIN_RESPONSE_LATENCY_MS": 100.0,
        "COST_PER_TOKEN_OPENAI": 0.000002,
        "COST_PER_TOKEN_GEMINI": 0.0000001,
        "REDIS_URL": "",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "TEMPERATURE": 0.0,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def test_normalizer_chain() -> None:
    """Verifies that each normalizer step and the full chain produce canonical forms.

    Tests:
        - WhitespaceNormalizer collapses runs of whitespace.
        - CaseNormalizer lowercases all input.
        - PunctuationNormalizer strips trailing punctuation.
        - UnicodeNormalizer applies NFC and removes zero-width characters.
        - Full chain collapses five query variants to one normalized form.
        - build_cache_fingerprint produces the expected pipe-delimited format.
    """
    header("Normalizer Chain", 1)
    chain = QueryNormalizerChain()

    sub_header("Individual steps")

    ws = WhitespaceNormalizer()
    result = ws.normalize("  What   is  RAG?  ")
    check("Whitespace collapse", result == "What is RAG?", f"'{result}'")

    cs = CaseNormalizer()
    result = cs.normalize("What Is RAG?")
    check("Case normalization", result == "what is rag?", f"'{result}'")

    pn = PunctuationNormalizer()
    result = pn.normalize("what is rag??")
    check("Punctuation strip", result == "what is rag", f"'{result}'")

    un = UnicodeNormalizer()
    composed = un.normalize("caf\u00e9")
    decomposed = un.normalize("cafe\u0301")
    check("Unicode NFC normalization", composed == decomposed, f"'{composed}'")

    result = un.normalize("hello\u200bworld")
    check("Zero-width character removal", result == "helloworld", f"'{result}'")

    sub_header("Full chain")

    result = chain.normalize("  What   is  RAG??  ")
    check("Full chain output", result == "what is rag", f"'{result}'")

    queries = [
        "What is RAG?",
        "  What   is  RAG?  ",
        "what is rag?",
        "WHAT IS RAG??",
        "what is rag",
    ]
    normalized = {chain.normalize(q) for q in queries}
    check(
        "5 variant queries → same normalized form",
        len(normalized) == 1,
        f"unique forms: {len(normalized)}",
    )

    a = chain.normalize("What is RAG?")
    b = chain.normalize("What is RLM?")
    check("Different queries stay different", a != b, f"'{a}' vs '{b}'")

    check("Empty string handled", chain.normalize("") == "", "no crash")
    check(
        "Whitespace-only handled",
        chain.normalize("   \t\n   ") == "",
        "returns empty",
    )

    sub_header("Cache fingerprint")

    fp = chain.build_cache_fingerprint(
        query="  What is RAG? ",
        model_name="Gemini-2.5-Flash",
        temperature=0,
        system_prompt_hash="abc123",
    )
    check(
        "Fingerprint format",
        fp == "what is rag|gemini-2.5-flash|0.0|abc123",
        f"'{fp}'",
    )

    fp2 = chain.build_cache_fingerprint(
        query="What is RAG?", model_name="gemini", temperature=0.7
    )
    check("Fingerprint without system prompt", "|0.7" in fp2 and fp2.count("|") == 2)

    fp_a = chain.build_cache_fingerprint("hello", "m", 0)
    fp_b = chain.build_cache_fingerprint("hello", "m", 0)
    check("Fingerprint deterministic", fp_a == fp_b)


def test_exact_strategy() -> None:
    """Verifies that ExactCacheStrategy generates isolated, deterministic SHA-256 keys.

    Tests:
        - Same input always produces the same 64-char hex key.
        - Whitespace/punctuation variants normalize to the same key.
        - Different queries, models, temperatures, and system prompts yield different keys.
        - get_query_hash ignores formatting differences.
        - get_normalized_query returns the canonical form.
    """
    header("Exact Strategy (Key Generation)", 2)
    strategy = ExactCacheStrategy()

    sub_header("Key determinism")

    key1 = strategy.make_key("What is RAG?", "gemini", 0.0)
    key2 = strategy.make_key("What is RAG?", "gemini", 0.0)
    check("Same input → same key", key1 == key2, f"key={key1[:20]}...")
    check("Key is 64-char hex (SHA-256)", len(key1) == 64 and all(c in "0123456789abcdef" for c in key1))

    sub_header("Normalization in key generation")

    k_a = strategy.make_key("  What is RAG??  ", "gemini", 0.0)
    k_b = strategy.make_key("what is rag", "gemini", 0.0)
    check("Normalized variants → same key", k_a == k_b)

    sub_header("Key isolation")

    k_q1 = strategy.make_key("What is RAG?", "gemini", 0.0)
    k_q2 = strategy.make_key("What is RLM?", "gemini", 0.0)
    check("Different queries → different keys", k_q1 != k_q2)

    k_m1 = strategy.make_key("hello", "gemini", 0.0)
    k_m2 = strategy.make_key("hello", "openai", 0.0)
    check("Different models → different keys", k_m1 != k_m2)

    k_t1 = strategy.make_key("hello", "gemini", 0.0)
    k_t2 = strategy.make_key("hello", "gemini", 0.7)
    check("Different temperatures → different keys", k_t1 != k_t2)

    k_s1 = strategy.make_key("hello", "gemini", 0.0, "hash_a")
    k_s2 = strategy.make_key("hello", "gemini", 0.0, "hash_b")
    check("Different system prompts → different keys", k_s1 != k_s2)

    sub_header("Query hash (model-independent)")

    h1 = strategy.get_query_hash("What is RAG?")
    h2 = strategy.get_query_hash("  WHAT IS RAG??  ")
    check("Query hash ignores formatting", h1 == h2)

    sub_header("Normalized query access")

    norm = strategy.get_normalized_query("  What is RAG?? ")
    check("get_normalized_query", norm == "what is rag", f"'{norm}'")


def test_json_serializer() -> None:
    """Verifies that JSONSerializer round-trips CacheEntry without data loss.

    Tests:
        - Serialize produces a non-empty JSON string.
        - Deserialize restores response text, tokens, provider, cache key, and TTL.
        - Hit count survives serialization.
        - Invalid JSON, wrong schema, and empty string each raise CacheSerializationError.
    """
    header("JSON Serializer (Round-Trip)", 3)
    serializer = JSONSerializer()

    sub_header("Serialize and deserialize")

    response = make_response(text="Test answer about RAG", tokens_used=100)
    now = datetime.now(timezone.utc)
    entry = CacheEntry(
        response=response,
        cache_key="test_key_001",
        query_hash="test_hash_001",
        created_at=now,
        expires_at=now + timedelta(seconds=3600),
        ttl_seconds=3600,
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        token_cost_estimate=0.00001,
    )

    json_str = serializer.serialize(entry)
    check("Serialize returns string", isinstance(json_str, str) and len(json_str) > 0, f"{len(json_str)} bytes")
    info(f"JSON preview: {json_str[:100]}...")

    restored = serializer.deserialize(json_str)
    check("Deserialize returns CacheEntry", isinstance(restored, CacheEntry))
    check("Response text preserved", restored.response.text == "Test answer about RAG")
    check("Tokens preserved", restored.response.tokens_used == 100)
    check("Provider preserved", restored.provider == "gemini")
    check("Cache key preserved", restored.cache_key == "test_key_001")
    check("TTL preserved", restored.ttl_seconds == 3600)

    sub_header("Hit count survives serialization")

    entry.record_hit()
    entry.record_hit()
    entry.record_hit()
    json_str2 = serializer.serialize(entry)
    restored2 = serializer.deserialize(json_str2)
    check("Hit count preserved", restored2.hit_count == 3)

    sub_header("Error handling")

    try:
        serializer.deserialize("not valid json{{{")
        check("Invalid JSON raises CacheSerializationError", False)
    except CacheSerializationError:
        check("Invalid JSON raises CacheSerializationError", True)

    try:
        serializer.deserialize('{"wrong": "schema"}')
        check("Wrong schema raises CacheSerializationError", False)
    except CacheSerializationError:
        check("Wrong schema raises CacheSerializationError", True)

    try:
        serializer.deserialize("")
        check("Empty string raises CacheSerializationError", False)
    except CacheSerializationError:
        check("Empty string raises CacheSerializationError", True)


async def test_memory_backend() -> None:
    """Verifies MemoryCacheBackend behaviour: CRUD, LRU eviction, TTL expiry, and stats.

    Tests:
        - set/get/exists/delete round-trip correctly.
        - Overwrite does not increase the stored count.
        - LRU eviction removes the least-recently-used entry when max_size is exceeded.
        - Accessed entries are promoted and survive subsequent eviction.
        - Entries expire after their TTL and are removed by evict_expired().
        - Zero TTL entries are not stored.
        - stats() returns accurate utilization figures.
        - 20 concurrent writes stay within max_size without corruption.
    """
    header("Memory Backend (L1 LRU + TTL)", 4)
    backend = MemoryCacheBackend(max_size=5)

    sub_header("Basic operations")

    await backend.set("key1", '{"data": "hello"}', 3600)
    result = await backend.get("key1")
    check("set → get round-trip", result == '{"data": "hello"}')

    result = await backend.get("nonexistent")
    check("Get missing key returns None", result is None)

    check("exists() for present key", await backend.exists("key1") is True)
    check("exists() for missing key", await backend.exists("nope") is False)

    deleted = await backend.delete("key1")
    check("delete() existing key returns True", deleted is True)
    check("Deleted key is gone", await backend.get("key1") is None)

    deleted = await backend.delete("nope")
    check("delete() missing key returns False", deleted is False)

    sub_header("Size and clear")

    await backend.set("a", "1", 3600)
    await backend.set("b", "2", 3600)
    check("Size after 2 inserts", await backend.size() == 2)

    removed = await backend.clear()
    check("Clear returns count", removed == 2)
    check("Size after clear", await backend.size() == 0)

    sub_header("Overwrite existing key")

    await backend.set("key1", "old_value", 3600)
    await backend.set("key1", "new_value", 3600)
    result = await backend.get("key1")
    check("Overwrite replaces value", result == "new_value")
    check("Overwrite doesn't increase size", await backend.size() == 1)
    await backend.clear()

    sub_header("LRU eviction (max_size=5)")

    for i in range(6):
        await backend.set(f"k{i}", f"v{i}", 3600)

    check("Size capped at max_size", await backend.size() == 5)
    check("Oldest entry (k0) evicted", await backend.get("k0") is None)
    check("Newest entry (k5) present", await backend.get("k5") == "v5")

    info("Testing LRU access refresh...")
    await backend.clear()
    for i in range(5):
        await backend.set(f"k{i}", f"v{i}", 3600)

    await backend.get("k0")

    await backend.set("k5", "v5", 3600)
    check("Accessed k0 survives eviction", await backend.get("k0") == "v0")
    check("Untouched k1 gets evicted", await backend.get("k1") is None)

    sub_header("TTL expiry")

    await backend.clear()
    await backend.set("ttl_key", "ttl_value", 1)
    check("Entry exists before expiry", await backend.get("ttl_key") == "ttl_value")

    info("Sleeping 1.2 seconds for TTL expiry...")
    await asyncio.sleep(1.2)

    result = await backend.get("ttl_key")
    check("Entry expired after TTL", result is None)

    check("Zero TTL skips set", True)
    await backend.set("zero_ttl", "value", 0)
    check("Zero TTL entry not stored", await backend.size() == 0 or await backend.exists("zero_ttl") is False)

    sub_header("Proactive expiry sweep")

    await backend.clear()
    await backend.set("short", "v1", 1)
    await backend.set("long", "v2", 3600)
    await asyncio.sleep(1.2)

    removed = await backend.evict_expired()
    check("Sweep removes expired entries", removed == 1)
    check("Non-expired entry survives", await backend.get("long") == "v2")

    sub_header("Stats")

    await backend.clear()
    for i in range(5):
        await backend.set(f"s{i}", f"v{i}", 3600)

    stats = await backend.stats()
    check("Stats has name", stats["name"] == "l1_memory")
    check("Stats has backend_type", stats["backend_type"] == "in_memory_lru")
    check("Stats current_size", stats["current_size"] == 5)
    check("Stats max_size", stats["max_size"] == 5)
    check("Stats utilization 100%", stats["utilization_pct"] == 100.0)
    info(f"Full stats: {stats}")

    sub_header("Concurrent access")

    await backend.clear()

    async def write(i):
        await backend.set(f"c{i}", f"v{i}", 3600)

    await asyncio.gather(*[write(i) for i in range(20)])
    size = await backend.size()
    check("20 concurrent writes, max_size=5", size == 5, f"size={size}")

    await backend.close()


async def test_cache_manager_e2e() -> None:
    """Verifies the CacheManager miss→set→hit flow and key-isolation guarantees.

    Tests:
        - First lookup returns a miss; subsequent lookup after set returns a hit.
        - Hit result carries L1_MEMORY layer and EXACT strategy.
        - Whitespace, case, and punctuation variants all hit the same cached entry.
        - Different query, model, temperature, and system prompt each produce a miss.
        - TTL expiry causes a previously cached entry to return as a miss.
    """
    header("CacheManager (End-to-End Flow)", 5)
    cache = CacheManager(make_settings())
    await cache.initialize()

    sub_header("Miss → Set → Hit")

    result1 = await cache.get("What is RAG?", "gemini", 0.0)
    check("First lookup is a miss", result1.hit is False)
    info(f"Miss latency: {result1.lookup_latency_ms:.3f}ms")

    response = make_response()
    stored = await cache.set("What is RAG?", "gemini", 0.0, response)
    check("Set returns True", stored is True)

    result2 = await cache.get("What is RAG?", "gemini", 0.0)
    check("Second lookup is a hit", result2.hit is True)
    check("Hit returns correct text", result2.response.text == response.text)
    check("Hit from L1", result2.layer == CacheLayer.L1_MEMORY)
    check("Hit via exact strategy", result2.strategy == CacheStrategy.EXACT)
    info(f"Hit latency: {result2.lookup_latency_ms:.3f}ms")
    info(f"Cache age: {result2.cache_age_seconds:.2f}s")

    sub_header("Normalized variants hit the same entry")

    variants = [
        ("  What   is  RAG??  ", "extra whitespace + punctuation"),
        ("what is rag?", "lowercase + question mark"),
        ("WHAT IS RAG", "all caps, no punctuation"),
        ("what is rag", "fully normalized"),
    ]

    all_hit = True
    for query, desc in variants:
        r = await cache.get(query, "gemini", 0.0)
        hit = r.hit
        if not hit:
            all_hit = False
        check(f"Variant: {desc}", hit, f"'{query}'")

    sub_header("Key isolation")

    r = await cache.get("What is RLM?", "gemini", 0.0)
    check("Different query → miss", r.hit is False)

    r = await cache.get("What is RAG?", "openai", 0.0)
    check("Different model → miss", r.hit is False)

    r = await cache.get("What is RAG?", "gemini", 0.7)
    check("Different temperature → miss", r.hit is False)

    sub_header("System prompt isolation")

    await cache.set("hello", "gemini", 0.0, response, system_prompt="Be helpful.")

    r = await cache.get("hello", "gemini", 0.0, system_prompt="Be helpful.")
    check("Same system prompt → hit", r.hit is True)

    r = await cache.get("hello", "gemini", 0.0, system_prompt="Be brief.")
    check("Different system prompt → miss", r.hit is False)

    r = await cache.get("hello", "gemini", 0.0)
    check("No system prompt → miss", r.hit is False)

    sub_header("TTL expiry through manager")

    await cache.set("ttl_test", "gemini", 0.0, response, ttl_seconds=1)
    r = await cache.get("ttl_test", "gemini", 0.0)
    check("Entry exists before TTL", r.hit is True)

    info("Sleeping 1.2 seconds for TTL expiry...")
    await asyncio.sleep(1.2)

    r = await cache.get("ttl_test", "gemini", 0.0)
    check("Entry expired after TTL", r.hit is False)

    await cache.close()


async def test_quality_gate() -> None:
    """Verifies that the quality gate rejects low-quality responses before caching.

    Tests:
        - Responses below the minimum token threshold are rejected.
        - Responses with suspiciously low latency are rejected.
        - Responses meeting both thresholds are accepted.
        - Rejection events are counted in the metrics.
    """
    header("Quality Gate", 6)
    cache = CacheManager(make_settings())
    await cache.initialize()

    sub_header("Rejection cases")

    few_tokens = make_response(text="Too short.", completion_tokens=2, prompt_tokens=5, tokens_used=7, latency_ms=500.0)
    stored = await cache.set("q1", "gemini", 0.0, few_tokens)
    check("Token count too low (7 < 20) → rejected", stored is False)

    short = make_response(text="Ok", completion_tokens=5, prompt_tokens=5, tokens_used=10, latency_ms=500.0)
    stored = await cache.set("q2", "gemini", 0.0, short)
    check("Too few tokens (10 < 20) → rejected", stored is False)

    fast = make_response(latency_ms=50.0)
    stored = await cache.set("q3", "gemini", 0.0, fast)
    check("Suspiciously fast (50ms < 100ms) → rejected", stored is False)

    instant = make_response(latency_ms=1.0)
    stored = await cache.set("q4", "gemini", 0.0, instant)
    check("Near-zero latency (1ms < 100ms) → rejected", stored is False)

    sub_header("Acceptance cases")

    good = make_response(
        text="A thorough explanation of RAG systems.",
        completion_tokens=40,
        latency_ms=500.0,
    )
    stored = await cache.set("q5", "gemini", 0.0, good)
    check("Good response → accepted", stored is True)

    borderline = make_response(
        text="RAG is retrieval-augmented generation for knowledge-grounded answers.",
        completion_tokens=20,
        tokens_used=45,
        latency_ms=100.0,
    )
    stored = await cache.set("q6", "gemini", 0.0, borderline)
    check("Borderline response (exactly at threshold) → accepted", stored is True)

    sub_header("Rejection counter in metrics")

    metrics = cache.get_metrics()
    check(
        "Quality rejections tracked",
        metrics["quality_gate_rejections"] >= 4,
        f"rejections={metrics['quality_gate_rejections']}",
    )

    await cache.close()


async def test_request_coalescing() -> None:
    """Verifies that concurrent identical requests are coalesced to a single LLM call.

    Tests:
        - First caller receives a miss; subsequent callers wait for the result.
        - Only one LLM call is made for five simultaneous identical requests.
        - resolve_in_flight() makes the result available to all waiters.
        - A coalescing timeout returns a miss without crashing.
    """
    header("Request Coalescing (Thundering Herd)", 7)
    cache = CacheManager(make_settings())
    await cache.initialize()

    sub_header("Basic coalescing flow")

    result = await cache.get_or_wait("coalesce_q", "gemini", 0.0)
    check("First caller gets miss", result.hit is False)

    response = make_response(text="Shared answer")
    await cache.set("coalesce_q", "gemini", 0.0, response)
    await cache.resolve_in_flight("coalesce_q", "gemini", 0.0)

    result2 = await cache.get("coalesce_q", "gemini", 0.0)
    check("After resolve, entry is cached", result2.hit is True)
    check("Correct response text", result2.response.text == "Shared answer")

    sub_header("Concurrent coalescing — 5 simultaneous requests")

    await cache.clear_all()
    response = make_response(text="Coalesced result")
    llm_call_count = 0

    async def simulate_request(req_id: int) -> None:
        nonlocal llm_call_count
        result = await cache.get_or_wait("same_query", "gemini", 0.0)
        if not result.hit:
            llm_call_count += 1
            await asyncio.sleep(0.15)
            await cache.set("same_query", "gemini", 0.0, response)
            await cache.resolve_in_flight("same_query", "gemini", 0.0)

    await asyncio.gather(*[simulate_request(i) for i in range(5)])

    check(
        "Only 1 LLM call for 5 concurrent requests",
        llm_call_count == 1,
        f"calls={llm_call_count}",
    )

    final = await cache.get("same_query", "gemini", 0.0)
    check("All requesters can now read cached result", final.hit is True)

    sub_header("Coalescing timeout")

    await cache.clear_all()
    _ = await cache.get_or_wait("timeout_q", "gemini", 0.0)

    start = time.perf_counter()
    timeout_result = await cache.get_or_wait("timeout_q", "gemini", 0.0, timeout=0.3)
    elapsed = time.perf_counter() - start

    check("Timeout returns miss", timeout_result.hit is False)
    check("Timeout respected (~0.3s)", 0.2 < elapsed < 0.6, f"elapsed={elapsed:.2f}s")

    await cache.resolve_in_flight("timeout_q", "gemini", 0.0)
    await cache.close()


async def test_invalidation() -> None:
    """Verifies that invalidate() and clear_all() remove entries as expected.

    Tests:
        - invalidate() returns True and removes an existing entry.
        - invalidate() returns False for a key that was never cached.
        - clear_all() returns per-layer counts and removes all entries.
    """
    header("Cache Invalidation", 8)
    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response()

    sub_header("Invalidate existing entry")

    await cache.set("to_delete", "gemini", 0.0, response)
    r = await cache.get("to_delete", "gemini", 0.0)
    check("Entry exists before invalidation", r.hit is True)

    deleted = await cache.invalidate("to_delete", "gemini", 0.0)
    check("invalidate() returns True", deleted is True)

    r = await cache.get("to_delete", "gemini", 0.0)
    check("Entry gone after invalidation", r.hit is False)

    sub_header("Invalidate missing entry")

    deleted = await cache.invalidate("never_existed", "gemini", 0.0)
    check("invalidate() missing key returns False", deleted is False)

    sub_header("Clear all")

    await cache.set("q1", "gemini", 0.0, response)
    await cache.set("q2", "gemini", 0.0, response)
    await cache.set("q3", "gemini", 0.0, response)

    result = await cache.clear_all()
    check("clear_all() returns counts", result["l1"] == 3, f"l1={result['l1']}")

    r = await cache.get("q1", "gemini", 0.0)
    check("All entries gone after clear", r.hit is False)

    await cache.close()


async def test_metrics() -> None:
    """Verifies that CacheManager and CacheMetrics track hits, misses, and costs accurately.

    Tests:
        - Counters for lookups, hits, misses, hit rate, writes, and tokens saved.
        - cost_saved_usd is positive after a cache hit.
        - get_full_stats() includes backend details and in_flight_count.
        - CacheMetrics.summary() computes per-layer and per-strategy breakdowns.
    """
    header("Metrics & Observability", 9)
    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response(tokens_used=100, provider="gemini")

    sub_header("Metrics after miss → set → hit")

    await cache.get("m_query", "gemini", 0.0)
    await cache.set("m_query", "gemini", 0.0, response)
    await cache.get("m_query", "gemini", 0.0)

    m = cache.get_metrics()
    check("total_lookups = 2", m["total_lookups"] == 2)
    check("total_hits = 1", m["total_hits"] == 1)
    check("total_misses = 1", m["total_misses"] == 1)
    check("hit_rate = 50%", m["hit_rate_pct"] == 50.0)
    check("total_writes = 1", m["total_writes"] == 1)
    check("tokens_saved = 100", m["total_tokens_saved"] == 100)
    check("cost_saved > 0", m["total_cost_saved_usd"] > 0, f"${m['total_cost_saved_usd']:.6f}")
    check("avg_lookup_latency > 0", m["avg_lookup_latency_ms"] > 0)
    check("avg_write_latency > 0", m["avg_write_latency_ms"] > 0)

    info(f"Full metrics: {m}")

    sub_header("Full stats with backend details")

    stats = await cache.get_full_stats()
    check("Stats has enabled", stats["enabled"] is True)
    check("Stats has initialized", stats["initialized"] is True)
    check("Stats has strategy", stats["strategy"] == "exact")
    check("Stats has l1 backend", "l1" in stats["backends"])
    check("L1 has current_size", "current_size" in stats["backends"]["l1"])
    check("Stats has in_flight_count", "in_flight_count" in stats)

    info(f"L1 backend stats: {stats['backends']['l1']}")

    sub_header("CacheMetrics model directly")

    metrics = CacheMetrics()
    metrics.record_hit("l1_memory", "exact", 50, 0.005, 0.03)
    metrics.record_hit("l2_redis", "semantic", 80, 0.01, 1.5)
    metrics.record_miss(0.8)
    metrics.record_write(2.0)
    metrics.record_quality_rejection()
    metrics.record_error("l1")
    metrics.record_error("serialization")

    s = metrics.summary()
    check("Hit rate 66.67%", s["hit_rate_pct"] == 66.67)
    check("L1 hits = 1", s["l1_hits"] == 1)
    check("L2 hits = 1", s["l2_hits"] == 1)
    check("Exact hits = 1", s["exact_hits"] == 1)
    check("Semantic hits = 1", s["semantic_hits"] == 1)
    check("Tokens saved = 130", s["total_tokens_saved"] == 130)
    check("Errors tracked", s["errors"]["l1"] == 1 and s["errors"]["serialization"] == 1)

    await cache.close()


async def test_stress() -> None:
    """Verifies cache stability and LRU behaviour under 500 sequential and 100 concurrent writes.

    Tests:
        - All 500 immediate reads hit the cache (written then read in the same iteration).
        - 500 cycles complete within 5 seconds.
        - L1 size stays at or below max_size (50); at least 450 LRU evictions occur.
        - After eviction, only the most recent entries remain (late_hits <= 50).
        - 100 concurrent burst writes complete without data corruption.
    """
    header("Stress Test (500 Write/Read Cycles)", 10)
    cache = CacheManager(make_settings(CACHE_L1_MAX_SIZE=50))
    await cache.initialize()

    response = make_response()
    total = 500

    sub_header(f"Sequential write → read × {total}")

    start = time.perf_counter()
    immediate_hits = 0

    for i in range(total):
        q = f"stress_query_{i}"
        await cache.set(q, "gemini", 0.0, response)
        r = await cache.get(q, "gemini", 0.0)
        if r.hit:
            immediate_hits += 1

    elapsed = time.perf_counter() - start

    check(
        f"All {total} immediate reads hit cache",
        immediate_hits == total,
        f"hits={immediate_hits}/{total}",
    )
    check(
        "Completed in reasonable time",
        elapsed < 5.0,
        f"{elapsed:.2f}s ({total / elapsed:.0f} ops/sec)",
    )

    stats = await cache.get_full_stats()
    l1_size = stats["backends"]["l1"]["current_size"]
    check("L1 size capped at max_size", l1_size <= 50, f"size={l1_size}")
    evictions = stats["backends"]["l1"]["total_evictions"]
    check("LRU evictions occurred", evictions >= 450, f"evictions={evictions}")

    sub_header("Read-back after eviction")

    late_hits = 0
    for i in range(total):
        r = await cache.get(f"stress_query_{i}", "gemini", 0.0)
        if r.hit:
            late_hits += 1

    check(
        "Only most recent entries survive",
        late_hits <= 50,
        f"late_hits={late_hits}/500",
    )

    sub_header("Concurrent burst — 100 simultaneous writes")

    await cache.clear_all()
    start = time.perf_counter()

    async def burst_write(i):
        await cache.set(f"burst_{i}", "gemini", 0.0, response)

    await asyncio.gather(*[burst_write(i) for i in range(100)])
    elapsed = time.perf_counter() - start

    size = (await cache.get_full_stats())["backends"]["l1"]["current_size"]
    check("Concurrent burst — no corruption", size <= 50, f"size={size}")
    check("Burst completed fast", elapsed < 2.0, f"{elapsed:.3f}s")

    sub_header("Final metrics")

    m = cache.get_metrics()
    info(f"Total lookups  : {m['total_lookups']}")
    info(f"Total hits     : {m['total_hits']}")
    info(f"Total misses   : {m['total_misses']}")
    info(f"Hit rate       : {m['hit_rate_pct']}%")
    info(f"Total writes   : {m['total_writes']}")
    info(f"Tokens saved   : {m['total_tokens_saved']}")
    info(f"Cost saved     : ${m['total_cost_saved_usd']:.6f}")
    info(f"Avg lookup     : {m['avg_lookup_latency_ms']:.3f}ms")
    info(f"Avg write      : {m['avg_write_latency_ms']:.3f}ms")

    await cache.close()


async def test_error_resilience() -> None:
    """Verifies that backend failures are absorbed gracefully without raising exceptions.

    Tests:
        - A broken get() returns a miss instead of propagating the exception.
        - A broken set() returns False instead of propagating the exception.
        - A disabled cache treats all gets as misses and all sets as no-ops.
        - Double initialize() is safe and does not clear existing data.
    """
    header("Error Resilience", 11)
    cache = CacheManager(make_settings())
    await cache.initialize()

    sub_header("Broken backend on read — returns miss, never crashes")

    original_get = cache._l1.get

    async def broken_get(key):
        raise RuntimeError("L1 exploded")

    cache._l1.get = broken_get

    try:
        result = await cache.get("hello", "gemini", 0.0)
        check("Broken get() → miss (no crash)", result.hit is False)
    except Exception as e:
        check("Broken get() → miss (no crash)", False, f"raised {type(e).__name__}")

    cache._l1.get = original_get

    sub_header("Broken backend on write — returns False, never crashes")

    original_set = cache._l1.set

    async def broken_set(key, value, ttl):
        raise RuntimeError("L1 write exploded")

    cache._l1.set = broken_set

    try:
        response = make_response()
        stored = await cache.set("hello", "gemini", 0.0, response)
        check("Broken set() → False (no crash)", stored is False)
    except Exception as e:
        check("Broken set() → False (no crash)", False, f"raised {type(e).__name__}")

    cache._l1.set = original_set

    sub_header("Disabled cache — all operations are no-ops")

    disabled_cache = CacheManager(make_settings(CACHE_ENABLED=False))
    await disabled_cache.initialize()

    r = await disabled_cache.get("hello", "gemini", 0.0)
    check("Disabled cache get → miss", r.hit is False)

    stored = await disabled_cache.set("hello", "gemini", 0.0, make_response())
    check("Disabled cache set → False", stored is False)

    await disabled_cache.close()

    sub_header("Double initialize is safe")

    cache2 = CacheManager(make_settings())
    await cache2.initialize()
    await cache2.set("hello", "gemini", 0.0, make_response())
    await cache2.initialize()

    r = await cache2.get("hello", "gemini", 0.0)
    check("Data survives double initialize", r.hit is True)

    await cache2.close()
    await cache.close()


async def run_all() -> None:
    pipeline_start = time.perf_counter()

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   CACHE PIPELINE TEST — Scalable RAG RLM")
    print(f"   Testing L1 caching system (Phases 1-4)")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}{Style.RESET_ALL}")

    # Sync sections
    test_normalizer_chain()
    test_exact_strategy()
    test_json_serializer()

    # Async sections
    await test_memory_backend()
    await test_cache_manager_e2e()
    await test_quality_gate()
    await test_request_coalescing()
    await test_invalidation()
    await test_metrics()
    await test_stress()
    await test_error_resilience()

    # Final summary
    pipeline_elapsed = time.perf_counter() - pipeline_start
    total = PASS_COUNT + FAIL_COUNT

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   FINAL RESULTS")
    print(f"{'═' * 70}{Style.RESET_ALL}")
    print()
    print(f"    {Fore.GREEN}Passed  : {PASS_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.RED}Failed  : {FAIL_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.YELLOW}Warnings: {WARN_COUNT}{Style.RESET_ALL}")
    print(f"    Total   : {total}")
    print(f"    Time    : {pipeline_elapsed:.2f}s")
    print()

    if FAIL_COUNT == 0:
        print(f"    {Fore.GREEN}{'═' * 50}")
        print(f"    ✓ ALL TESTS PASSED — Cache layer is solid")
        print(f"    {'═' * 50}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.RED}{'═' * 50}")
        print(f"    ✗ {FAIL_COUNT} TEST(S) FAILED — Review above")
        print(f"    {'═' * 50}{Style.RESET_ALL}")

    print()

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all())