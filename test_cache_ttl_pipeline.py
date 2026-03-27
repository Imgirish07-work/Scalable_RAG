"""
TTL Classifier + Quality Gate Pipeline Test — Phase 7 validation.

Run from project root:
    python test_cache_ttl_pipeline.py

No external dependencies — pure CPU tests, no Redis, no Qdrant, no model.

Sections:
    1. TTL classifier — query type detection
    2. TTL classifier — TTL assignment per type
    3. TTL classifier — edge cases
    4. TTL classifier — custom overrides
    5. Quality gate — rejection cases
    6. Quality gate — acceptance cases
    7. CacheManager integration — TTL varies by query type
    8. CacheManager integration — quality gate with reason
    9. Full stats include TTL and quality gate info
"""

import asyncio
import time
import sys
from datetime import datetime
from unittest.mock import MagicMock

from colorama import Fore, Style, init as colorama_init

from llm.models.llm_response import LLMResponse
from cache.quality.ttl_classifier import TTLClassifier, QueryType, TTL_MAP
from cache.quality.quality_gate import QualityGate
from cache.cache_manager import CacheManager

colorama_init(autoreset=True)

PASS_COUNT = 0
FAIL_COUNT = 0


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


def info(msg: str) -> None:
    print(f"    {Fore.WHITE}ℹ {msg}{Style.RESET_ALL}")


def make_response(**overrides) -> LLMResponse:
    defaults = dict(
        text="A thorough answer about the topic.",
        model="gemini-2.5-flash",
        provider="gemini",
        prompt_tokens=25,
        completion_tokens=40,
        tokens_used=65,
        latency_ms=450.0,
    )
    defaults.update(overrides)
    # Use model_construct to bypass Pydantic validation so tests can
    # deliberately create invalid responses (e.g. empty text, low tokens)
    # to exercise quality gate rejection logic.
    return LLMResponse.model_construct(**defaults)


def make_settings(**overrides) -> MagicMock:
    defaults = {
        "CACHE_ENABLED": True,
        "CACHE_L1_MAX_SIZE": 100,
        "CACHE_TTL_SECONDS": 3600,
        "CACHE_STRATEGY": "exact",
        "CACHE_SEMANTIC_THRESHOLD": 0.98,
        "CACHE_SEMANTIC_THRESHOLD_HIGH": 0.93,
        "CACHE_SEMANTIC_THRESHOLD_PARTIAL": 0.88,
        "CACHE_SEMANTIC_COLLECTION": "cache_semantic",
        "CACHE_CIRCUIT_BREAKER_THRESHOLD": 5,
        "CACHE_CIRCUIT_BREAKER_RESET_SECONDS": 60.0,
        "CACHE_MIN_RESPONSE_TOKENS": 20,
        "CACHE_MIN_RESPONSE_LATENCY_MS": 100.0,
        "COST_PER_TOKEN_OPENAI": 0.000002,
        "COST_PER_TOKEN_GEMINI": 0.0000001,
        "REDIS_ENV": "disabled",
        "REDIS_URL": "",
        "REDIS_CLOUD_URL": "",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "TEMPERATURE": 0.0,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ══════════════════════════════════════════════════
# SECTION 1 — Query type detection
# ══════════════════════════════════════════════════


def test_query_type_detection() -> None:
    header("TTL Classifier — Query Type Detection", 1)
    classifier = TTLClassifier()

    sub_header("Factual queries")

    factual_queries = [
        ("What is the latest Python version?", "latest"),
        ("Who is the current CEO of Google?", "who is"),
        ("What is the price of Bitcoin today?", "price of + today"),
        ("When was the last election?", "when was"),
        ("How much does GPT-4 cost?", "how much"),
        ("What are the current standings?", "current + standings"),
    ]
    for query, hint in factual_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → factual", result == QueryType.FACTUAL, hint)

    sub_header("Conceptual queries")

    conceptual_queries = [
        ("Explain how transformers work", "explain how"),
        ("What is retrieval augmented generation?", "what is"),
        ("Difference between RAG and fine-tuning", "difference between"),
        ("Why does gradient descent converge?", "why does"),
        ("Tell me about machine learning", "tell me about"),
        ("Pros and cons of microservices", "pros and cons"),
    ]
    for query, hint in conceptual_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → conceptual", result == QueryType.CONCEPTUAL, hint)

    sub_header("Code queries")

    code_queries = [
        ("Write a function to sort a list", "write a function"),
        ("Python code for binary search", "python + code for"),
        ("Implement a linked list in Java", "implement"),
        ("Fix this bug in my code", "fix this"),
        ("Write a SQL query for user counts", "sql query"),
        ("Regex for email validation", "regex for"),
    ]
    for query, hint in code_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → code", result == QueryType.CODE, hint)

    sub_header("Summarization queries")

    summarization_queries = [
        ("Summarize this document", "summarize"),
        ("Give me the key points", "key points"),
        ("TLDR of this article", "tldr"),
        ("Condense this into a paragraph", "condense"),
        ("What are the main takeaways?", "takeaways"),
    ]
    for query, hint in summarization_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → summarization", result == QueryType.SUMMARIZATION, hint)

    sub_header("Translation queries")

    translation_queries = [
        ("Translate this to Spanish", "translate"),
        ("How do you say hello in French?", "how do you say"),
        ("Convert this text to Japanese", "in japanese"),
    ]
    for query, hint in translation_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → translation", result == QueryType.TRANSLATION, hint)

    sub_header("Creative queries")

    creative_queries = [
        ("Write a poem about the ocean", "write a poem"),
        ("Brainstorm names for my startup", "brainstorm"),
        ("Generate ideas for a blog post", "generate ideas"),
        ("Write marketing copy for shoes", "marketing copy"),
    ]
    for query, hint in creative_queries:
        result = classifier.classify(query)
        check(f"'{query[:45]}' → creative", result == QueryType.CREATIVE, hint)


# ══════════════════════════════════════════════════
# SECTION 2 — TTL assignment per type
# ══════════════════════════════════════════════════


def test_ttl_assignment() -> None:
    header("TTL Classifier — TTL Assignment", 2)
    classifier = TTLClassifier(default_ttl=3600)

    sub_header("TTL values by type")

    test_cases = [
        ("What is the latest news?", QueryType.FACTUAL, 3_600, "1 hour"),
        ("Explain how TCP works", QueryType.CONCEPTUAL, 86_400, "24 hours"),
        ("Write a Python sort function", QueryType.CODE, 43_200, "12 hours"),
        ("Summarize this document", QueryType.SUMMARIZATION, 604_800, "7 days"),
        ("Translate to Spanish", QueryType.TRANSLATION, 604_800, "7 days"),
        ("Write a poem about rain", QueryType.CREATIVE, 259_200, "3 days"),
        ("Hello there", QueryType.DEFAULT, 3_600, "1 hour default"),
    ]

    for query, expected_type, expected_ttl, ttl_label in test_cases:
        ttl, detected_type = classifier.get_ttl_with_type(query)
        check(
            f"'{query[:35]}' → {ttl_label}",
            ttl == expected_ttl and detected_type == expected_type,
            f"type={detected_type.value}, ttl={ttl}s",
        )

    sub_header("get_ttl() convenience method")

    ttl = classifier.get_ttl("Explain how RAG works")
    check("get_ttl() returns int", isinstance(ttl, int) and ttl == 86_400, f"ttl={ttl}")


# ══════════════════════════════════════════════════
# SECTION 3 — Edge cases
# ══════════════════════════════════════════════════


def test_ttl_edge_cases() -> None:
    header("TTL Classifier — Edge Cases", 3)
    classifier = TTLClassifier(default_ttl=3600)

    sub_header("Empty and whitespace queries")

    check("Empty string → default", classifier.classify("") == QueryType.DEFAULT)
    check("Whitespace only → default", classifier.classify("   \t\n  ") == QueryType.DEFAULT)
    check("None-like empty → default TTL", classifier.get_ttl("") == 3600)

    sub_header("Mixed-type queries (first match wins)")

    q = "Explain the latest Python features"
    result = classifier.classify(q)
    info(f"'{q}' → {result.value} (has both 'explain' and 'latest')")
    check(
        "Mixed query classified",
        result in (QueryType.FACTUAL, QueryType.CONCEPTUAL),
        f"type={result.value}",
    )

    sub_header("Case insensitivity")

    check("UPPERCASE works", classifier.classify("EXPLAIN HOW RAG WORKS") == QueryType.CONCEPTUAL)
    check("MiXeD case works", classifier.classify("wRiTe A fUnCtIoN") == QueryType.CODE)

    sub_header("Pattern priority — code before conceptual")

    q2 = "Explain how to implement a function in Python"
    result2 = classifier.classify(q2)
    check(
        "Code keywords take priority",
        result2 == QueryType.CODE,
        f"type={result2.value} ('implement' + 'python' → code)",
    )

    sub_header("Unrecognized queries")

    unrecognized = [
        "Hello",
        "Thanks",
        "OK",
        "42",
        "asdfghjkl",
    ]
    for q in unrecognized:
        result = classifier.classify(q)
        check(f"'{q}' → default", result == QueryType.DEFAULT)


# ══════════════════════════════════════════════════
# SECTION 4 — Custom overrides
# ══════════════════════════════════════════════════


def test_ttl_custom_overrides() -> None:
    header("TTL Classifier — Custom Overrides", 4)

    sub_header("Override specific types")

    custom = TTLClassifier(
        default_ttl=1800,
        ttl_overrides={
            QueryType.FACTUAL: 600,
            QueryType.CONCEPTUAL: 172_800,
        },
    )

    ttl_factual = custom.get_ttl("What is the latest news?")
    check("Factual overridden to 600s", ttl_factual == 600, f"ttl={ttl_factual}")

    ttl_conceptual = custom.get_ttl("Explain how RAG works")
    check("Conceptual overridden to 172800s", ttl_conceptual == 172_800, f"ttl={ttl_conceptual}")

    ttl_code = custom.get_ttl("Write a Python function")
    check("Code NOT overridden (stays default map)", ttl_code == 43_200, f"ttl={ttl_code}")

    ttl_default = custom.get_ttl("Hello")
    check("Default overridden to 1800s", ttl_default == 1800, f"ttl={ttl_default}")

    sub_header("TTL map readable")

    ttl_map = custom.ttl_map
    check("ttl_map is a dict", isinstance(ttl_map, dict))
    check("factual in map", ttl_map.get("factual") == 600)
    info(f"Custom TTL map: {ttl_map}")


# ══════════════════════════════════════════════════
# SECTION 5 — Quality gate rejection
# ══════════════════════════════════════════════════


def test_quality_gate_rejection() -> None:
    header("Quality Gate — Rejection Cases", 5)
    gate = QualityGate(min_tokens=20, min_latency_ms=100.0)

    sub_header("Empty text")

    passed, reason = gate.check(make_response(text=""))
    check("Empty text → rejected", passed is False, reason)

    passed, reason = gate.check(make_response(text="   \n\t  "))
    check("Whitespace-only → rejected", passed is False, reason)

    sub_header("Too few tokens")

    passed, reason = gate.check(make_response(completion_tokens=5, tokens_used=10))
    check("5 tokens (< 20) → rejected", passed is False, reason)

    passed, reason = gate.check(make_response(completion_tokens=19, tokens_used=44))
    check("19 tokens (< 20) → rejected", passed is False, reason)

    sub_header("Suspiciously fast")

    passed, reason = gate.check(make_response(latency_ms=50.0))
    check("50ms (< 100ms) → rejected", passed is False, reason)

    passed, reason = gate.check(make_response(latency_ms=1.0))
    check("1ms (< 100ms) → rejected", passed is False, reason)

    sub_header("Reason messages are descriptive")

    _, reason = gate.check(make_response(text=""))
    check("Empty reason is clear", "empty" in reason)

    _, reason = gate.check(make_response(completion_tokens=5, tokens_used=10))
    check("Token reason includes counts", "5" in reason and "20" in reason)

    _, reason = gate.check(make_response(latency_ms=50.0))
    check("Latency reason includes values", "50" in reason and "100" in reason)


# ══════════════════════════════════════════════════
# SECTION 6 — Quality gate acceptance
# ══════════════════════════════════════════════════


def test_quality_gate_acceptance() -> None:
    header("Quality Gate — Acceptance Cases", 6)
    gate = QualityGate(min_tokens=20, min_latency_ms=100.0)

    sub_header("Good responses")

    passed, reason = gate.check(make_response())
    check("Normal response → accepted", passed is True and reason is None)

    passed, reason = gate.check(make_response(completion_tokens=20, latency_ms=100.0))
    check("Exactly at thresholds → accepted", passed is True)

    passed, reason = gate.check(make_response(completion_tokens=5000, latency_ms=10000.0))
    check("Very large response → accepted", passed is True)

    sub_header("passes() convenience method")

    check("passes() returns True for good", gate.passes(make_response()) is True)
    check("passes() returns False for bad", gate.passes(make_response(text="")) is False)

    sub_header("Thresholds readable")

    t = gate.thresholds
    check("Thresholds dict", t["min_tokens"] == 20 and t["min_latency_ms"] == 100.0)
    info(f"Thresholds: {t}")


# ══════════════════════════════════════════════════
# SECTION 7 — CacheManager integration: TTL varies
# ══════════════════════════════════════════════════


async def test_manager_ttl_varies() -> None:
    header("CacheManager — TTL Varies by Query Type", 7)
    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response()

    sub_header("Different queries get different TTLs")

    queries_and_expected = [
        ("What is the latest Python version?", "factual", 3_600),
        ("Explain how transformers work", "conceptual", 86_400),
        ("Write a Python sort function", "code", 43_200),
        ("Summarize this document for me", "summarization", 604_800),
        ("Translate this to French", "translation", 604_800),
        ("Write a poem about coding", "creative", 259_200),
        ("Hello there", "default", 3_600),
    ]

    for query, expected_type, expected_ttl in queries_and_expected:
        await cache.set(query, "gemini", 0.0, response)
        raw = await cache._l1.get(
            cache._exact_strategy.make_key(query, "gemini", 0.0)
        )
        if raw:
            from cache.serializers.json_serializer import JSONSerializer
            entry = JSONSerializer().deserialize(raw)
            check(
                f"'{query[:40]}' → {expected_type} ({expected_ttl}s)",
                entry.ttl_seconds == expected_ttl,
                f"actual_ttl={entry.ttl_seconds}s",
            )
        else:
            check(f"'{query[:40]}' cached", False, "not found in L1")

    sub_header("Manual TTL override still works")

    await cache.set("test override", "gemini", 0.0, response, ttl_seconds=999)
    raw = await cache._l1.get(
        cache._exact_strategy.make_key("test override", "gemini", 0.0)
    )
    if raw:
        from cache.serializers.json_serializer import JSONSerializer
        entry = JSONSerializer().deserialize(raw)
        check("Manual TTL=999 respected", entry.ttl_seconds == 999, f"ttl={entry.ttl_seconds}")

    await cache.close()


# ══════════════════════════════════════════════════
# SECTION 8 — CacheManager integration: quality gate
# ══════════════════════════════════════════════════


async def test_manager_quality_gate() -> None:
    header("CacheManager — Quality Gate with Reason", 8)
    cache = CacheManager(make_settings())
    await cache.initialize()

    sub_header("Rejections still work through manager")

    stored = await cache.set("q1", "gemini", 0.0, make_response(text=""))
    check("Empty text → rejected via manager", stored is False)

    stored = await cache.set("q2", "gemini", 0.0, make_response(completion_tokens=5, tokens_used=10))
    check("Few tokens → rejected via manager", stored is False)

    stored = await cache.set("q3", "gemini", 0.0, make_response(latency_ms=50.0))
    check("Fast response → rejected via manager", stored is False)

    sub_header("Good responses pass through manager")

    stored = await cache.set("q4", "gemini", 0.0, make_response())
    check("Good response → accepted via manager", stored is True)

    sub_header("Rejection counter")

    metrics = cache.get_metrics()
    check(
        "Quality rejections = 3",
        metrics["quality_gate_rejections"] == 3,
        f"rejections={metrics['quality_gate_rejections']}",
    )

    await cache.close()


# ══════════════════════════════════════════════════
# SECTION 9 — Full stats include TTL + quality info
# ══════════════════════════════════════════════════


async def test_full_stats() -> None:
    header("Full Stats — TTL + Quality Gate Info", 9)
    cache = CacheManager(make_settings())
    await cache.initialize()

    stats = await cache.get_full_stats()

    sub_header("TTL classifier in stats")

    check("ttl_classifier present", "ttl_classifier" in stats)
    if "ttl_classifier" in stats:
        ttl_map = stats["ttl_classifier"]
        check("Has factual TTL", "factual" in ttl_map, f"ttl={ttl_map.get('factual')}")
        check("Has conceptual TTL", "conceptual" in ttl_map, f"ttl={ttl_map.get('conceptual')}")
        check("Has code TTL", "code" in ttl_map, f"ttl={ttl_map.get('code')}")
        check("Has summarization TTL", "summarization" in ttl_map, f"ttl={ttl_map.get('summarization')}")
        info(f"TTL map: {ttl_map}")

    sub_header("Quality gate in stats")

    check("quality_gate present", "quality_gate" in stats)
    if "quality_gate" in stats:
        qg = stats["quality_gate"]
        check("Has min_tokens", "min_tokens" in qg, f"min_tokens={qg.get('min_tokens')}")
        check("Has min_latency_ms", "min_latency_ms" in qg, f"min_latency={qg.get('min_latency_ms')}")
        info(f"Quality gate: {qg}")

    await cache.close()


# ══════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════


async def run_all() -> None:
    pipeline_start = time.perf_counter()

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   TTL CLASSIFIER + QUALITY GATE PIPELINE TEST — Phase 7")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}{Style.RESET_ALL}")

    # Sync tests
    test_query_type_detection()
    test_ttl_assignment()
    test_ttl_edge_cases()
    test_ttl_custom_overrides()
    test_quality_gate_rejection()
    test_quality_gate_acceptance()

    # Async tests
    await test_manager_ttl_varies()
    await test_manager_quality_gate()
    await test_full_stats()

    elapsed = time.perf_counter() - pipeline_start

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   FINAL RESULTS — Phase 7")
    print(f"{'═' * 70}{Style.RESET_ALL}")
    print()
    print(f"    {Fore.GREEN}Passed  : {PASS_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.RED}Failed  : {FAIL_COUNT}{Style.RESET_ALL}")
    print(f"    Time    : {elapsed:.2f}s")
    print()

    if FAIL_COUNT == 0:
        print(f"    {Fore.GREEN}{'═' * 50}")
        print(f"    ✓ ALL TESTS PASSED — TTL + Quality gate verified")
        print(f"    {'═' * 50}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.RED}{'═' * 50}")
        print(f"    ✗ {FAIL_COUNT} TEST(S) FAILED — Review above")
        print(f"    {'═' * 50}{Style.RESET_ALL}")

    print()
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all())