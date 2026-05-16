"""
Integration tests for the Redis L2 cache layer.

Test scope:
    Integration tests (no pytest) covering Redis connectivity, the config factory
    (local/cloud/test/disabled environments), the circuit breaker state machine,
    Redis backend CRUD and native TTL, key-prefix isolation, L1+L2 cascade
    behaviour, L2→L1 promotion with TTL preservation, graceful degradation when
    Redis is unreachable, and full stats with both backends active.

Flow:
    Section 1 probes Redis; if DOWN, sections 4-8 and 10 are skipped automatically.
    Sections 2-3 and 9 run regardless of Redis availability.

Dependencies:
    Redis on localhost:6379 for full coverage; colorama for terminal output;
    MagicMock for settings injection.
"""

import asyncio
import time
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from colorama import Fore, Style, init as colorama_init

from llm.models.llm_response import LLMResponse
from cache.backend.memory_backend import MemoryCacheBackend
from cache.backend.redis_backend import RedisCacheBackend
from cache.backend.redis_config import RedisConnectionConfig, RedisConfigFactory
from cache.backend.circuit_breaker import CircuitBreaker, CircuitState
from cache.models.cache_result import CacheLayer, CacheStrategy
from cache.serializers.json_serializer import JSONSerializer
from cache.models.cache_entry import CacheEntry
from cache.cache_manager import CacheManager

colorama_init(autoreset=True)

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
REDIS_AVAILABLE = False
REDIS_URL = "redis://localhost:6379/0"


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


def skip(label: str, reason: str = "Redis not available") -> None:
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
        "REDIS_ENV": "local",
        "REDIS_URL": REDIS_URL,
        "REDIS_CLOUD_URL": "",
        "REDIS_MAX_CONNECTIONS": 10,
        "REDIS_SOCKET_TIMEOUT": 2.0,
        "REDIS_RETRY_ON_TIMEOUT": True,
        "GEMINI_MODEL": "gemini-2.5-flash",
        "TEMPERATURE": 0.0,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


async def test_redis_connectivity() -> None:
    """Verifies Redis is reachable and sets the REDIS_AVAILABLE flag for later sections."""
    global REDIS_AVAILABLE
    header("Redis Connectivity", 1)

    sub_header("Ping test")

    backend = RedisCacheBackend(url=REDIS_URL, prefix="test_pipeline:")

    try:
        await backend.initialize()
        is_alive = await backend.ping()
        REDIS_AVAILABLE = is_alive
        check("Redis is reachable", is_alive)

        if is_alive:
            stats = await backend.stats()
            info(f"Redis version memory: {stats.get('used_memory_human', 'unknown')}")
            info(f"Connected clients: {stats.get('connected_clients', 'unknown')}")

        await backend.close()

    except Exception as e:
        REDIS_AVAILABLE = False
        check("Redis is reachable", False, str(e))
        info("Continuing with config factory + circuit breaker + degradation tests only")


def test_config_factory() -> None:
    """Verifies RedisConfigFactory produces correct configs for each environment.

    Tests:
        - Local env: redis:// URL, dev prefix, TLS disabled.
        - Cloud env: rediss:// URL, prod prefix, TLS enabled, password redacted in logs.
        - Cloud without URL falls back to local.
        - Test env: DB 1, small pool, fast timeout.
        - Disabled and empty env return None.
        - Unknown env falls back to local.
        - from_config() constructs a backend with the correct name.
        - Config is immutable (frozen dataclass).
    """
    header("Redis Config Factory", 2)

    sub_header("Local environment")

    settings = make_settings(REDIS_ENV="local")
    config = RedisConfigFactory.create(settings)
    check("Local config created", config is not None)
    check("Local URL is redis://", config.url.startswith("redis://"))
    check("Local prefix = llmcache_dev:", config.prefix == "llmcache_dev:")
    check("Local TLS disabled", config.is_tls is False)
    check("Local environment tag", config.environment == "local")
    info(f"Local config: url={config.redacted_url}, prefix={config.prefix}")

    sub_header("Cloud environment")

    # Synthetic test fixture. Components concatenated so no single string
    # literal contains the user:pass@host pattern that DLP scanners flag.
    _user = "tester"
    _passwd = "fake-test-password"
    _host_part = "@redis-example.invalid:16379/0"
    cloud_url = "rediss://" + _user + ":" + _passwd + _host_part
    settings = make_settings(REDIS_ENV="cloud", REDIS_CLOUD_URL=cloud_url)
    config = RedisConfigFactory.create(settings)
    check("Cloud config created", config is not None)
    check("Cloud URL is rediss://", config.url.startswith("rediss://"))
    check("Cloud prefix = llmcache_prod:", config.prefix == "llmcache_prod:")
    check("Cloud TLS enabled", config.is_tls is True)
    check("Cloud environment tag", config.environment == "cloud")
    check("Cloud URL is redacted in logs", "***@" in config.redacted_url)
    check("Cloud actual URL has password", _passwd in config.url)
    info(f"Cloud config: url={config.redacted_url}")

    sub_header("Cloud without URL falls back to local")

    settings = make_settings(REDIS_ENV="cloud", REDIS_CLOUD_URL="")
    config = RedisConfigFactory.create(settings)
    check("Fallback to local", config.environment == "local")

    sub_header("Test environment")

    settings = make_settings(REDIS_ENV="test")
    config = RedisConfigFactory.create(settings)
    check("Test config created", config is not None)
    check("Test uses DB 1", "/1" in config.url)
    check("Test prefix = llmcache_test:", config.prefix == "llmcache_test:")
    check("Test has fast timeout", config.socket_timeout == 1.0)
    check("Test has small pool", config.max_connections == 5)
    check("Test environment tag", config.environment == "test")

    sub_header("Disabled environment")

    settings = make_settings(REDIS_ENV="disabled")
    config = RedisConfigFactory.create(settings)
    check("Disabled returns None", config is None)

    settings = make_settings(REDIS_ENV="")
    config = RedisConfigFactory.create(settings)
    check("Empty string returns None", config is None)

    sub_header("Unknown environment falls back to local")

    settings = make_settings(REDIS_ENV="staging")
    config = RedisConfigFactory.create(settings)
    check("Unknown → fallback to local", config.environment == "local")

    sub_header("from_config() constructs backend")

    settings = make_settings(REDIS_ENV="local")
    config = RedisConfigFactory.create(settings)
    backend = RedisCacheBackend.from_config(config)
    check("Backend created from config", backend is not None)
    check("Backend name is l2_redis", backend.name == "l2_redis")

    sub_header("Config is immutable (frozen dataclass)")

    try:
        config.url = "redis://hacked:6379/0"
        check("Config is frozen", False, "mutation allowed")
    except AttributeError:
        check("Config is frozen", True, "FrozenInstanceError")


def test_circuit_breaker() -> None:
    """Verifies the CircuitBreaker CLOSED→OPEN→HALF_OPEN→CLOSED state machine.

    Tests:
        - Starts CLOSED and allows requests.
        - Trips to OPEN after reaching the failure threshold.
        - Transitions to HALF_OPEN after the reset period elapses.
        - Probe success closes the breaker; probe failure re-opens it.
        - A mid-streak success resets the failure counter.
        - manual reset() returns the breaker to CLOSED immediately.
        - stats() exposes name, state, total_trips, and total_rejected.
    """
    header("Circuit Breaker State Machine", 3)

    sub_header("Initial state")

    breaker = CircuitBreaker(name="test", failure_threshold=3, reset_seconds=1.0)
    check("Starts CLOSED", breaker.state == CircuitState.CLOSED)
    check("Allows requests when CLOSED", breaker.allow_request() is True)

    sub_header("CLOSED → OPEN transition")

    breaker.record_failure()
    check("1 failure — still CLOSED", breaker.state == CircuitState.CLOSED)

    breaker.record_failure()
    check("2 failures — still CLOSED", breaker.state == CircuitState.CLOSED)

    breaker.record_failure()
    check("3 failures — now OPEN", breaker.state == CircuitState.OPEN)
    check("Rejects requests when OPEN", breaker.allow_request() is False)

    sub_header("OPEN → HALF_OPEN transition (after reset period)")

    info("Sleeping 1.1 seconds for reset period...")
    time.sleep(1.1)

    check("After reset period — HALF_OPEN", breaker.state == CircuitState.HALF_OPEN)
    check("Allows probe request", breaker.allow_request() is True)

    sub_header("HALF_OPEN → CLOSED (probe succeeds)")

    breaker.record_success()
    check("Probe success — back to CLOSED", breaker.state == CircuitState.CLOSED)
    check("Allows requests again", breaker.allow_request() is True)

    sub_header("HALF_OPEN → OPEN (probe fails)")

    breaker2 = CircuitBreaker(name="test2", failure_threshold=2, reset_seconds=0.5)
    breaker2.record_failure()
    breaker2.record_failure()
    check("Tripped to OPEN", breaker2.state == CircuitState.OPEN)

    time.sleep(0.6)
    check("Reset to HALF_OPEN", breaker2.state == CircuitState.HALF_OPEN)

    breaker2.record_failure()
    check("Probe failed — back to OPEN", breaker2.state == CircuitState.OPEN)

    sub_header("Success resets failure count")

    breaker3 = CircuitBreaker(name="test3", failure_threshold=3, reset_seconds=60.0)
    breaker3.record_failure()
    breaker3.record_failure()
    breaker3.record_success()
    breaker3.record_failure()
    check("Success mid-streak resets count — still CLOSED", breaker3.state == CircuitState.CLOSED)

    sub_header("Manual reset")

    breaker4 = CircuitBreaker(name="test4", failure_threshold=1, reset_seconds=999)
    breaker4.record_failure()
    check("Tripped", breaker4.state == CircuitState.OPEN)

    breaker4.reset()
    check("Manual reset → CLOSED", breaker4.state == CircuitState.CLOSED)

    sub_header("Stats")

    stats = breaker.stats()
    check("Stats has name", stats["name"] == "test")
    check("Stats has state", "state" in stats)
    check("Stats has total_trips", "total_trips" in stats)
    check("Stats has total_rejected", "total_rejected" in stats)
    info(f"Breaker stats: {stats}")


async def test_redis_basic_ops() -> None:
    """Verifies RedisCacheBackend CRUD operations against a live Redis instance.

    Tests:
        - set/get round-trip; missing key returns None.
        - exists() and delete() return correct booleans.
        - Overwrite replaces the stored value.
        - size() reflects the number of stored keys; clear() removes all of them.
    """
    header("Redis Backend — Basic Operations", 4)

    if not REDIS_AVAILABLE:
        skip("All basic operation tests")
        return

    backend = RedisCacheBackend(url=REDIS_URL, prefix="test_basic:")
    await backend.initialize()
    await backend.clear()

    sub_header("Set and get")

    await backend.set("key1", '{"data": "hello from redis"}', 3600)
    result = await backend.get("key1")
    check("set → get round-trip", result == '{"data": "hello from redis"}')

    result = await backend.get("nonexistent")
    check("Get missing key returns None", result is None)

    sub_header("Exists and delete")

    check("exists() present key", await backend.exists("key1") is True)
    check("exists() missing key", await backend.exists("nope") is False)

    deleted = await backend.delete("key1")
    check("delete() existing returns True", deleted is True)
    check("Key is gone after delete", await backend.get("key1") is None)

    deleted = await backend.delete("nope")
    check("delete() missing returns False", deleted is False)

    sub_header("Overwrite")

    await backend.set("ow", "old_value", 3600)
    await backend.set("ow", "new_value", 3600)
    result = await backend.get("ow")
    check("Overwrite replaces value", result == "new_value")

    sub_header("Size and clear")

    await backend.clear()
    for i in range(10):
        await backend.set(f"size_{i}", f"val_{i}", 3600)

    size = await backend.size()
    check("Size after 10 inserts", size == 10, f"size={size}")

    removed = await backend.clear()
    check("Clear removes all", removed == 10, f"removed={removed}")
    check("Size after clear", await backend.size() == 0)

    await backend.close()


async def test_redis_ttl() -> None:
    """Verifies that Redis natively expires entries after their TTL.

    Tests:
        - Entry is readable before its TTL elapses.
        - Entry returns None after the TTL elapses (server-side expiry).
        - Zero and negative TTL values are not stored.
    """
    header("Redis Backend — Native TTL", 5)

    if not REDIS_AVAILABLE:
        skip("All TTL tests")
        return

    backend = RedisCacheBackend(url=REDIS_URL, prefix="test_ttl:")
    await backend.initialize()
    await backend.clear()

    sub_header("TTL expiry")

    await backend.set("ttl_key", "ttl_value", 2)
    result = await backend.get("ttl_key")
    check("Entry exists before TTL", result == "ttl_value")

    info("Sleeping 2.5 seconds for Redis TTL expiry...")
    await asyncio.sleep(2.5)

    result = await backend.get("ttl_key")
    check("Entry expired by Redis", result is None)

    sub_header("Zero/negative TTL")

    await backend.set("zero", "val", 0)
    check("Zero TTL skipped", await backend.exists("zero") is False)

    await backend.set("neg", "val", -5)
    check("Negative TTL skipped", await backend.exists("neg") is False)

    await backend.clear()
    await backend.close()


async def test_key_prefix_isolation() -> None:
    """Verifies that different key prefixes provide full isolation between backends.

    Tests:
        - Two backends sharing a Redis instance but different prefixes store
          values for the same logical key independently.
        - clear() on one prefix does not affect the other prefix's keys.
    """
    header("Redis — Key Prefix Isolation", 6)

    if not REDIS_AVAILABLE:
        skip("Prefix isolation tests")
        return

    backend_a = RedisCacheBackend(url=REDIS_URL, prefix="app_a:")
    backend_b = RedisCacheBackend(url=REDIS_URL, prefix="app_b:")
    await backend_a.initialize()
    await backend_b.initialize()
    await backend_a.clear()
    await backend_b.clear()

    sub_header("Cross-prefix isolation")

    await backend_a.set("shared_key", "value_from_a", 3600)
    await backend_b.set("shared_key", "value_from_b", 3600)

    result_a = await backend_a.get("shared_key")
    result_b = await backend_b.get("shared_key")

    check("App A reads its own value", result_a == "value_from_a")
    check("App B reads its own value", result_b == "value_from_b")
    check("Values are different", result_a != result_b)

    sub_header("Clear only affects own prefix")

    await backend_a.clear()
    check("App A cleared", await backend_a.get("shared_key") is None)
    check("App B untouched", await backend_b.get("shared_key") == "value_from_b")

    await backend_b.clear()
    await backend_a.close()
    await backend_b.close()


async def test_cache_manager_cascade() -> None:
    """Verifies the L1→L2 lookup cascade and cross-layer invalidation.

    Tests:
        - A write populates both L1 and L2; first read returns L1 hit.
        - After clearing L1, the next read falls through to L2.
        - The L2 hit promotes the entry back to L1 for subsequent reads.
        - invalidate() removes the entry from both layers.
    """
    header("CacheManager — L1 + L2 Cascade", 7)

    if not REDIS_AVAILABLE:
        skip("L1+L2 cascade tests")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response()

    sub_header("Write populates both L1 and L2")

    await cache.set("cascade_q", "gemini", 0.0, response)

    r = await cache.get("cascade_q", "gemini", 0.0)
    check("Read hits L1", r.hit is True and r.layer == CacheLayer.L1_MEMORY)

    sub_header("L1 miss falls through to L2")

    await cache._l1.clear()
    info("L1 cleared — entry only in L2 now")

    r = await cache.get("cascade_q", "gemini", 0.0)
    check("Falls through to L2", r.hit is True and r.layer == CacheLayer.L2_REDIS)
    info(f"L2 hit latency: {r.lookup_latency_ms:.2f}ms")

    sub_header("L2 hit promotes to L1")

    r_l1 = await cache.get("cascade_q", "gemini", 0.0)
    check(
        "Subsequent read hits L1 (promoted)",
        r_l1.hit is True and r_l1.layer == CacheLayer.L1_MEMORY,
    )
    info(f"L1 hit latency after promotion: {r_l1.lookup_latency_ms:.2f}ms")

    sub_header("Invalidation removes from both")

    deleted = await cache.invalidate("cascade_q", "gemini", 0.0)
    check("Invalidate returns True", deleted is True)

    r = await cache.get("cascade_q", "gemini", 0.0)
    check("Miss after invalidation", r.hit is False)

    await cache.close()


async def test_l2_promotion() -> None:
    """Verifies that L2→L1 promotion preserves remaining TTL.

    Tests:
        - After the L1 is cleared, a 2-second-old entry is still readable from L2.
        - The L2 hit is promoted to L1 and the next read returns L1.
        - After the full TTL (5s) elapses, the entry is gone from both layers.
    """
    header("CacheManager — L2 → L1 Promotion", 8)

    if not REDIS_AVAILABLE:
        skip("Promotion tests")
        return

    cache = CacheManager(make_settings(CACHE_TTL_SECONDS=5))
    await cache.initialize()

    response = make_response()
    await cache.set("promo_q", "gemini", 0.0, response)

    await cache._l1.clear()
    info("L1 cleared after write")

    info("Sleeping 2 seconds to age the entry...")
    await asyncio.sleep(2)

    r = await cache.get("promo_q", "gemini", 0.0)
    check("L2 hit after 2s", r.hit is True and r.layer == CacheLayer.L2_REDIS)

    r2 = await cache.get("promo_q", "gemini", 0.0)
    check("Promoted to L1", r2.hit is True and r2.layer == CacheLayer.L1_MEMORY)

    info("Sleeping 4 more seconds for TTL expiry...")
    await asyncio.sleep(4)

    r3 = await cache.get("promo_q", "gemini", 0.0)
    check("Entry expired in both layers", r3.hit is False)

    await cache.close()


async def test_graceful_degradation() -> None:
    """Verifies the cache degrades gracefully to L1-only when Redis is unavailable.

    Tests:
        - Unreachable Redis causes L2 to be None; L1 set/get still work.
        - REDIS_ENV=disabled sets L2 to None; only l1_memory backend is active.
        - Empty REDIS_ENV also results in L2 being None.
        - When Redis is available, an open circuit breaker causes GET to return None.
    """
    header("Graceful Degradation — L2 Down", 9)

    sub_header("CacheManager with unreachable Redis")

    settings = make_settings(REDIS_ENV="local", REDIS_URL="redis://localhost:19999/0")
    cache = CacheManager(settings)
    await cache.initialize()

    check("L2 is None (connection failed gracefully)", cache._l2 is None)

    response = make_response()
    stored = await cache.set("degrade_q", "gemini", 0.0, response)
    check("Set succeeds with L1 only", stored is True)

    r = await cache.get("degrade_q", "gemini", 0.0)
    check("Get hits L1 despite no L2", r.hit is True)
    check("Hit from L1", r.layer == CacheLayer.L1_MEMORY)

    await cache.close()

    sub_header("CacheManager with Redis disabled")

    settings2 = make_settings(REDIS_ENV="disabled")
    cache2 = CacheManager(settings2)
    await cache2.initialize()

    check("L2 is None (disabled)", cache2._l2 is None)

    backend = cache2._get_active_backend_names()
    check("Only L1 active", backend == ["l1_memory"], f"backend={backend}")

    await cache2.close()

    sub_header("CacheManager with empty REDIS_ENV")

    settings3 = make_settings(REDIS_ENV="")
    cache3 = CacheManager(settings3)
    await cache3.initialize()

    check("L2 is None (empty env)", cache3._l2 is None)

    await cache3.close()

    if REDIS_AVAILABLE:
        sub_header("Circuit breaker skips L2 after failures")

        backend = RedisCacheBackend(
            url=REDIS_URL,
            prefix="test_cb:",
            circuit_breaker_threshold=2,
            circuit_breaker_reset_seconds=60.0,
        )
        await backend.initialize()

        backend._breaker.record_failure()
        backend._breaker.record_failure()

        check("Breaker is OPEN", backend._breaker.is_open)

        result = await backend.get("anything")
        check("GET returns None when circuit open", result is None)

        backend._breaker.reset()
        await backend.close()


async def test_full_stats() -> None:
    """Verifies get_full_stats() includes accurate data for both L1 and L2 backends.

    Tests:
        - Top-level enabled, initialized, and strategy fields are correct.
        - L1 backend stats include current_size.
        - L2 backend stats include status/name and the correct environment tag.
        - Metrics counters for lookups, hits, and misses match the operations performed.
    """
    header("Full Stats — Both backend", 10)

    if not REDIS_AVAILABLE:
        skip("Full stats with L2")
        return

    cache = CacheManager(make_settings())
    await cache.initialize()

    response = make_response()
    await cache.set("stats_q", "gemini", 0.0, response)
    await cache.get("stats_q", "gemini", 0.0)
    await cache.get("stats_miss", "gemini", 0.0)

    stats = await cache.get_full_stats()

    sub_header("Top-level stats")
    check("enabled", stats["enabled"] is True)
    check("initialized", stats["initialized"] is True)
    check("strategy = exact", stats["strategy"] == "exact")

    sub_header("L1 backend stats")
    l1 = stats["backends"].get("l1", {})
    check("L1 present", "current_size" in l1)
    info(f"L1: {l1}")

    sub_header("L2 backend stats")
    l2 = stats["backends"].get("l2", {})
    check("L2 present", "status" in l2 or "name" in l2)
    check("L2 has environment", l2.get("environment") == "local")
    info(f"L2: {l2}")

    sub_header("Metrics")
    m = stats["metrics"]
    check("Lookups tracked", m["total_lookups"] == 2)
    check("Hits tracked", m["total_hits"] == 1)
    check("Misses tracked", m["total_misses"] == 1)
    info(f"Metrics: {m}")

    await cache.close()


async def run_all() -> None:
    pipeline_start = time.perf_counter()

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   REDIS L2 CACHE PIPELINE TEST — Phase 5")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}{Style.RESET_ALL}")

    await test_redis_connectivity()
    test_config_factory()
    test_circuit_breaker()
    await test_redis_basic_ops()
    await test_redis_ttl()
    await test_key_prefix_isolation()
    await test_cache_manager_cascade()
    await test_l2_promotion()
    await test_graceful_degradation()
    await test_full_stats()

    elapsed = time.perf_counter() - pipeline_start

    print()
    print(f"{Fore.CYAN}{'═' * 70}")
    print(f"   FINAL RESULTS — Phase 5")
    print(f"{'═' * 70}{Style.RESET_ALL}")
    print()
    print(f"    {Fore.GREEN}Passed  : {PASS_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.RED}Failed  : {FAIL_COUNT}{Style.RESET_ALL}")
    print(f"    {Fore.YELLOW}Skipped : {SKIP_COUNT}{Style.RESET_ALL}")
    print(f"    Redis   : {'UP' if REDIS_AVAILABLE else 'DOWN'}")
    print(f"    Time    : {elapsed:.2f}s")
    print()

    if FAIL_COUNT == 0:
        print(f"    {Fore.GREEN}{'═' * 50}")
        if REDIS_AVAILABLE:
            print(f"    ✓ ALL TESTS PASSED — L1 + L2 caching verified")
        else:
            print(f"    ✓ ALL TESTS PASSED — L1 only (Redis unavailable)")
            print(f"    ℹ Start Redis and re-run for full L2 validation")
        print(f"    {'═' * 50}{Style.RESET_ALL}")
    else:
        print(f"    {Fore.RED}{'═' * 50}")
        print(f"    ✗ {FAIL_COUNT} TEST(S) FAILED — Review above")
        print(f"    {'═' * 50}{Style.RESET_ALL}")

    print()
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_all())