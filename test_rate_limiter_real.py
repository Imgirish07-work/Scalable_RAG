"""
Real rate limiter test — live Gemini API calls.

What this test proves:
    1. WITHOUT rate limiter  → requests fire at full LLM speed
    2. WITH rate limiter     → token bucket throttles; each request beyond
                               the first waits for a token to refill

Key design choices:
    TEST_RPM = 3  -> 1 token per 20 seconds
                  -> safely BELOW free tier (5/min) so we never hit a real 429
                  -> throttling is clearly visible in wall-clock timestamps

    Bucket is PRE-DRAINED to 1 token so throttling kicks in from request 2.
    (Production buckets start full for burst tolerance — in this test we want
    to observe throttling immediately, not after burning through the burst.)

    Phase 1: 1 request  — no limiter baseline
    Phase 2: 3 requests — req 1 immediate, req 2 & 3 each wait ~20s
    Total: 4 requests in ~40s -> well within 5/min free tier limit

Run:
    python test_rate_limiter_real.py
"""

import asyncio
import time

from llm.llm_factory import LLMFactory
from llm.rate_limiter import LLMRateLimiter, RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Must be set BELOW the actual API limit (free tier = 5/min).
# If TEST_RPM > API limit, the real API rejects before the limiter kicks in.
TEST_RPM            = 3       # 1 token per 20s — safely below free tier 5/min
TEST_RPD            = 10000   # daily cap high — not the constraint here
TEST_MAX_CONCURRENT = 2       # max 2 requests in-flight simultaneously
TEST_BURST          = 1.0     # no burst — strict rate enforcement

NUM_REQUESTS = 2              # req 1 immediate, req 2 throttled (~20s wait)
QUERY = "What is the capital of France? Answer in one word."


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _section(title: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


async def run_without_limiter(llm) -> None:
    """Single request with NO rate limiter — establishes baseline LLM latency."""
    _section("Phase 1: WITHOUT rate limiter (raw Gemini provider)")
    print("  Sending 1 request — baseline LLM latency\n")

    t0 = time.perf_counter()
    response = await llm.generate(QUERY, max_tokens=50)
    elapsed = time.perf_counter() - t0

    print(f"  [req 1/1] elapsed={elapsed:.2f}s | answer='{response.text.strip()}'")
    print(f"\n  Baseline LLM latency: {elapsed:.2f}s (no throttle)")


async def run_with_limiter(llm) -> None:
    """3 sequential requests through the token bucket rate limiter.

    Bucket pre-drained to 1 token:
        req 1 -> immediate (1 token available)
        req 2 -> waits ~20s (bucket empty, refills at 1 token/20s)
        req 3 -> waits ~20s
    """
    _section(
        f"Phase 2: WITH rate limiter  "
        f"(RPM={TEST_RPM} -> 1 token per {60 // TEST_RPM}s)"
    )

    config = RateLimiterConfig(
        rpm=TEST_RPM,
        rpd=TEST_RPD,
        max_concurrent=TEST_MAX_CONCURRENT,
        burst_multiplier=TEST_BURST,
    )

    print(f"  Config   : rpm={config.rpm} | rpd={config.rpd} | "
          f"max_concurrent={config.max_concurrent} | "
          f"burst_multiplier={config.burst_multiplier}")
    print(f"  Refill   : {config.refill_rate:.4f} tokens/sec "
          f"(1 token every {1 / config.refill_rate:.0f}s)")
    print(f"  Capacity : {config.bucket_capacity:.0f} tokens")

    rate_limited_llm = LLMRateLimiter(provider=llm, config=config)

    # Pre-drain the bucket to 1 token so throttling starts from request 2.
    #
    # Why: In production the bucket starts FULL (capacity = 3 tokens here),
    # so requests 1, 2, 3 all fire immediately (that is the burst buffer).
    # For this test we pre-drain to 1 so throttling is immediately visible.
    # This is a test-only pattern — never do this in production code.
    rate_limited_llm._rpm_bucket._tokens = 1.0

    print(
        f"\n  Bucket pre-drained to 1 token — "
        f"req 1 immediate, req 2 & 3 each wait ~{60 // TEST_RPM}s\n"
    )

    test_start = time.perf_counter()

    for i in range(1, NUM_REQUESTS + 1):
        wall_before = time.perf_counter() - test_start
        print(
            f"  [req {i}/{NUM_REQUESTS}] @ T+{wall_before:.1f}s — "
            f"waiting for token ...",
            end="",
            flush=True,
        )

        t0 = time.perf_counter()
        response = await rate_limited_llm.generate(QUERY, max_tokens=50)
        call_time = time.perf_counter() - t0
        wall_after = time.perf_counter() - test_start

        # throttle_wait = total elapsed since req start minus actual LLM call time
        wait_time = (wall_after - wall_before) - call_time

        print(
            f"\r  [req {i}/{NUM_REQUESTS}] @ T+{wall_before:.1f}s "
            f"-> done T+{wall_after:.1f}s | "
            f"throttle_wait={wait_time:.1f}s | "
            f"llm_call={call_time:.2f}s | "
            f"rpm_tokens={rate_limited_llm._rpm_bucket.available_tokens:.2f} | "
            f"answer='{response.text.strip()[:30]}'"
        )

    total = time.perf_counter() - test_start
    expected_min = (NUM_REQUESTS - 1) * (60.0 / TEST_RPM)
    print(f"\n  Total wall-clock    : {total:.1f}s")
    print(f"  Expected minimum    : ~{expected_min:.0f}s "
          f"({NUM_REQUESTS - 1} throttled requests x {60 // TEST_RPM}s each)")
    print(
        f"  Rate limiter verdict: "
        f"{'WORKING' if total >= expected_min * 0.85 else 'NOT throttling — check config'}"
    )


async def run() -> None:
    _banner("RATE LIMITER REAL TEST — Live Gemini API (Free Tier Safe)")

    print(f"""
  Design:
    Total requests : {NUM_REQUESTS} (Phase 1 skipped, Phase 2 only)
    Free tier limit: 5 req/min
    TEST_RPM       : {TEST_RPM}  <- must be below API limit to be effective
    Expected time  : ~{(NUM_REQUESTS - 1) * (60 // TEST_RPM) + 5}s
    req 1 -> immediate (1 token pre-loaded)
    req 2 -> waits ~{60 // TEST_RPM}s (token bucket throttle)
    """)

    print("  Creating Gemini provider (real API, no mock)...")
    llm = LLMFactory.create("gemini")
    print(f"  Provider : {llm.provider_name}")
    print(f"  Model    : {llm.model_name}")

    # await run_without_limiter(llm)  # commented out — saves quota
    # await asyncio.sleep(2)
    await run_with_limiter(llm)

    _banner("TEST COMPLETE")
    print("""
  Reading the results:
    throttle_wait ~= 0s   on req 1  ->  immediate (token was available)
    throttle_wait ~= 20s  on req 2  ->  bucket empty, waited for refill
    throttle_wait ~= 20s  on req 3  ->  same

    rpm_tokens ~= 0.00 after each req -> bucket drained, limiter is active

  Log lines to check (set LOG_LEVEL=DEBUG for token bucket internals):
    INFO  | LLMRateLimiter initialized | provider=gemini | rpm=3 ...
    INFO  | Rate limit: waiting XX.XXs for token   <- throttle log
    DEBUG | Throttle passed | rpm_tokens=0.00 ...
    """)


if __name__ == "__main__":
    asyncio.run(run())
