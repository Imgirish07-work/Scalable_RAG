"""
Integration test for LLMRateLimiter using live Gemini API calls.

Test scope:
    Integration test (no pytest) demonstrating that the token-bucket rate
    limiter throttles requests to the configured RPM. The bucket is pre-drained
    to 1 token so throttling is observable from request 2 onward.

Flow:
    Phase 2 only (Phase 1 baseline is commented out to conserve quota):
    req 1 fires immediately (1 pre-loaded token); req 2 waits ~20s for refill.
    Total: 2 requests, safely within the free-tier limit of 5/min.

Dependencies:
    GEMINI_API_KEY in .env; network access to Gemini API.
    TEST_RPM=3 (1 token per 20s) — must be below the API's free-tier limit.
"""

import asyncio
import time

from llm.llm_factory import LLMFactory
from llm.rate_limiter import LLMRateLimiter, RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# TEST_RPM must stay below the API's actual limit (free tier: 5/min) so the
# limiter fires before the API returns a 429.
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
    """Send a single request without the rate limiter to establish a baseline latency."""
    _section("Phase 1: WITHOUT rate limiter (raw Gemini provider)")
    print("  Sending 1 request — baseline LLM latency\n")

    t0 = time.perf_counter()
    response = await llm.generate(QUERY, max_tokens=50)
    elapsed = time.perf_counter() - t0

    print(f"  [req 1/1] elapsed={elapsed:.2f}s | answer='{response.text.strip()}'")
    print(f"\n  Baseline LLM latency: {elapsed:.2f}s (no throttle)")


async def run_with_limiter(llm) -> None:
    """Send NUM_REQUESTS sequential requests through the token-bucket rate limiter.

    The bucket is pre-drained to 1 token so throttling is visible from request 2:
        req 1 fires immediately (1 token available).
        req 2 waits ~20s (bucket empty, refills at 1 token/20s).
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

    # Pre-drain to 1 token so throttling starts at request 2.
    # In production the bucket starts full (burst buffer). This is test-only.
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

        # Total elapsed minus actual LLM call time gives the throttle wait
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

    # await run_without_limiter(llm)  # omitted to conserve API quota
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
