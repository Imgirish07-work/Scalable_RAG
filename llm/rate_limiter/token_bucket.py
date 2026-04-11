"""
Async-safe token bucket algorithm for rate limiting.

Design:
    The token bucket works like a physical bucket with a slow drip:
        REFILL  — tokens accumulate continuously at refill_rate tokens/second,
                  up to a maximum capacity. Excess tokens overflow and are lost.
        CONSUME — each request takes one token. If the bucket is empty, the
                  caller waits precisely until one token has accumulated.

    Advantages over a fixed sleep(60/rpm) approach:
        - Accounts for time already elapsed in the previous request cycle.
        - Zero wasted waiting — sleeps only as long as strictly necessary.
        - Burst absorption — capacity > 1 allows short bursts above sustained rate.

    asyncio.Lock() ensures the check-and-consume operation is atomic across
    concurrent coroutines. Without the lock, two coroutines could both observe
    tokens >= 1, both consume the same token, and one request would bypass the limit.
    The wait sleep happens OUTSIDE the lock so other coroutines can proceed
    concurrently while a waiting coroutine sleeps.

Chain of Responsibility:
    LLMRateLimiter._throttle() calls acquire() on two TokenBucket instances
    (one for RPM, one for RPD) before every LLM API call.

Dependencies:
    asyncio, time.monotonic, utils.logger.
"""

import asyncio
from time import time, monotonic

from utils.logger import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Async-safe token bucket for a single rate window (RPM or RPD).

    Instantiate two TokenBucket objects in LLMRateLimiter to enforce
    both per-minute and per-day limits independently.

    Attributes:
        _capacity: Maximum tokens the bucket can hold.
        _refill_rate: Tokens added per second.
        _tokens: Current token count (float to accumulate fractional tokens).
        _last_refill: Monotonic timestamp of the last refill calculation.
        _lock: asyncio.Lock serializing check-and-consume operations.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """Initialize a full bucket ready to serve requests immediately.

        Args:
            capacity: Max tokens the bucket holds. Requests may burst up to
                this many before throttling begins.
            refill_rate: Tokens added per second. Set to rpm/60 for a
                per-minute window, or rpd/86400 for a per-day window.
        """
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity  # Start full so the first requests are not delayed
        self._last_refill = monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since the last refill.

        Called inside the lock before every acquire(). Uses the monotonic
        clock so the calculation is unaffected by system clock adjustments.

        Formula:
            new_tokens = elapsed_seconds * refill_rate
            tokens = min(tokens + new_tokens, capacity)
        """
        now = monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._tokens + new_tokens, self._capacity)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until one token is available, then consume it.

        Callers await this before every LLM request. If a token is available
        immediately, the call returns without sleeping. If not, it calculates
        the exact wait duration and yields to the event loop for that period.

        The sleep happens outside the lock so other coroutines can check the
        bucket concurrently while this one waits, rather than queuing behind
        the lock.
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1:
                # Token available — consume immediately without sleeping
                self._tokens -= 1
                return

            # Calculate the exact time needed for one token to accumulate
            tokens_needed = 1 - self._tokens
            wait_seconds = tokens_needed / self._refill_rate

        # Release the lock before sleeping — other coroutines must not block
        # waiting for the lock while this one is sleeping for rate recovery
        logger.debug("Rate limit: waiting %.2fs for token", wait_seconds)
        await asyncio.sleep(wait_seconds)

        # Reacquire the lock to consume the token we waited for
        async with self._lock:
            self._refill()
            self._tokens = max(self._tokens - 1, 0)

    @property
    def available_tokens(self) -> float:
        """Current token count snapshot for monitoring and logging.

        Not lock-protected — for observability only. Do not use for flow control.

        Returns:
            Current token count rounded to 2 decimal places.
        """
        return round(self._tokens, 2)
