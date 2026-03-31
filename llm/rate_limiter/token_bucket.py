"""
Async-safe Token Bucket implementation.

How the Token Bucket algorithm works:
Imagine a bucket that holds tokens. Each token = permission for 1 request.

    REFILL: Tokens drip into the bucket continuously at a fixed rate
            (e.g. 1 token/second for rpm=60). The bucket has a max
            capacity — excess tokens overflow and are lost.

    CONSUME: Each request takes 1 token. If the bucket is empty,
             the request WAITS until enough tokens have accumulated.

This gives you:
    - Sustained rate enforcement (you can't exceed rpm over time)
    - Burst tolerance (capacity > 1 means short spikes are absorbed)
    - Zero wasted time (waits only as long as needed, no over-sleeping)

Why NOT just sleep(60/rpm)?
    Fixed sleep ignores time already elapsed. If a request took 2s to
    process, you'd sleep the full interval anyway. Token bucket accounts
    for elapsed time — no wasted waiting.

asyncio.Lock() usage:
    Multiple coroutines call acquire() concurrently. Without a lock,
    two coroutines could both see tokens > 0, both consume the same
    token, and one request sneaks through above the limit.
    The lock serializes the check-and-consume operation atomically.
"""

import asyncio
from time import time, monotonic

from utils.logger import get_logger

logger = get_logger(__name__)

class TokenBucket:
    """Async-safe token bucket for rate limiting.

    Tracks one rate window (either rpm or rpd). For both windows,
    instantiate two TokenBucket objects in LLMRateLimiter.

    Attributes:
        _capacity: Max tokens the bucket can hold.
        _refill_rate: Tokens added per second.
        _tokens: Current token count (float for fractional accumulation).
        _last_refill: Timestamp of last refill calculation.
        _lock: asyncio.Lock ensuring atomic check-and-consume.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """Initialize a full bucket.

        Args:
            capacity: Max tokens the bucket holds. Requests burst upto this many before throttling kicks in.

            refill_rate: Tokens added per second. Set to rpm/60 for a per-minute window.
        """
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity  # Start full
        self._last_refill = monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill.

        Called inside the lock before every acquire(). Uses monotonic 
        clock (never goes backwards, unaffected by system clock changes).

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
        """Wait until a token is available, then consume it.

        This is the key method — callers await this before every LLM
        request. It blocks (yields to the event loop) until the rate
        allows the request through.

        The wait time is calculated precisely:
            tokens_needed = 1 - current_tokens  (if tokens < 1)
            wait = tokens_needed / refill_rate
        """
        async with self._lock:
            self._refill() 

            # if tokens >= 1, consume immediately
            if self._tokens >= 1:
                self._tokens -= 1
                return

            # Calculate wait time for 1 token to accumulate
            tokens_needed = 1 - self._tokens
            wait_seconds = tokens_needed / self._refill_rate

        # Wait OUTSIDE the lock — other coroutines can check in parallel
        # while we wait. Without this, all other requests would block
        # waiting for the lock, not just waiting for tokens.
         
        logger.debug("Rate limit: waiting %.2fs for token", wait_seconds)
        await asyncio.sleep(wait_seconds)

        # Reacquire the lock and consume the token we waited for
        async with self._lock:
            self._refill()
            self._tokens = max(self._tokens - 1, 0)

    @property
    def available_tokens(self) -> float:
        """Current token count (snapshot, not lock-protected).

        For monitoring/logging only — do NOT use for flow control.
        """
        return round(self._tokens, 2)