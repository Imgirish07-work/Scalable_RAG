"""
Rate limiter configuration.

Each field maps to a real Gemini quota dimension:
    rpm              → requests per minute  (short window)
    rpd              → requests per day     (long window)
    max_concurrent   → how many requests can be IN-FLIGHT simultaneously
    burst_multiplier → how much above rpm the bucket can momentarily spike

Why two windows (rpm + rpd)?
    Gemini enforces BOTH. Staying under rpm doesn't protect you from
    hitting the daily cap mid-afternoon. The rate limiter tracks both.
"""

from pydantic import BaseModel, Field


class RateLimiterConfig(BaseModel):
    """Configuration for LLMRateLimiter.

    Attributes:
        rpm: Max requests allowed per minute. Token bucket refills
             at rpm/60 tokens per second.
        rpd: Max requests allowed per day. A second bucket drains
             once per request and refills at midnight (86400s window).
        max_concurrent: asyncio.Semaphore size — limits in-flight
             requests regardless of rate. Prevents thundering herd.
        burst_multiplier: Bucket capacity = rpm * burst_multiplier.
             1.0 = no burst allowed (strict). 1.5 = 50% burst above
             sustained rate for short spikes.
    """

    rpm: int = Field(gt=0, description="Requests per minute")
    rpd: int = Field(gt=0, description="Requests per day")
    max_concurrent: int = Field(
        default=5,
        gt=0,
        description="Max simultaneous in-flight LLM requests",
    )
    burst_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        description="Bucket capacity multiplier for burst tolerance",
    )

    @property
    def refill_rate(self) -> float:
        """Tokens added to the rpm bucket per second.

        Example: rpm=60 → 1.0 token/sec (one request per second sustained).
                 rpm=15 → 0.25 token/sec (one request per 4 seconds).
        """
        return self.rpm / 60.0

    @property
    def bucket_capacity(self) -> float:
        """Max tokens the rpm bucket can hold (controls burst size).

        Example: rpm=60, burst_multiplier=1.5 → capacity=90.
                 Allows a short burst of 90 req/min before throttling.
        """
        return self.rpm * self.burst_multiplier
