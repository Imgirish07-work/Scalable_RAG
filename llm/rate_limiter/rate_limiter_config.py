"""
Configuration model for the LLM rate limiter.

Design:
    Immutable Pydantic BaseModel. Holds all tuneable parameters for one
    rate-limiter instance. Computed properties (refill_rate, bucket_capacity)
    derive token bucket parameters from the human-readable rpm/burst values.

Chain of Responsibility:
    get_rate_limit_config() in model_limits.py constructs this config →
    passed to LLMRateLimiter.__init__() → used to initialize TokenBucket
    instances and the asyncio.Semaphore.

Dependencies:
    pydantic (BaseModel, Field).
"""

from pydantic import BaseModel, Field


class RateLimiterConfig(BaseModel):
    """Configuration parameters for a single LLMRateLimiter instance.

    Attributes:
        rpm: Max requests allowed per minute. The RPM token bucket refills
            at rpm/60 tokens per second.
        rpd: Max requests allowed per day. A second token bucket drains once
            per request and refills over an 86400-second window.
        max_concurrent: asyncio.Semaphore size — limits the number of in-flight
            requests regardless of rate quota. Prevents thundering herd bursts.
        burst_multiplier: Bucket capacity = rpm * burst_multiplier.
            1.0 = strict (no burst above sustained rate).
            1.5 = allows 50% burst above the sustained rate for short spikes.
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
        """Tokens added to the RPM bucket per second.

        Returns:
            rpm / 60. Example: rpm=60 → 1.0 token/sec; rpm=15 → 0.25 token/sec.
        """
        return self.rpm / 60.0

    @property
    def bucket_capacity(self) -> float:
        """Max tokens the RPM bucket can hold, controlling burst size.

        Returns:
            rpm * burst_multiplier.
            Example: rpm=60, burst_multiplier=1.5 → capacity=90,
            allowing a short burst of up to 90 req/min before throttling.
        """
        return self.rpm * self.burst_multiplier
