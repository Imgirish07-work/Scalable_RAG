"""
LLM Rate Limiter — transparent BaseLLM wrapper.

Design: Decorator Pattern
─────────────────────────
LLMRateLimiter IS-A BaseLLM and HAS-A BaseLLM (the real provider).

    pipeline.query()
        → LLMRateLimiter.generate()  ← throttles here
            → GeminiProvider.generate()  ← real API call

The pipeline never knows rate limiting is happening. It just calls
generate() / chat() on what it thinks is a regular LLM provider.

Three layers of protection (applied in order):
    1. asyncio.Semaphore — caps simultaneous in-flight requests.
       Prevents thundering herd: 100 concurrent requests hitting
       the API at the same millisecond.

    2. RPM TokenBucket — enforces requests-per-minute limit.
       Waits until the minute-window allows another request.

    3. RPD TokenBucket — enforces requests-per-day limit.
       Waits until the day-window allows another request.
       (Effectively blocks for the rest of the day if daily quota hit.)

Why Semaphore first, then buckets?
    Semaphore limits the number of requests competing for tokens.
    Without it, 100 coroutines would all pile up waiting for tokens,
    consuming memory and CPU. The semaphore keeps the queue bounded.
"""

import asyncio

from llm.contracts.base_llm import BaseLLM
from llm.models.llm_response import LLMResponse
from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from llm.rate_limiter.token_bucket import TokenBucket
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMRateLimiter(BaseLLM):
    """Transparent rate-limiting wrapper around any BaseLLM provider.

    Drop-in replacement — wraps an existing provider and adds:
        - Concurrency cap via asyncio.Semaphore
        - Per-minute rate limiting via RPM TokenBucket
        - Per-day rate limiting via RPD TokenBucket

    Usage:
        provider = GeminiProvider(...)
        config = RateLimiterConfig(rpm=60, rpd=1500, max_concurrent=5)
        llm = LLMRateLimiter(provider, config)
        # Use llm exactly like provider — same interface

    Attributes:
        _provider: The real LLM provider being wrapped.
        _config: Rate limiter configuration.
        _semaphore: asyncio.Semaphore for concurrency cap.
        _rpm_bucket: Token bucket for per-minute rate limit.
        _rpd_bucket: Token bucket for per-day rate limit.
    """

    def __init__(self, provider: BaseLLM, config: RateLimiterConfig) -> None:
        """Wrap a provider with rate limiting.

        Args:
            provider: Any BaseLLM implementation to wrap.
            config: Rate limits and concurrency settings.
        """
        self._provider = provider
        self._config = config

        # Layer 1: concurrency cap
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        # Layer 2: per-minute token bucket
        # capacity = rpm * burst_multiplier (allows short bursts)
        # refill_rate = rpm / 60 (tokens per second)
        self._rpm_bucket = TokenBucket(
            capacity=config.bucket_capacity,
            refill_rate=config.refill_rate,
        )

        # Layer 3: per-day token bucket
        # capacity = rpd (no burst — daily cap is hard)
        # refill_rate = rpd / 86400 (tokens per second over 24h)
        self._rpd_bucket = TokenBucket(
            capacity=float(config.rpd),
            refill_rate=config.rpd / 86400.0,
        )

        logger.info(
            "LLMRateLimiter initialized | provider=%s | rpm=%d | rpd=%d | "
            "max_concurrent=%d | burst=%.1fx",
            provider.provider_name,
            config.rpm,
            config.rpd,
            config.max_concurrent,
            config.burst_multiplier,
        )

    async def _throttle(self) -> None:
        """Apply all three rate-limiting layers before an LLM call.

        Layers applied in sequence:
            1. Acquire semaphore slot (concurrency cap)
            2. Acquire RPM token (minute-window rate)
            3. Acquire RPD token (day-window rate)

        Note: Semaphore is NOT used as a context manager here because
        we release it AFTER the actual LLM call (in generate/chat),
        not after throttling. The caller holds the semaphore for the
        full duration of the LLM request.
        """
        # Layer 2 + 3: rate buckets (semaphore held by caller)
        await self._rpm_bucket.acquire()
        await self._rpd_bucket.acquire()

        logger.debug(
            "Throttle passed | rpm_tokens=%.2f | rpd_tokens=%.2f",
            self._rpm_bucket.available_tokens,
            self._rpd_bucket.available_tokens,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Rate-limited single-turn generation.

        Acquires semaphore + both token buckets before delegating
        to the wrapped provider. Semaphore is held for the full
        duration of the API call (not just during throttle check).

        Args:
            prompt: Input text prompt.
            **kwargs: Forwarded to the wrapped provider unchanged.

        Returns:
            LLMResponse from the wrapped provider.
        """
        async with self._semaphore: 
            # First acquire the semaphore slot to limit concurrency, then apply rate limiting before the actual API call.
            await self._throttle() # Layer 2 + 3: wait for rate limits
            return await self._provider.generate(prompt, **kwargs)

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Rate-limited multi-turn chat.

        Args:
            messages: List of message dicts forwarded unchanged.
            **kwargs: Forwarded to the wrapped provider unchanged.

        Returns:
            LLMResponse from the wrapped provider.
        """
        async with self._semaphore:
            # First acquire the semaphore slot to limit concurrency, then apply rate limiting before the actual API call.
            await self._throttle()  # Layer 2 + 3: wait for rate limits
            return await self._provider.chat(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Token counting — NOT rate limited.

        count_tokens() is a lightweight metadata call. Rate limiting
        it would throttle the pipeline's context-window decisions,
        which is counterproductive. Delegates directly.

        Args:
            text: Input text to count tokens for.

        Returns:
            Token count from the wrapped provider.
        """
        return await self._provider.count_tokens(text)

    async def is_available(self) -> bool:
        """Health check — delegates to the wrapped provider.

        Returns:
            True if the underlying provider is reachable.
        """
        return await self._provider.is_available()

    @property
    def provider_name(self) -> str:
        """Returns the wrapped provider's name unchanged."""
        return self._provider.provider_name

    @property
    def model_name(self) -> str:
        """Returns the wrapped provider's model name unchanged."""
        return self._provider.model_name
