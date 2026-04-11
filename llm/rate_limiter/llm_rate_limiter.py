"""
Transparent rate-limiting wrapper around any BaseLLM provider.

Design:
    Decorator pattern. LLMRateLimiter IS-A BaseLLM and HAS-A BaseLLM. The
    pipeline calls generate() or chat() on this wrapper exactly as it would
    on the real provider — rate limiting is invisible to the caller.

    Three layers are applied in order before each LLM call:
        1. asyncio.Semaphore — caps simultaneous in-flight requests, preventing
           a thundering herd of coroutines from hitting the API simultaneously.
        2. RPM TokenBucket — enforces requests-per-minute quota.
        3. RPD TokenBucket — enforces requests-per-day quota.

    The semaphore is acquired first to bound the number of coroutines competing
    for rate-limit tokens. Without it, many coroutines would pile up inside
    the bucket's acquire() loop, consuming memory and CPU while waiting.

Chain of Responsibility:
    LLMFactory.create_rate_limited() wraps provider → pipeline calls
    generate()/chat() on this wrapper → wrapper throttles → delegates to
    the real provider (GeminiProvider, OpenAIProvider, or GroqProvider).

Dependencies:
    asyncio, llm.contracts.base_llm, llm.models.llm_response,
    llm.rate_limiter.rate_limiter_config, llm.rate_limiter.token_bucket,
    utils.logger.
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

    Drop-in replacement — wraps an existing provider and enforces:
        - Concurrency cap via asyncio.Semaphore
        - Per-minute rate limit via RPM TokenBucket
        - Per-day rate limit via RPD TokenBucket

    Attributes:
        _provider: The wrapped BaseLLM provider receiving the real API calls.
        _config: Rate limiter configuration (rpm, rpd, max_concurrent, burst).
        _semaphore: asyncio.Semaphore enforcing the concurrency cap.
        _rpm_bucket: TokenBucket enforcing the per-minute request rate.
        _rpd_bucket: TokenBucket enforcing the per-day request rate.
    """

    def __init__(self, provider: BaseLLM, config: RateLimiterConfig) -> None:
        """Wrap a provider with semaphore and token bucket rate limiting.

        Args:
            provider: Any BaseLLM implementation to wrap.
            config: Rate limits and concurrency settings.
        """
        self._provider = provider
        self._config = config

        # Layer 1: semaphore caps concurrency before any bucket checks
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        # Layer 2: RPM bucket — capacity allows short bursts above sustained rate
        # refill_rate = rpm / 60 → tokens per second
        self._rpm_bucket = TokenBucket(
            capacity=config.bucket_capacity,
            refill_rate=config.refill_rate,
        )

        # Layer 3: RPD bucket — no burst; daily cap is a hard limit
        # refill_rate = rpd / 86400 → tokens per second over 24 hours
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
        """Apply RPM and RPD token buckets before an LLM call.

        Called inside the semaphore context in generate() and chat().
        The semaphore (Layer 1) is already held by the caller — this method
        applies Layers 2 and 3 only.

        Note: The semaphore is not used as a context manager here because
        it must remain held for the full duration of the LLM request,
        not just for the throttle check.
        """
        await self._rpm_bucket.acquire()
        await self._rpd_bucket.acquire()

        logger.debug(
            "Throttle passed | rpm_tokens=%.2f | rpd_tokens=%.2f",
            self._rpm_bucket.available_tokens,
            self._rpd_bucket.available_tokens,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Rate-limited single-turn generation.

        Acquires the semaphore slot and both token buckets before delegating
        to the wrapped provider. The semaphore is held for the full duration
        of the API call, not just during the throttle check.

        Args:
            prompt: Input text prompt.
            **kwargs: Forwarded to the wrapped provider unchanged.

        Returns:
            LLMResponse from the wrapped provider.
        """
        async with self._semaphore:
            await self._throttle()
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
            await self._throttle()
            return await self._provider.chat(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Token counting — bypasses rate limiting and delegates directly.

        count_tokens() is a lightweight metadata call used for context window
        decisions. Rate-limiting it would throttle pipeline planning logic,
        which is counterproductive.

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
