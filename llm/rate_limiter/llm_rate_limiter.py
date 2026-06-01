"""Transparent rate-limiting wrapper around any BaseLLM provider — semaphore + RPM/RPD token buckets."""

import asyncio

from llm.contracts.base_llm import BaseLLM
from llm.models.llm_response import LLMResponse
from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from llm.rate_limiter.token_bucket import TokenBucket
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMRateLimiter(BaseLLM):
    """Transparent rate-limiting wrapper around any BaseLLM provider."""

    def __init__(self, provider: BaseLLM, config: RateLimiterConfig) -> None:
        """Wrap a provider with semaphore and token bucket rate limiting."""
        self._provider = provider
        self._config = config

        # layer 1: semaphore caps concurrency before any bucket checks
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        # layer 2: rpm bucket — capacity allows short bursts above sustained rate
        self._rpm_bucket = TokenBucket(
            capacity=config.bucket_capacity,
            refill_rate=config.refill_rate,
        )

        # layer 3: rpd bucket — no burst; daily cap is a hard limit
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
        """Apply RPM and RPD token buckets before an LLM call; caller already holds the semaphore."""
        await self._rpm_bucket.acquire()
        await self._rpd_bucket.acquire()

        logger.debug(
            "Throttle passed | rpm_tokens=%.2f | rpd_tokens=%.2f",
            self._rpm_bucket.available_tokens,
            self._rpd_bucket.available_tokens,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Rate-limited single-turn generation; semaphore held for the full API call."""
        logger.info(
            "Request added to queue | provider=%s | type=generate | "
            "slots_free=%d/%d | query='%s'",
            self._provider.provider_name,
            max(0, self._semaphore._value),
            self._config.max_concurrent,
            prompt[:100],
        )
        async with self._semaphore:
            await self._throttle()
            return await self._provider.generate(prompt, **kwargs)

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Rate-limited multi-turn chat."""
        last_user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        logger.info(
            "Request added to queue | provider=%s | type=chat | "
            "slots_free=%d/%d | query='%s'",
            self._provider.provider_name,
            self._semaphore._value,
            self._config.max_concurrent,
            last_user_msg[:100],
        )
        async with self._semaphore:
            await self._throttle()
            return await self._provider.chat(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Token counting — bypasses rate limiting and delegates directly."""
        return await self._provider.count_tokens(text)

    async def is_available(self) -> bool:
        """Health check — delegates to the wrapped provider."""
        return await self._provider.is_available()

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def model_name(self) -> str:
        return self._provider.model_name
