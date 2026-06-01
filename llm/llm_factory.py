"""Factory for creating and configuring LLM provider instances via a class-level registry."""

from typing import Optional

from llm.contracts.base_llm import BaseLLM
from llm.exceptions.llm_exceptions import LLMProviderError
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.groq_provider import GroqProvider
from llm.providers.groq_model_pool import GroqModelPool
from llm.rate_limiter import LLMRateLimiter, get_rate_limit_config
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Creates and configures LLM provider instances from a registry; all methods are classmethods."""

    _registry: dict[str, type[BaseLLM]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> BaseLLM:
        """Create and return a raw (non-rate-limited) LLM provider instance."""
        cleaned_name = cls._validate_provider(provider_name)
        provider_class = cls._registry[cleaned_name]

        logger.info("Creating LLM provider | provider=%s", cleaned_name)

        kwargs = cls._build_kwargs(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        return provider_class(**kwargs)

    @classmethod
    def create_rate_limited(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> BaseLLM:
        """Create a provider wrapped with LLMRateLimiter; skips wrapping if rate limiter disabled."""
        provider = cls.create(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        if not settings.LLM_RATE_LIMITER_ENABLED:
            return provider

        return LLMRateLimiter(
            provider=provider,
            config=get_rate_limit_config(
                model_name=provider.model_name,
                max_concurrent=settings.LLM_MAX_CONCURRENT,
                burst_multiplier=settings.LLM_BURST_MULTIPLIER,
            ),
        )

    @classmethod
    def create_groq_pool(
        cls,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> GroqModelPool:
        """Create a GroqModelPool that dynamically routes across multiple Groq models."""
        logger.info("Creating GroqModelPool (multi-model dynamic routing)")

        return GroqModelPool(
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @classmethod
    def create_from_settings(cls) -> BaseLLM:
        """Create a provider from settings.default_provider; uses GroqModelPool when provider=='groq'."""
        provider_name = settings.default_provider

        logger.info(
            "Creating LLM provider from settings | provider=%s",
            provider_name,
        )

        # groq uses the multi-model pool with reactive header-based rate limiting
        if provider_name.strip().lower() == "groq":
            return cls.create_groq_pool()

        return cls.create_rate_limited(provider_name)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return all registered provider names sorted alphabetically."""
        return sorted(cls._registry.keys())

    @classmethod
    def register(cls, provider_name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new provider at runtime; validates that provider_class subclasses BaseLLM."""
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseLLM)):
            raise LLMProviderError(
                f"Cannot register '{provider_name}'. "
                f"Provider class must implement BaseLLM."
            )

        cls._registry[provider_name.strip().lower()] = provider_class
        logger.info("Registered new LLM provider | provider=%s", provider_name)

    @classmethod
    def _validate_provider(cls, provider_name: str) -> str:
        """Validate the provider name and return its cleaned lowercase form."""
        if not provider_name or not provider_name.strip():
            raise LLMProviderError(
                "Provider name cannot be empty. "
                f"Available providers: {cls.available_providers()}"
            )

        cleaned = provider_name.strip().lower()

        if cleaned not in cls._registry:
            raise LLMProviderError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {cls.available_providers()}"
            )

        return cleaned

    @classmethod
    def _build_kwargs(
        cls,
        api_key: Optional[str],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout: Optional[float],
    ) -> dict:
        """Build a kwargs dict containing only non-None overrides."""
        kwargs = {}

        if api_key is not None:
            kwargs["api_key"] = api_key
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout

        return kwargs
