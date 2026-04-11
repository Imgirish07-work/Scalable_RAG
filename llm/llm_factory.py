"""
Factory for creating and configuring LLM provider instances.

Design:
    Class-level registry pattern. The registry dict maps provider name strings
    to their concrete BaseLLM subclasses. Adding a new provider requires only
    one registry entry — no other code changes. All public methods are
    classmethods so no factory instantiation is needed at the call site.

Chain of Responsibility:
    Pipeline calls LLMFactory.create() or create_from_settings() →
    factory instantiates the concrete provider → wraps it in LLMRateLimiter
    when enabled → returns BaseLLM to RAGPipeline.configure_llm().

Dependencies:
    llm.contracts.base_llm, llm.providers.*, llm.exceptions.llm_exceptions,
    llm.rate_limiter (LLMRateLimiter, get_rate_limit_config),
    config.settings, utils.logger.
"""

from typing import Optional

from llm.contracts.base_llm import BaseLLM
from llm.exceptions.llm_exceptions import LLMProviderError
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.groq_provider import GroqProvider
from llm.rate_limiter import LLMRateLimiter, get_rate_limit_config
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Creates and configures LLM provider instances from a registry.

    All methods are classmethods — no instantiation of LLMFactory is needed.

    Attributes:
        _registry: Maps provider name strings to their BaseLLM subclasses.
    """

    # Single source of truth for all registered providers
    _registry: dict[str, type[BaseLLM]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
    }

    # Public methods

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
        """Create and return a raw (non-rate-limited) LLM provider instance.

        Only passes explicitly-provided overrides to the constructor.
        The provider's __init__ falls back to settings for any unset parameter.

        Args:
            provider_name: Provider to create e.g. 'openai', 'gemini', 'groq'.
            api_key: Optional API key override.
            model: Optional model name override.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max output tokens override.
            timeout: Optional request timeout override in seconds.

        Returns:
            BaseLLM instance — always the contract type, never a concrete class.

        Raises:
            LLMProviderError: If provider_name is empty or not in the registry.
        """
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
        """Create a provider and wrap it with the per-model rate limiter.

        Single call that handles both creation and rate limiting, preventing
        any code path from accidentally using an unthrottled raw provider.
        Rate limiting is skipped when LLM_RATE_LIMITER_ENABLED=False
        (used in tests and local dev).

        Args:
            provider_name: Provider name e.g. 'groq', 'gemini', 'openai'.
            api_key: Optional API key override.
            model: Optional model name override.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            timeout: Optional request timeout override.

        Returns:
            LLMRateLimiter-wrapped BaseLLM when rate limiting is enabled,
            raw provider BaseLLM otherwise.
        """
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
    def create_from_settings(cls) -> BaseLLM:
        """Create a rate-limited provider from the default_provider setting.

        Always returns a rate-limited provider when LLM_RATE_LIMITER_ENABLED
        is True — no manual wrapping is needed at the call site.

        Returns:
            Rate-limited BaseLLM configured from settings.

        Raises:
            LLMProviderError: If default_provider in settings is not registered.
        """
        provider_name = settings.default_provider

        logger.info(
            "Creating LLM provider from settings | provider=%s",
            provider_name,
        )

        return cls.create_rate_limited(provider_name)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return all registered provider names sorted alphabetically.

        Useful for validation, health checks, and CLI help text.

        Returns:
            Sorted list of provider name strings.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def register(cls, provider_name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new provider at runtime.

        Allows external plugins or test mocks to add providers without
        modifying the factory source. Validates that the class implements BaseLLM.

        Args:
            provider_name: Unique identifier string e.g. 'anthropic'.
            provider_class: Class that subclasses BaseLLM.

        Raises:
            LLMProviderError: If provider_class does not subclass BaseLLM.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseLLM)):
            raise LLMProviderError(
                f"Cannot register '{provider_name}'. "
                f"Provider class must implement BaseLLM."
            )

        cls._registry[provider_name.strip().lower()] = provider_class
        logger.info("Registered new LLM provider | provider=%s", provider_name)

    # Private methods

    @classmethod
    def _validate_provider(cls, provider_name: str) -> str:
        """Validate the provider name and return its cleaned lowercase form.

        Args:
            provider_name: Raw provider name string from the caller.

        Returns:
            Cleaned lowercase provider name.

        Raises:
            LLMProviderError: If provider_name is empty or not in the registry.
        """
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
        """Build a kwargs dict containing only explicitly-provided overrides.

        Excludes None values so the provider's __init__ falls back to settings
        for any parameter not explicitly set by the caller.

        Args:
            api_key: Optional API key override.
            model: Optional model name override.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            timeout: Optional timeout override.

        Returns:
            Dict containing only non-None overrides.
        """
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
