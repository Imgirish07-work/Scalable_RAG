"""
Factory for creating LLM provider instances.

Design:
    - Pipeline calls LLMFactory.create() — never imports providers directly.
    - Registry dict maps provider name → provider class.
    - Adding a new provider = 1 line in _registry, nothing else changes.
    - register() allows runtime provider registration for plugins and tests.

Usage:
    llm = LLMFactory.create("openai")
    llm = LLMFactory.create("gemini", temperature=0.7)
    llm = LLMFactory.create_from_settings()   # reads .env defaults
"""

from typing import Optional

from llm.contracts.base_llm import BaseLLM
from llm.exceptions.llm_exceptions import LLMProviderError
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.gemini_provider import GeminiProvider
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating LLM provider instances.

    Class-level registry maps provider names to their implementation classes.
    All public methods are classmethods — no instantiation needed.

    Attributes:
        _registry: Dict mapping provider name → BaseLLM subclass.
    """

    # Registry — single source of truth for all providers
    _registry: dict[str, type[BaseLLM]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
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
        """Create and return an LLM provider instance.

        Only passes explicitly-provided overrides to the provider constructor.
        The provider's __init__ falls back to settings for anything not passed.

        Args:
            provider_name: Provider to create e.g. 'openai' or 'gemini'.
            api_key: Optional override for API key.
            model: Optional override for model name.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max output tokens.
            timeout: Optional override for request timeout.

        Returns:
            BaseLLM instance — always returns the contract, never a concrete class.

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
    def create_from_settings(cls) -> BaseLLM:
        """Create provider using default_provider from settings.

        Reads the provider name from .env / settings automatically.
        Use this in pipeline code — no hardcoded provider names.

        Returns:
            BaseLLM instance configured from settings.

        Raises:
            LLMProviderError: If default_provider in settings is not registered.
        """
        provider_name = settings.default_provider

        logger.info(
            "Creating LLM provider from settings | provider=%s",
            provider_name,
        )

        return cls.create(provider_name)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return list of all registered provider names.

        Useful for validation, logging, health checks, and CLI help text.

        Returns:
            Sorted list of provider name strings.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def register(cls, provider_name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new provider at runtime.

        Allows external plugins or test mocks to add providers without
        modifying the factory source code.

        Args:
            provider_name: Unique name string e.g. 'anthropic'.
            provider_class: Class that implements BaseLLM.

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
        """Validate that provider name exists in the registry.

        Args:
            provider_name: Raw provider name string.

        Returns:
            Cleaned lowercase provider name.

        Raises:
            LLMProviderError: If provider name is empty or not registered.
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
        """Build kwargs dict with only explicitly-provided overrides.

        Provider __init__ falls back to settings for anything not in this dict.

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