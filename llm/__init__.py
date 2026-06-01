"""LLM layer public API — BaseLLM contract, LLMResponse DTO, exception hierarchy, factory."""

from llm.contracts.base_llm import BaseLLM

from llm.models.llm_response import (
    LLMResponse,
    SUPPORTED_PROVIDERS,
    VALID_FINISH_REASONS,
)

from llm.exceptions.llm_exceptions import (
    LLMError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMProviderError,
)

from llm.llm_factory import LLMFactory

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "SUPPORTED_PROVIDERS",
    "VALID_FINISH_REASONS",
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMTokenLimitError",
    "LLMProviderError",
    "LLMFactory",
]