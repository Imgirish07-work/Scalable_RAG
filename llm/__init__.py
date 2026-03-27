# from llm.contracts.base_llm import BaseLLM
# from llm.models.llm_response import LLMResponse, SUPPORTED_PROVIDERS
# from llm.exceptions.llm_exceptions import (
#     LLMError,
#     LLMAuthError,
#     LLMRateLimitError,
#     LLMTimeoutError,
#     LLMTokenLimitError,
#     LLMProviderError,
# )

# # Factory imported last — depends on everything above
# from llm.factory import LLMFactory

# __all__ = [
#     # Contract
#     "BaseLLM",
#     # DTO
#     "LLMResponse",
#     "SUPPORTED_PROVIDERS",
#     # Exceptions
#     "LLMError",
#     "LLMAuthError",
#     "LLMRateLimitError",
#     "LLMTimeoutError",
#     "LLMTokenLimitError",
#     "LLMProviderError",
#     # Factory
#     "LLMFactory",
# ]

"""
LLM layer public API.

This module exposes the complete LLM interface for all downstream layers:
    - Contract: BaseLLM (the only type pipeline code should reference)
    - DTO: LLMResponse (the only return type from any provider)
    - Exceptions: Full LLMError hierarchy (the only errors pipeline code catches)
    - Factory: LLMFactory (the only way to create provider instances)

Usage:
    from llm import LLMFactory, LLMResponse, LLMError

    llm = LLMFactory.create("gemini")
    response = await llm.generate("Hello")
    assert isinstance(response, LLMResponse)

Import order follows project convention: contract → DTO → exceptions → factory.
"""

# Contract — the abstract interface all providers implement
from llm.contracts.base_llm import BaseLLM

# DTO — standardized response model and constants
from llm.models.llm_response import (
    LLMResponse,
    SUPPORTED_PROVIDERS,
    VALID_FINISH_REASONS,
)

# Exceptions — full error hierarchy for uniform error handling
from llm.exceptions.llm_exceptions import (
    LLMError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMProviderError,
)

# Factory — depends on everything above, imported last
from llm.llm_factory import LLMFactory

__all__ = [
    # Contract
    "BaseLLM",
    # DTO
    "LLMResponse",
    "SUPPORTED_PROVIDERS",
    "VALID_FINISH_REASONS",
    # Exceptions
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMTokenLimitError",
    "LLMProviderError",
    # Factory
    "LLMFactory",
]