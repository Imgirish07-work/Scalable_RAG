# from abc import ABC, abstractmethod
# from llm.models.llm_response import LLMResponse

# class BaseLLM(ABC):
#     """
#     Abstract contract for all LLM providers.

#     Rules:
#         - Pipeline imports ONLY BaseLLM — never OpenAI or Gemini directly
#         - Every provider MUST implement all abstract methods
#         - fits_context() is shared logic — all providers inherit free
#     """

#     # Abstract Methods — must be implemented by every provider
#     @abstractmethod
#     async def generate(self, prompt: str, **kwargs) -> LLMResponse:
#         """
#         Single-turn text generation.
#         Use for: summarization, extraction, classification.
#         """
#         pass

#     @abstractmethod
#     async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
#         """
#         Multi-turn conversation.
#         messages format: [{"role": "user", "content": "..."}]
#         Use for: agents, multi-hop reasoning, ReAct loops.
#         """
#         pass

#     @abstractmethod
#     async def count_tokens(self, text: str) -> int:
#         """
#         Count tokens for a given text.
#         Critical for RLM — must know if text fits context window.
#         """
#         pass

#     @abstractmethod
#     async def is_available(self) -> bool:
#         """
#         Health check — can this provider accept requests right now?
#         Used by smart router for fallback decisions.
#         """
#         pass

#     @property
#     @abstractmethod
#     def provider_name(self) -> str:
#         """Returns provider identifier e.g. 'openai' or 'gemini'."""
#         pass

#     @property
#     @abstractmethod
#     def model_name(self) -> str:
#         """Returns active model name e.g. 'gpt-4o-mini'."""
#         pass

#     #  Concrete Methods — shared logic, all providers inherit free
#     async def fits_context(self, text: str, max_tokens: int) -> bool:
#         """
#         RLM decision helper.
#         IF True  → direct LLM call
#         IF False → chunk → process → combine → recurse
#         """
#         return await self.count_tokens(text) <= max_tokens

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}"
#             f"(provider={self.provider_name}, model={self.model_name})"
#         )

"""
Abstract contract for all LLM providers.

Rules:
    - Pipeline imports ONLY BaseLLM — never OpenAIProvider or GeminiProvider.
    - LLMFactory returns BaseLLM instances — callers never know the concrete type.
    - Every provider MUST implement all abstract methods.
    - fits_context() is shared logic — all providers inherit it free.

Note on async in this ABC:
    count_tokens() is declared async because GeminiProvider's implementation
    makes an API call (I/O-bound). OpenAIProvider's implementation uses tiktoken
    (CPU-only), which technically violates Rule 2, but the ABC interface must be
    async to accommodate the I/O-bound implementation. This is an accepted
    tradeoff — the alternative (two separate interfaces) adds complexity without
    meaningful benefit.
"""

from abc import ABC, abstractmethod

from llm.models.llm_response import LLMResponse


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement:
        - generate()      — single-turn text generation
        - chat()          — multi-turn conversation
        - count_tokens()  — token counting for context window decisions
        - is_available()  — health check for routing / fallback
        - provider_name   — identifier string (property)
        - model_name      — active model string (property)
    """

    # Abstract methods — must be implemented by every provider

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation.

        Use for: summarization, extraction, classification, RAG generation.

        Args:
            prompt: Input text prompt.
            **kwargs: Per-call overrides (temperature, max_tokens).

        Returns:
            LLMResponse with generated text, token usage, and timing.

        Raises:
            LLMAuthError: If API key is invalid.
            LLMRateLimitError: If quota is exceeded.
            LLMTimeoutError: If request deadline is exceeded.
            LLMTokenLimitError: If prompt exceeds context window.
            LLMProviderError: For any other provider-side failure.
        """

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation.

        Message format follows OpenAI convention (providers convert internally):
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Use for: agents, multi-hop reasoning, ReAct loops, conversation-aware RAG.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Accepted roles: 'user', 'assistant', 'system'.
            **kwargs: Per-call overrides (temperature, max_tokens).

        Returns:
            LLMResponse with generated text, token usage, and timing.

        Raises:
            ValueError: If messages list is empty.
            LLMAuthError: If API key is invalid.
            LLMRateLimitError: If quota is exceeded.
            LLMTimeoutError: If request deadline is exceeded.
            LLMTokenLimitError: If prompt exceeds context window.
            LLMProviderError: For any other provider-side failure.
        """

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens for a given text.

        Critical for RLM — determines whether text fits the context window
        or needs to be chunked and processed recursively.

        Implementation note:
            - OpenAI: uses tiktoken locally (CPU-only, fast).
            - Gemini: calls the count_tokens API (I/O, accurate).

        Args:
            text: Input text to count tokens for.

        Returns:
            Token count as integer. Returns 0 for empty input.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Health check — can this provider accept requests right now?

        Used by the smart router (Layer 5) for fallback decisions.
        Sends a minimal request to verify API reachability.

        Returns:
            True if the provider is reachable and responding.
            False if the health check fails for any reason.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string.

        Returns:
            Lowercase provider name e.g. 'openai', 'gemini'.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier string.

        Returns:
            Model name e.g. 'gpt-4o-mini', 'gemini-2.5-flash'.
        """

    # Concrete methods — shared logic, all providers inherit free

    async def fits_context(self, text: str, max_tokens: int) -> bool:
        """RLM decision helper — does this text fit the context window?

        If True  → direct LLM call (no chunking needed).
        If False → chunk → process → combine → recurse.

        Args:
            text: Input text to check.
            max_tokens: Maximum token budget for the context window.

        Returns:
            True if token count of text <= max_tokens.
        """
        return await self.count_tokens(text) <= max_tokens

    def __repr__(self) -> str:
        """Human-readable representation for logging and debugging.

        Returns:
            String like 'GeminiProvider(provider=gemini, model=gemini-2.5-flash)'.
        """
        return (
            f"{self.__class__.__name__}"
            f"(provider={self.provider_name}, model={self.model_name})"
        )