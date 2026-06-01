"""Abstract base class defining the contract all LLM providers must satisfy."""

from abc import ABC, abstractmethod

from llm.models.llm_response import LLMResponse


class BaseLLM(ABC):
    """Abstract contract for all LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation."""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation in OpenAI message format."""

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens for the given text; critical for context window decisions."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Health check — verify the provider can accept requests right now."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string e.g. 'openai', 'gemini'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier string."""

    async def fits_context(self, text: str, max_tokens: int) -> bool:
        """RLM decision helper — True if token count of text fits within max_tokens."""
        return await self.count_tokens(text) <= max_tokens

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(provider={self.provider_name}, model={self.model_name})"
        )
