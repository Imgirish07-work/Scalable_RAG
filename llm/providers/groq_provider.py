"""Groq LLM provider — OpenAI-compatible subclass of OpenAIProvider."""

from typing import Optional

import tiktoken

from llm.providers.openai_provider import OpenAIProvider
from llm.exceptions.llm_exceptions import LLMAuthError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    """Groq implementation of BaseLLM via OpenAI-compatible API; default model is GROQ_MODEL_STRONG."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize Groq provider with Groq-specific key and base URL."""
        resolved_key = api_key or settings.groq_api_key
        if not resolved_key:
            raise LLMAuthError(
                "Groq API key is required. "
                "Set GROQ_API_KEY in .env or pass api_key argument."
            )

        # groq-specific timeout — shorter than global so zscaler blocks fail fast
        resolved_timeout = timeout if timeout is not None else settings.GROQ_TIMEOUT

        super().__init__(
            api_key=resolved_key,
            model=model or settings.GROQ_MODEL_STRONG,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=resolved_timeout,
            base_url=_GROQ_BASE_URL,
            max_retries=0,  # disable sdk retries — GroqModelPool handles 429 by switching models
        )

        self._encoder = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "GroqProvider initialized | model=%s",
            self._model,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    async def count_tokens(self, text: str) -> int:
        """Count tokens via cl100k_base — closest tiktoken match for LLaMA's 128k vocab."""
        if not text:
            return 0
        return len(self._encoder.encode(text))
