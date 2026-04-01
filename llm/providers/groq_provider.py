"""
Groq provider — OpenAI-compatible subclass.

Groq exposes an OpenAI-compatible REST API, so this provider inherits
all logic from OpenAIProvider (call, parse, error handling, token counting)
and only overrides:
    - __init__  : uses groq_api_key, GROQ_MODEL_STRONG, and Groq base_url
    - provider_name : returns "groq" for logging and tracing

Model roles (configured in settings / .env):
    GROQ_MODEL_FAST     llama-3.1-8b-instant      classify, decompose, simple
    GROQ_MODEL_STRONG   llama-3.3-70b-versatile   final synthesis (default)
    GROQ_MODEL_FALLBACK qwen/qwen3-32b             fallback if strong hits 429
"""

from typing import Optional

from llm.providers.openai_provider import OpenAIProvider
from llm.exceptions.llm_exceptions import LLMAuthError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    """Groq implementation of BaseLLM via OpenAI-compatible API.

    Inherits all API call logic, error translation, and token counting
    from OpenAIProvider. Only the client base_url and API key differ.

    Default model is GROQ_MODEL_STRONG (llama-3.3-70b-versatile).
    Pass model= explicitly to use GROQ_MODEL_FAST or GROQ_MODEL_FALLBACK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize Groq provider.

        Args:
            api_key: Groq API key. Falls back to settings.groq_api_key.
            model: Model name. Falls back to settings.GROQ_MODEL_STRONG.
            temperature: Sampling temperature. Falls back to settings.
            max_tokens: Max output tokens. Falls back to settings.
            timeout: Request timeout in seconds. Falls back to settings.

        Raises:
            LLMAuthError: If no Groq API key is available.
        """
        resolved_key = api_key or settings.groq_api_key
        if not resolved_key:
            raise LLMAuthError(
                "Groq API key is required. "
                "Set GROQ_API_KEY in .env or pass api_key argument."
            )

        super().__init__(
            api_key=resolved_key,
            model=model or settings.GROQ_MODEL_STRONG,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            base_url=_GROQ_BASE_URL,
        )

        logger.info(
            "GroqProvider initialized | model=%s",
            self._model,
        )

    @property
    def provider_name(self) -> str:
        """Returns provider identifier."""
        return "groq"
