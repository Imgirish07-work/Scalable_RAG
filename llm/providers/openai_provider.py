"""OpenAI implementation of BaseLLM; also base class for GroqProvider via OpenAI-compatible API."""

import time
from typing import List, Optional

import tiktoken
from openai import AsyncOpenAI
from openai import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIError,
)

from llm.contracts.base_llm import BaseLLM
from llm.models.llm_response import LLMResponse
from llm.exceptions.llm_exceptions import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMProviderError,
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseLLM):
    """OpenAI implementation of BaseLLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize the AsyncOpenAI client and tiktoken encoder."""
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None else settings.max_tokens
        )
        self._timeout = (
            timeout if timeout is not None else settings.request_timeout
        )

        if not self._api_key:
            raise LLMAuthError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in .env or pass api_key argument."
            )

        # base_url enables openai-compatible providers (e.g. groq) without subclassing _call_api
        client_kwargs = {"api_key": self._api_key, "timeout": self._timeout}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = AsyncOpenAI(**client_kwargs)

        # tiktoken may lack an encoding for newer/custom models — fall back gracefully
        try:
            self._encoder = tiktoken.encoding_for_model(self._model)
        except KeyError:
            logger.warning(
                "tiktoken has no encoding for model=%s, falling back to o200k_base",
                self._model,
            )
            self._encoder = tiktoken.get_encoding("o200k_base")

        logger.info(
            "OpenAIProvider initialized | model=%s | max_tokens=%s",
            self._model,
            self._max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation."""
        messages = self._build_messages(prompt)
        return await self._call_api(messages, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        return await self._call_api(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (CPU-only, no I/O)."""
        if not text:
            return 0

        return len(self._encoder.encode(text))

    async def is_available(self) -> bool:
        """Health check — send a minimal request to verify API reachability."""
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as exc:
            logger.warning("OpenAI health check failed | error=%s", str(exc))
            return False

    async def _call_api(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Execute the OpenAI chat completion call with timing and error handling."""
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        start_time = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - start_time) * 1000
            return self._parse_response(response, latency_ms)

        except (
            LLMAuthError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMTokenLimitError,
            LLMProviderError,
        ):
            # already translated — re-raise without double-wrapping
            raise

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            err_str = str(exc)
            # zscaler/proxy block pages return html — log a clean message
            if "<html" in err_str.lower() or "<!doctype" in err_str.lower():
                err_display = "blocked by corporate proxy/firewall (HTML response received)"
            else:
                err_display = err_str[:200]
            logger.error(
                "OpenAI API call failed | latency_ms=%.1f | error=%s",
                latency_ms,
                err_display,
            )
            self._handle_error(exc)
            # _handle_error always raises; satisfies the type checker
            raise LLMProviderError(
                f"Unhandled error in OpenAI provider. | {err_display}"
            )

    def _build_messages(self, prompt: str) -> list[dict]:
        """Wrap a plain prompt string into OpenAI message format."""
        return [{"role": "user", "content": prompt}]

    def _parse_response(self, response, latency_ms: float) -> LLMResponse:
        """Parse a raw OpenAI response into a standard LLMResponse."""
        choice = response.choices[0]
        usage = response.usage

        finish_reason = choice.finish_reason or "unknown"

        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tokens_used=usage.total_tokens,
            latency_ms=round(latency_ms, 2),
        )

    def _handle_error(self, error: Exception) -> None:
        """Translate OpenAI SDK exceptions into the LLMError hierarchy."""
        if isinstance(error, AuthenticationError):
            raise LLMAuthError(
                f"OpenAI authentication failed. Check your API key. | {error}"
            ) from error

        if isinstance(error, RateLimitError):
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded. Retry after delay. | {error}"
            ) from error

        if isinstance(error, APITimeoutError):
            raise LLMTimeoutError(
                f"OpenAI request timed out after {self._timeout}s. | {error}"
            ) from error

        if isinstance(error, BadRequestError):
            error_message = str(error).lower()
            if "context_length_exceeded" in error_message or "maximum context" in error_message:
                raise LLMTokenLimitError(
                    f"Prompt exceeds OpenAI model context window. | {error}"
                ) from error

        if isinstance(error, APIError):
            err_str = str(error)
            # sanitize proxy block html before raising
            if "<html" in err_str.lower() or "<!doctype" in err_str.lower():
                err_detail = "blocked by corporate proxy/firewall (HTML response received)"
            else:
                err_detail = err_str[:200]
            raise LLMProviderError(
                f"OpenAI API error occurred. | {err_detail}"
            ) from error

        # catch-all — sanitize html before raising
        err_str = str(error)
        if "<html" in err_str.lower() or "<!doctype" in err_str.lower():
            err_str = "blocked by corporate proxy/firewall (HTML response received)"
        raise LLMProviderError(
            f"Unexpected error from OpenAI provider. | {err_str[:200]}"
        ) from error
