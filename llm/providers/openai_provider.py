"""
OpenAI implementation of BaseLLM.

Supports:
    - GPT-4o
    - GPT-4o Mini
    - Any OpenAI chat completion model

Responsibilities:
    - Call OpenAI API via AsyncOpenAI client.
    - Translate OpenAI SDK errors → LLMError hierarchy.
    - Parse finish_reason from response (already lowercase from OpenAI).
    - Return standard LLMResponse always.

OpenAI-specific notes:
    - count_tokens() uses tiktoken (CPU-only, fast, local). Declared async
      to satisfy BaseLLM ABC, but does not perform I/O.
    - finish_reason comes as lowercase string directly from OpenAI API:
      'stop', 'length', 'content_filter', 'tool_calls'.
"""

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
    """OpenAI implementation of BaseLLM.

    Attributes:
        _client: AsyncOpenAI client instance.
        _encoder: tiktoken encoder for local token counting.
        _model: Active model name.
        _temperature: Default sampling temperature.
        _max_tokens: Default max output tokens.
        _timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Falls back to settings.openai_api_key.
            model: Model name. Falls back to settings.openai_model.
            temperature: Sampling temperature 0.0–2.0. Falls back to settings.
            max_tokens: Max tokens in response. Falls back to settings.
            timeout: Request timeout in seconds. Falls back to settings.
            base_url: Optional API base URL. Used by OpenAI-compatible providers
                (e.g. Groq). None uses the default OpenAI endpoint.

        Raises:
            LLMAuthError: If no API key is available from args or settings.
        """
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

        # Initialize async client — base_url allows OpenAI-compatible providers
        client_kwargs = {"api_key": self._api_key, "timeout": self._timeout}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

        # Initialize tiktoken encoder for local token counting
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

    # Properties — required by BaseLLM

    @property
    def provider_name(self) -> str:
        """Returns provider identifier.

        Returns:
            The string 'openai'.
        """
        return "openai"

    @property
    def model_name(self) -> str:
        """Returns the active model name.

        Returns:
            Model name string e.g. 'gpt-4o-mini'.
        """
        return self._model

    # Abstract method implementations

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation.

        Wraps the prompt in a user message and calls chat completion.

        Args:
            prompt: Input text prompt.
            **kwargs: Per-call overrides for temperature, max_tokens.

        Returns:
            LLMResponse with generated text, token usage, and timing.

        Raises:
            LLMAuthError: If API key is invalid.
            LLMRateLimitError: If quota is exceeded.
            LLMTimeoutError: If request deadline is exceeded.
            LLMTokenLimitError: If prompt exceeds context window.
            LLMProviderError: For any other provider-side failure.
        """
        messages = self._build_messages(prompt)
        return await self._call_api(messages, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation.

        Args:
            messages: List of dicts [{"role": "user", "content": "..."}].
                Roles: 'user', 'assistant', 'system'.
            **kwargs: Per-call overrides for temperature, max_tokens.

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
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        return await self._call_api(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken.

        CPU-only operation (no I/O). Declared async to satisfy BaseLLM ABC
        because GeminiProvider's implementation makes an API call.

        Args:
            text: Input text to count tokens for.

        Returns:
            Token count as integer. Returns 0 for empty input.
        """
        if not text:
            return 0

        return len(self._encoder.encode(text))

    async def is_available(self) -> bool:
        """Health check — sends minimal request to verify API reachability.

        Returns:
            True if OpenAI API responds successfully.
            False if the health check fails for any reason.
        """
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

    # Private methods

    async def _call_api(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Core API call with timing, error handling, and response parsing.

        Args:
            messages: OpenAI-formatted message list.
            **kwargs: Per-call overrides for temperature, max_tokens.

        Returns:
            LLMResponse with generated text, token usage, and timing.

        Raises:
            LLMAuthError: If API key is invalid.
            LLMRateLimitError: If quota is exceeded.
            LLMTimeoutError: If request deadline is exceeded.
            LLMTokenLimitError: If prompt exceeds context window.
            LLMProviderError: For any other provider-side failure.
        """
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
            # Already translated — re-raise without wrapping again
            raise

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            err_str = str(exc)
            # Zscaler/proxy block pages return full HTML in the exception body.
            # Log a clean message instead of dumping thousands of lines of HTML.
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
            # _handle_error always raises, but this satisfies the type checker
            raise LLMProviderError(
                f"Unhandled error in OpenAI provider. | {err_display}"
            )

    def _build_messages(self, prompt: str) -> list[dict]:
        """Wrap a plain prompt into OpenAI message format.

        Args:
            prompt: Raw text prompt.

        Returns:
            List with a single user message dict.
        """
        return [{"role": "user", "content": prompt}]

    def _parse_response(self, response, latency_ms: float) -> LLMResponse:
        """Parse raw OpenAI response into standard LLMResponse.

        OpenAI finish_reason values are already lowercase strings:
        'stop', 'length', 'content_filter', 'tool_calls'.

        Args:
            response: Raw OpenAI API response object.
            latency_ms: Time taken for the API call.

        Returns:
            LLMResponse with normalized fields.
        """
        choice = response.choices[0]
        usage = response.usage

        # OpenAI finish_reason is already lowercase — no mapping needed
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
        """Translate OpenAI SDK errors → LLMError hierarchy.

        Pipeline catches our errors — never OpenAI SDK errors directly.

        Args:
            error: Raw OpenAI exception.

        Raises:
            LLMAuthError: For authentication failures (401/403).
            LLMRateLimitError: For quota exceeded (429).
            LLMTimeoutError: For request deadline exceeded.
            LLMTokenLimitError: For context window exceeded.
            LLMProviderError: For all other errors.
        """
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
            if "<html" in err_str.lower() or "<!doctype" in err_str.lower():
                err_detail = "blocked by corporate proxy/firewall (HTML response received)"
            else:
                err_detail = err_str[:200]
            raise LLMProviderError(
                f"OpenAI API error occurred. | {err_detail}"
            ) from error

        # Unknown error — still wrap in our hierarchy
        err_str = str(error)
        if "<html" in err_str.lower() or "<!doctype" in err_str.lower():
            err_str = "blocked by corporate proxy/firewall (HTML response received)"
        raise LLMProviderError(
            f"Unexpected error from OpenAI provider. | {err_str[:200]}"
        ) from error