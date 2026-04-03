"""
Google Gemini implementation of BaseLLM.

Supports:
    - gemini-2.5-flash  (free tier — primary for light tasks)
    - gemini-1.5-pro    (paid — heavy reasoning tasks)
    - gemini-2.0-flash  (latest fast model)

Responsibilities:
    - Call Gemini API via google.genai async client.
    - Translate Gemini SDK errors → LLMError hierarchy.
    - Normalize finish_reason from Gemini enum → lowercase string.
    - Return standard LLMResponse always.

Gemini-specific notes:
    - Gemini has no 'system' role — system messages are treated as user messages.
    - Gemini returns finish_reason as an enum (STOP, MAX_TOKENS, SAFETY, etc.)
      which we normalize to match OpenAI's lowercase convention.
    - count_tokens() makes an API call (I/O-bound, accurate) — unlike OpenAI's
      local tiktoken approach.
"""

import asyncio
import re
import time
from typing import List, Optional

from google import genai
from google.genai import types
from google.api_core.exceptions import (
    Unauthenticated,
    ResourceExhausted,
    DeadlineExceeded,
    InvalidArgument,
    GoogleAPIError,
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

# Gemini finish_reason enum → normalized lowercase string
# Matches OpenAI convention so downstream code doesn't branch on provider
_GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "safety",
    "RECITATION": "recitation",
    "OTHER": "other",
    "FINISH_REASON_UNSPECIFIED": "unknown",
}

# Rate-limit retry config
# Gemini 429 errors include "Please retry in X.Xs" — we parse and honor that delay
_RETRY_AFTER_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_DEFAULT_RETRY_DELAY_S = 65.0   # safe default when delay not parseable (1-min window)
_MAX_RATE_LIMIT_RETRIES = 2     # retry up to 2 times before raising LLMRateLimitError


def _parse_retry_after(error_message: str) -> float:
    """Extract retry-after delay from a Gemini 429 error message.

    Parses "Please retry in 37.5s" from the error text.
    Adds a 2-second buffer to the parsed delay to avoid immediate re-hit.

    Args:
        error_message: Raw exception string from ResourceExhausted.

    Returns:
        Seconds to wait before retrying.
    """
    match = _RETRY_AFTER_PATTERN.search(error_message)
    if match:
        return float(match.group(1)) + 2.0
    return _DEFAULT_RETRY_DELAY_S


class GeminiProvider(BaseLLM):
    """Google Gemini implementation of BaseLLM.

    Attributes:
        _client: google.genai.Client instance.
        _generation_config: Default generation parameters.
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
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Gemini API key. Falls back to settings.gemini_api_key.
            model: Model name. Falls back to settings.gemini_model.
            temperature: Sampling temperature 0.0–2.0. Falls back to settings.
            max_tokens: Max tokens in response. Falls back to settings.
            timeout: Request timeout in seconds. Falls back to settings.

        Raises:
            LLMAuthError: If no API key is available from args or settings.
        """
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model
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
                "Gemini API key is required. "
                "Set GEMINI_API_KEY in .env or pass api_key argument."
            )

        # Initialize Gemini client and default generation config
        self._client = genai.Client(api_key=self._api_key)
        self._generation_config = types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

        logger.info(
            "GeminiProvider initialized | model=%s | max_tokens=%s",
            self._model,
            self._max_tokens,
        )

    # Properties — required by BaseLLM

    @property
    def provider_name(self) -> str:
        """Returns provider identifier.

        Returns:
            The string 'gemini'.
        """
        return "gemini"

    @property
    def model_name(self) -> str:
        """Returns the active model name.

        Returns:
            Model name string e.g. 'gemini-2.5-flash'.
        """
        return self._model

    # Abstract method implementations

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation.

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
        contents = self._build_contents(prompt)
        return await self._call_api(contents, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation.

        Accepts OpenAI-format messages — converted to Gemini format internally.

        Args:
            messages: List of dicts [{"role": "user", "content": "..."}].
                Roles: 'user', 'assistant' (→ 'model'), 'system' (→ 'user').
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

        contents = self._convert_messages(messages)
        return await self._call_api(contents, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's native count_tokens API.

        More accurate than tiktoken for Gemini models because it uses
        the model's actual tokenizer. Makes an API call (I/O-bound).

        Args:
            text: Input text to count tokens for.

        Returns:
            Token count as integer. Returns 0 for empty input.
            Falls back to character-based estimate on API failure.
        """
        if not text:
            return 0

        try:
            result = await self._client.aio.models.count_tokens(
                model=self._model,
                contents=text,
            )
            return result.total_tokens
        except Exception as exc:
            logger.warning(
                "Gemini count_tokens failed, estimating via char count | error=%s",
                str(exc),
            )
            # Fallback — rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

    async def is_available(self) -> bool:
        """Health check — sends minimal request to verify API reachability.

        Returns:
            True if Gemini API responds successfully.
            False if the health check fails for any reason.
        """
        try:
            await self._client.aio.models.generate_content(
                model=self._model,
                contents="ping",
            )
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed | error=%s", str(exc))
            return False

    # Private methods

    async def _call_api(self, contents: list, **kwargs) -> LLMResponse:
        """Core API call with timing, error handling, and response parsing.

        Rebuilds generation config if per-call overrides are provided.

        Args:
            contents: Gemini-formatted message list.
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
        response_mime_type = kwargs.get("response_mime_type")
        thinking_budget = kwargs.get("thinking_budget")

        # Rebuild config only if overrides differ from defaults
        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            # AFC adds ~200ms overhead and is irrelevant for RAG generation
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )

        generation_config = types.GenerateContentConfig(**config_kwargs)

        last_exc: Exception | None = None
        for attempt in range(1 + _MAX_RATE_LIMIT_RETRIES):
            start_time = time.monotonic()
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=generation_config,
                )
                latency_ms = (time.monotonic() - start_time) * 1000
                return self._parse_response(response, latency_ms)

            except ResourceExhausted as exc:
                latency_ms = (time.monotonic() - start_time) * 1000
                last_exc = exc
                if attempt < _MAX_RATE_LIMIT_RETRIES:
                    delay = _parse_retry_after(str(exc))
                    logger.warning(
                        "Gemini rate limit hit (attempt %d/%d) | waiting %.1fs | error=%s",
                        attempt + 1,
                        1 + _MAX_RATE_LIMIT_RETRIES,
                        delay,
                        str(exc)[:120],
                    )
                    await asyncio.sleep(delay)
                    continue
                # Retries exhausted — translate and raise
                self._handle_error(exc)

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
                logger.error(
                    "Gemini API call failed | latency_ms=%.1f | error=%s",
                    latency_ms,
                    str(exc),
                )
                self._handle_error(exc)
                # _handle_error always raises, but this satisfies the type checker
                raise LLMProviderError(
                    f"Unhandled error in Gemini provider. | {exc}"
                )

        # Should never reach here, but satisfies the type checker
        raise LLMProviderError(f"Gemini call failed after retries. | {last_exc}")

    def _build_contents(self, prompt: str) -> list:
        """Wrap a plain prompt into Gemini contents format.

        Args:
            prompt: Raw text prompt.

        Returns:
            Gemini contents list with a single user message.
        """
        return [{"role": "user", "parts": [{"text": prompt}]}]

    def _convert_messages(self, messages: List[dict]) -> list:
        """Convert OpenAI message format → Gemini contents format.

        Mapping:
            OpenAI 'user'      → Gemini 'user'
            OpenAI 'assistant' → Gemini 'model'
            OpenAI 'system'    → Gemini 'user' (Gemini has no system role)

        Args:
            messages: OpenAI-format message list.

        Returns:
            Gemini-format contents list.
        """
        role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",
        }

        contents = []
        for message in messages:
            role = role_map.get(message["role"], "user")
            contents.append({
                "role": role,
                "parts": [{"text": message["content"]}],
            })

        return contents

    def _parse_response(self, response, latency_ms: float) -> LLMResponse:
        """Parse raw Gemini response into standard LLMResponse.

        Handles:
            - Token usage extraction from usage_metadata.
            - finish_reason normalization from Gemini enum to lowercase string.
            - Safety-blocked responses (candidates may be empty).

        Args:
            response: Raw Gemini API response object.
            latency_ms: Time taken for the API call.

        Returns:
            LLMResponse with normalized fields.

        Raises:
            LLMProviderError: If response has no usable text (safety block, etc.).
        """
        # Extract token usage from metadata
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        tokens_used = usage.total_token_count if usage else 0

        # Normalize finish_reason from Gemini enum
        finish_reason = "unknown"
        if response.candidates:
            raw_reason = response.candidates[0].finish_reason
            # Handle both enum objects (.name) and raw strings
            reason_str = (
                raw_reason.name
                if hasattr(raw_reason, "name")
                else str(raw_reason)
            )
            finish_reason = _GEMINI_FINISH_REASON_MAP.get(
                reason_str, reason_str.lower()
            )

        # Extract text — response.text raises if candidates are blocked
        try:
            text = response.text
        except (ValueError, AttributeError) as exc:
            logger.error(
                "Gemini returned no usable text | finish_reason=%s | error=%s",
                finish_reason,
                str(exc),
            )
            raise LLMProviderError(
                f"Gemini response blocked or empty. "
                f"finish_reason={finish_reason} | {exc}"
            ) from exc

        if not text or not text.strip():
            raise LLMProviderError(
                f"Gemini returned empty text. finish_reason={finish_reason}"
            )

        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_used=tokens_used,
            latency_ms=round(latency_ms, 2),
        )

    def _handle_error(self, error: Exception) -> None:
        """Translate Gemini SDK errors → LLMError hierarchy.

        Pipeline catches our errors — never Gemini SDK errors directly.
        Order matters: check string-based patterns first (400 errors with
        'api key not valid' come as generic exceptions, not Unauthenticated).

        Args:
            error: Raw Gemini exception.

        Raises:
            LLMAuthError: For authentication failures.
            LLMRateLimitError: For quota exceeded.
            LLMTimeoutError: For deadline exceeded.
            LLMTokenLimitError: For context window exceeded.
            LLMProviderError: For all other errors.
        """
        error_message = str(error).lower()

        # String-based check first — 400 errors with invalid API key
        # arrive as generic exceptions, not Unauthenticated
        if "api key not valid" in error_message or "api_key_invalid" in error_message:
            raise LLMAuthError(
                f"Gemini authentication failed. Check your API key. | {error}"
            ) from error

        # String-based rate limit check before isinstance — the new google.genai
        # SDK raises ClientError(429) which is NOT a subclass of ResourceExhausted
        # (google.api_core). Both include "resource_exhausted" in the message.
        if "resource_exhausted" in error_message or "quota" in error_message:
            raise LLMRateLimitError(
                f"Gemini rate limit exceeded. Retry after delay. | {error}"
            ) from error

        if isinstance(error, Unauthenticated):
            raise LLMAuthError(
                f"Gemini authentication failed. Check your API key. | {error}"
            ) from error

        if isinstance(error, ResourceExhausted):
            raise LLMRateLimitError(
                f"Gemini rate limit exceeded. Retry after delay. | {error}"
            ) from error

        if isinstance(error, DeadlineExceeded):
            raise LLMTimeoutError(
                f"Gemini request timed out after {self._timeout}s. | {error}"
            ) from error

        if isinstance(error, InvalidArgument):
            if "token" in error_message or "context" in error_message:
                raise LLMTokenLimitError(
                    f"Prompt exceeds Gemini model context window. | {error}"
                ) from error

        if isinstance(error, GoogleAPIError):
            raise LLMProviderError(
                f"Gemini API error occurred. | {error}"
            ) from error

        # Unknown error — still wrap in our hierarchy
        raise LLMProviderError(
            f"Unexpected error from Gemini provider. | {error}"
        ) from error