"""Google Gemini implementation of BaseLLM with automatic retry on 429."""

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

# maps gemini finish_reason enum names to normalized lowercase strings
_GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "safety",
    "RECITATION": "recitation",
    "OTHER": "other",
    "FINISH_REASON_UNSPECIFIED": "unknown",
}

# gemini 429 messages include "Please retry in X.Xs" — parse and honor it
_RETRY_AFTER_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_DEFAULT_RETRY_DELAY_S = 65.0
_MAX_RATE_LIMIT_RETRIES = 2


def _parse_retry_after(error_message: str) -> float:
    """Extract retry-after seconds from a Gemini 429 message; adds 2s buffer."""
    match = _RETRY_AFTER_PATTERN.search(error_message)
    if match:
        return float(match.group(1)) + 2.0
    return _DEFAULT_RETRY_DELAY_S


class GeminiProvider(BaseLLM):
    """Google Gemini implementation of BaseLLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the Gemini client and generation config."""
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

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation."""
        contents = self._build_contents(prompt)
        return await self._call_api(contents, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation with OpenAI-format message input."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        contents = self._convert_messages(messages)
        return await self._call_api(contents, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens via Gemini's native API; falls back to char-based estimate on failure."""
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
            # rough fallback: 1 token ~ 4 chars
            return len(text) // 4

    async def is_available(self) -> bool:
        """Health check — send a minimal request to verify API reachability."""
        try:
            await self._client.aio.models.generate_content(
                model=self._model,
                contents="ping",
            )
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed | error=%s", str(exc))
            return False

    async def _call_api(self, contents: list, **kwargs) -> LLMResponse:
        """Execute the Gemini API call with retry, timing, and error handling."""
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        response_mime_type = kwargs.get("response_mime_type")
        # thinking budget defaults to 0 — saves 8-11s per call
        thinking_budget = kwargs.get("thinking_budget", 0)

        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            # afc adds ~200ms overhead and is irrelevant for rag generation
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

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
                # retries exhausted — translate and raise
                self._handle_error(exc)

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
                logger.error(
                    "Gemini API call failed | latency_ms=%.1f | error=%s",
                    latency_ms,
                    str(exc),
                )
                self._handle_error(exc)
                # _handle_error always raises; satisfies the type checker
                raise LLMProviderError(
                    f"Unhandled error in Gemini provider. | {exc}"
                )

        # unreachable in practice; satisfies the type checker
        raise LLMProviderError(f"Gemini call failed after retries. | {last_exc}")

    def _build_contents(self, prompt: str) -> list:
        """Wrap a plain prompt string into the Gemini contents format."""
        return [{"role": "user", "parts": [{"text": prompt}]}]

    def _convert_messages(self, messages: List[dict]) -> list:
        """Convert OpenAI message format to Gemini contents format."""
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
        """Parse a raw Gemini API response into a standard LLMResponse."""
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        tokens_used = usage.total_token_count if usage else 0

        finish_reason = "unknown"
        if response.candidates:
            raw_reason = response.candidates[0].finish_reason
            # handle both enum objects (have .name) and plain strings
            reason_str = (
                raw_reason.name
                if hasattr(raw_reason, "name")
                else str(raw_reason)
            )
            finish_reason = _GEMINI_FINISH_REASON_MAP.get(
                reason_str, reason_str.lower()
            )

        # response.text raises ValueError when candidates are blocked by safety filters
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
        """Translate Gemini SDK exceptions into the LLMError hierarchy."""
        error_message = str(error).lower()

        # string check first — 400 errors with invalid api key arrive as generic exceptions
        if "api key not valid" in error_message or "api_key_invalid" in error_message:
            raise LLMAuthError(
                f"Gemini authentication failed. Check your API key. | {error}"
            ) from error

        # string check before isinstance — new sdk raises ClientError(429), not ResourceExhausted
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

        # catch-all — still wrap in our hierarchy
        raise LLMProviderError(
            f"Unexpected error from Gemini provider. | {error}"
        ) from error
