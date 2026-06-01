"""Multi-model Groq pool — drop-in BaseLLM replacement that routes across models via a bounded queue."""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import RateLimitError

from llm.contracts.base_llm import BaseLLM
from llm.providers.groq_provider import GroqProvider
from llm.providers.model_router import ModelRouter, CallRole
from llm.rate_limiter.rate_limit_tracker import get_tracker
from llm.models.llm_response import LLMResponse
from llm.exceptions.llm_exceptions import LLMRateLimitError, LLMProviderError, LLMTimeoutError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# max_tokens at or below this value → FAST call; above → STRONG call
_FAST_MAX_TOKENS_THRESHOLD: int = 512

# 3 workers is safe for groq's 30 rpm per-model limit
_NUM_WORKERS: int = 3

# queue full → immediate LLMRateLimitError
_QUEUE_MAX_SIZE: int = 50

# all models for which GroqProvider instances are pre-created
_ALL_POOL_MODELS: list[str] = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# models with reasoning/thinking mode — always disabled to avoid <think> tags breaking json parsers
_THINKING_MODELS: frozenset[str] = frozenset([
    "qwen/qwen3-32b",
])


@dataclass
class _RequestItem:
    """A single queued LLM call with its associated completion future."""

    messages: list[dict]
    kwargs: dict
    future: asyncio.Future


class GroqModelPool(BaseLLM):
    """Multi-model Groq pool that routes every call to the best available model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        router: Optional[ModelRouter] = None,
    ) -> None:
        """Create one GroqProvider per pool model and initialize the router; workers start lazily."""
        self._providers: dict[str, GroqProvider] = {}

        for model_id in _ALL_POOL_MODELS:
            self._providers[model_id] = GroqProvider(
                api_key=api_key,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            logger.debug("GroqModelPool: initialized provider for model=%s", model_id)

        self._router: ModelRouter = router or ModelRouter()
        self._tracker = get_tracker()

        # tracks most recently used model; starts with top-priority strong model
        self._active_model: str = settings.GROQ_MODEL_STRONG

        # queue + worker state — workers start lazily on first call
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._workers: list[asyncio.Task] = []
        self._workers_started: bool = False

        logger.info(
            "GroqModelPool initialized | models=%s | workers=%d | queue_max=%d",
            list(self._providers.keys()),
            _NUM_WORKERS,
            _QUEUE_MAX_SIZE,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        """Return model_id last successfully used; on init returns top-priority strong model."""
        return self._active_model

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation routed to the best available pool model."""
        messages = [{"role": "user", "content": prompt}]
        return await self._enqueue(messages, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation routed to the best available pool model."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")
        return await self._enqueue(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens using the active model's encoder; falls back to any provider."""
        provider = self._providers.get(self._active_model)
        if provider is None:
            provider = next(iter(self._providers.values()))
        return await provider.count_tokens(text)

    async def is_available(self) -> bool:
        """Return True if at least one pool model responds to a health check."""
        for model_id, provider in self._providers.items():
            try:
                if await provider.is_available():
                    logger.debug("Health check passed for model=%s", model_id)
                    return True
            except Exception as exc:
                logger.debug(
                    "Health check failed for model=%s | error=%s",
                    model_id, str(exc)[:100],
                )
        logger.warning("GroqModelPool health check failed — all models unreachable")
        return False

    async def _ensure_workers_started(self) -> None:
        """Lazily start worker tasks on first call (event loop may not exist at __init__)."""
        if self._workers_started:
            return

        self._workers_started = True
        for worker_id in range(_NUM_WORKERS):
            task = asyncio.create_task(
                self._worker(worker_id=worker_id),
                name=f"groq-pool-worker-{worker_id}",
            )
            self._workers.append(task)

        logger.info("GroqModelPool: started %d queue workers", _NUM_WORKERS)

    async def _enqueue(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Enqueue a request and suspend the caller until a worker resolves it."""
        await self._ensure_workers_started()

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        item = _RequestItem(messages=messages, kwargs=kwargs, future=future)

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning(
                "Request queue full — rejecting request | queue_size=%d/%d",
                self._queue.qsize(), _QUEUE_MAX_SIZE,
            )
            raise LLMRateLimitError(
                f"GroqModelPool request queue is full (max={_QUEUE_MAX_SIZE}). "
                "The system is overloaded — try again shortly."
            )

        logger.debug(
            "Request added to queue | queue_size=%d/%d",
            self._queue.qsize(), _QUEUE_MAX_SIZE,
        )
        return await future

    async def _worker(self, worker_id: int) -> None:
        """Persistent worker that pulls items from the queue and resolves their futures."""
        logger.debug("GroqModelPool worker %d started", worker_id)

        while True:
            item: _RequestItem = await self._queue.get()
            logger.debug(
                "Worker %d picked up request | queue_remaining=%d",
                worker_id, self._queue.qsize(),
            )
            try:
                response = await self._dispatch(item.messages, **item.kwargs)
                item.future.set_result(response)
                logger.debug("Worker %d request fulfilled | queue_remaining=%d", worker_id, self._queue.qsize())
            except Exception as exc:
                # surface to caller via future — workers must never crash
                if not item.future.done():
                    logger.warning(
                        "Worker %d dispatch failed | error=%s: %s — surfacing exception to caller",
                        worker_id, type(exc).__name__, str(exc)[:120],
                    )
                    item.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _dispatch(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Core dispatch loop — route, call, handle errors, retry or escalate."""
        role = self._detect_role(kwargs.get("max_tokens"))
        est_tokens = await self._estimate_tokens(messages, kwargs.get("max_tokens"))

        logger.debug("Routing started | role=%s | est_tokens=%d", role, est_tokens)

        # retry loop — each 429 or 404 puts that model in cooldown; router skips it next pass
        while True:
            model_id = await self._router.route(role=role, est_tokens=est_tokens)

            if model_id is None:
                raise LLMRateLimitError(
                    "All Groq pool models are rate-limited or exhausted. "
                    "Try again after the shortest cooldown expires."
                )

            logger.info(
                "Routing to model | model=%s | role=%s | est_tokens=%d",
                model_id, role, est_tokens,
            )

            try:
                response = await self._call_provider(model_id, messages, **kwargs)

                self._active_model = model_id

                logger.info(
                    "GroqModelPool dispatch succeeded | model=%s | role=%s | "
                    "tokens=%d | latency_ms=%.1f",
                    model_id, role, response.tokens_used, response.latency_ms,
                )
                return response

            except LLMRateLimitError as exc:
                retry_after = self._parse_retry_after(str(exc))
                logger.warning(
                    "429 on model=%s — cooldown=%ss | switching to next available model",
                    model_id, retry_after if retry_after is not None else 60,
                )
                await self._router.on_429(model_id, retry_after=retry_after)

            except LLMTimeoutError:
                # timeout on one model isn't fatal — cooldown and try next
                logger.warning(
                    "Timeout on model=%s — applying 60s cooldown and trying next model",
                    model_id,
                )
                await self._router.on_429(model_id, retry_after=60)

            except LLMProviderError as exc:
                error_msg = str(exc).lower()
                if "does not exist" in error_msg or "model_not_found" in error_msg:
                    # 404 — no account access; 24h cooldown and try next
                    logger.warning(
                        "model=%s returned 404 (not found / no access) — "
                        "excluding from pool for 24 hours. "
                        "Remove it from _ALL_POOL_MODELS to silence this warning.",
                        model_id,
                    )
                    await self._router.on_429(model_id, retry_after=86_400)

                elif "string_too_short" in error_msg or "at least 1 character" in error_msg:
                    # http 200 but empty completion — pydantic rejects; treat as transient
                    logger.warning(
                        "model=%s returned empty response — 30s cooldown | retrying with next model",
                        model_id,
                    )
                    await self._router.on_429(model_id, retry_after=30)

                elif "413" in error_msg or "request too large" in error_msg or (
                    "rate_limit_exceeded" in error_msg and "tokens per minute" in error_msg
                ):
                    # 413 — per-request token cap exceeded (model-specific); cooldown and switch
                    logger.warning(
                        "model=%s rejected request (413 payload too large) — "
                        "60s cooldown | switching to next model",
                        model_id,
                    )
                    await self._router.on_429(model_id, retry_after=60)

                else:
                    # auth or unexpected provider failure — propagate
                    raise

    async def _call_provider(
        self,
        model_id: str,
        messages: list[dict],
        **kwargs,
    ) -> LLMResponse:
        """Call a specific model's provider and update the rate limit tracker."""
        provider = self._providers[model_id]

        response, raw_headers = await self._call_with_headers(provider, messages, **kwargs)

        # update tracker from server-authoritative headers
        if raw_headers:
            await self._tracker.update_from_headers(model_id, raw_headers)

        await self._tracker.increment_daily(model_id, response.tokens_used)

        return response

    async def _call_with_headers(
        self,
        provider: GroqProvider,
        messages: list[dict],
        **kwargs,
    ) -> tuple[LLMResponse, dict[str, str]]:
        """Dispatch to the provider and extract response headers; falls back to chat() on sdk gap."""
        temperature = kwargs.get("temperature", provider._temperature)
        max_tokens  = kwargs.get("max_tokens",  provider._max_tokens)

        # disable thinking for models that emit <think> tags — uses both reasoning_format
        # and a /no_think system token to ensure suppression even if one is ignored
        extra_body: dict = {}
        if provider._model in _THINKING_MODELS:
            extra_body = {"reasoning_format": "hidden"}
            if messages and messages[0].get("role") == "system":
                messages = [
                    {**messages[0], "content": f"/no_think\n{messages[0]['content']}"},
                    *messages[1:],
                ]
            else:
                messages = [{"role": "system", "content": "/no_think"}, *messages]
            logger.debug(
                "Thinking mode suppressed | model=%s | method=reasoning_format_hidden+no_think",
                provider._model,
            )

        try:
            # with_raw_response exposes .parse() and .headers from the httpx response
            start_time = time.monotonic()
            raw = await provider._client.chat.completions.with_raw_response.create(
                model=provider._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            completion = raw.parse()

            headers: dict[str, str] = dict(raw.headers)

            llm_response = provider._parse_response(completion, latency_ms)
            return llm_response, headers

        except RateLimitError as exc:
            # re-raise as internal error type so the dispatch loop handles it
            raise LLMRateLimitError(
                f"Groq 429 on model {provider._model}: {str(exc)[:200]}"
            ) from exc

        except (AttributeError, TypeError):
            # with_raw_response unavailable on this sdk version — fall back without headers
            logger.warning(
                "Header extraction unavailable | model=%s | fallback=provider.chat() "
                "— rate limit headers missing, tracker accuracy reduced this call",
                provider._model,
            )
            response = await provider.chat(messages, **kwargs)
            return response, {}

        except Exception as exc:
            # translate via provider's handler — never make a second http call
            provider._handle_error(exc)
            raise

    @staticmethod
    def _detect_role(max_tokens: Optional[int]) -> CallRole:
        """Determine call role from max_tokens: <=512 FAST, else STRONG (None counts as STRONG)."""
        role: CallRole = "STRONG" if (max_tokens is None or max_tokens > _FAST_MAX_TOKENS_THRESHOLD) else "FAST"
        logger.debug("Role detected | role=%s | max_tokens=%s", role, max_tokens)
        return role

    async def _estimate_tokens(
        self,
        messages: list[dict],
        max_tokens: Optional[int],
    ) -> int:
        """Estimate total tokens (prompt + expected completion) for TPM/TPD headroom checks."""
        full_text = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )

        # any provider's encoder works — they share the tiktoken fallback
        provider = next(iter(self._providers.values()))
        prompt_tokens = await provider.count_tokens(full_text)

        completion_budget = max_tokens if max_tokens is not None else 512
        total = prompt_tokens + completion_budget
        logger.debug(
            "Token estimate | prompt=%d | completion_budget=%d | total=%d",
            prompt_tokens, completion_budget, total,
        )
        return total

    @staticmethod
    def _parse_retry_after(error_str: str) -> Optional[int]:
        """Extract Retry-After seconds from a 429 error message string; None if absent."""
        # pattern: "Please try again in 57.123s"
        match = re.search(r"try again in\s+([\d.]+)s", error_str, re.IGNORECASE)
        if match:
            try:
                return int(float(match.group(1))) + 1
            except ValueError:
                pass

        # pattern: "retry_after: 60"
        match = re.search(r"retry.?after[:\s]+(\d+)", error_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        return None
