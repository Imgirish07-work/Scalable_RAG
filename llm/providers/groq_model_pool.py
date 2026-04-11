"""
Multi-model Groq pool — drop-in BaseLLM replacement for single-model GroqProvider.

Design:
    GroqModelPool implements BaseLLM and manages a fixed set of GroqProvider
    instances (one per model). It is a transparent replacement for a plain
    GroqProvider — the RAG pipeline, BaseRAG, and agents all interact with
    BaseLLM and never know they are talking to a pool.

    All generate() / chat() calls are funnelled through a bounded asyncio.Queue.
    A fixed pool of _NUM_WORKERS worker coroutines drain the queue one item at
    a time. This caps in-flight concurrency and prevents the burst-429 race
    condition where multiple callers simultaneously route to the same model
    before any response headers have been received.

    On every dispatched call the pool:

        1. Detects the call role (FAST or STRONG) from the max_tokens kwarg:
               max_tokens ≤ 512  → FAST   (eval, rewrite, classify)
               max_tokens > 512  → STRONG (final synthesis)
               max_tokens is None → STRONG (open-ended generation)

           This is zero-touch for all existing RAG variants — they already pass
           explicit small max_tokens for lightweight calls and omit / use large
           values for generate(). No changes to SimpleRAG, CorrectiveRAG, or
           ChainRAG are needed.

        2. Asks ModelRouter to pick the best available model.

        3. Dispatches the call to that model's GroqProvider._call_api() using
           the raw openai client so we can access the response headers.

        4. Parses x-ratelimit-* headers from the raw HTTP response and calls
           tracker.update_from_headers() + tracker.increment_daily().

        5. On HTTP 429 (LLMRateLimitError):
               a. Calls router.on_429(model_id, retry_after) to start cooldown.
               b. Immediately re-routes to the next available model and retries.
               c. If ALL models are exhausted, raises LLMRateLimitError to the caller.

        6. On HTTP 404 "model not found" (LLMProviderError):
               Puts the inaccessible model in a 24-hour cooldown so it is
               permanently skipped for the session, then retries with the next
               available model. Does NOT propagate to BaseRAG — the pool stays up.

    Header extraction:
        The OpenAI SDK does not expose raw headers on the parsed response object.
        We use the undocumented `.response` / `._response` attribute on
        AsyncCompletions to capture headers. If that attribute is absent (SDK
        version changes), we degrade gracefully — the tracker simply has no
        new header data and falls back to locally-tracked daily counters.

    Token estimation for routing:
        Before dispatching, we estimate the total tokens for the call using
        tiktoken on the message content. This estimate feeds ModelRouter's TPM
        and TPD headroom checks. It is never exact (Groq counts slightly
        differently) but is accurate enough for capacity planning.

    model_name / provider_name properties:
        Return the *last successfully used* model to give callers a meaningful
        value for logging. On init they return the top-priority STRONG model.

Chain of Responsibility:
    LLMFactory.create_groq_pool() instantiates GroqModelPool →
    returned as BaseLLM to RAGPipeline → BaseRAG calls generate() / chat() →
    GroqModelPool enqueues request → worker pulls from queue → _dispatch() →
    ModelRouter → GroqProvider → parses headers → updates RateLimitTracker.

Dependencies:
    asyncio, re, time, typing (stdlib), dataclasses (stdlib),
    openai (RateLimitError), llm.contracts.base_llm, llm.providers.groq_provider,
    llm.providers.model_router, llm.rate_limiter.rate_limit_tracker,
    llm.models.llm_response, llm.exceptions.llm_exceptions,
    config.settings, utils.logger.
"""

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
from llm.exceptions.llm_exceptions import LLMRateLimitError, LLMProviderError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Pool configuration constants

# max_tokens at or below this value → FAST call; above → STRONG call
_FAST_MAX_TOKENS_THRESHOLD: int = 512

# Number of concurrent worker coroutines draining the request queue.
# Higher values increase throughput but also increase burst risk.
# 3 is a safe default for Groq's 30 RPM per-model limit.
_NUM_WORKERS: int = 3

# Maximum number of requests that can wait in the queue.
# Callers receive LLMRateLimitError immediately when the queue is full.
_QUEUE_MAX_SIZE: int = 50

# All models for which GroqProvider instances are pre-created.
# Mirrors the union of FAST + STRONG pool lists in ModelRouter.
# To add a model: register it in model_limits.py AND add it here.
_ALL_POOL_MODELS: list[str] = [
    "llama-3.1-8b-instant",                       # FAST primary
    "moonshotai/kimi-k2-instruct",                # STRONG priority-1 (unlisted; 404-safe)
    "llama-3.3-70b-versatile",                    # STRONG priority-2
    "qwen/qwen3-32b",                             # FAST overflow / STRONG priority-3
    "meta-llama/llama-4-scout-17b-16e-instruct",  # STRONG priority-4
]

# Queue item dataclass


@dataclass
class _RequestItem:
    """A single queued LLM call with its associated completion future.

    Attributes:
        messages: OpenAI-formatted message list.
        kwargs:   Per-call overrides (temperature, max_tokens, etc.).
        future:   asyncio.Future that the worker resolves when the call
                  completes. The caller awaits this future.
    """

    messages: list[dict]
    kwargs: dict
    future: asyncio.Future  # always passed explicitly by _enqueue; no default


# GroqModelPool

class GroqModelPool(BaseLLM):
    """Multi-model Groq pool that routes every call to the best available model.

    Drop-in replacement for GroqProvider — implements BaseLLM and is returned
    from LLMFactory. All RAG variants continue to call generate() / chat() on
    a BaseLLM reference without any changes.

    Attributes:
        _providers:         Dict mapping model_id → GroqProvider instance.
        _router:            ModelRouter for pool selection decisions.
        _tracker:           Shared RateLimitTracker for state reads/writes.
        _active_model:      model_id of the last successfully dispatched model.
        _queue:             Bounded asyncio.Queue holding pending _RequestItems.
        _workers:           List of asyncio.Tasks running _worker() coroutines.
        _workers_started:   Flag — True once worker tasks have been created.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        router: Optional[ModelRouter] = None,
    ) -> None:
        """Create one GroqProvider per pool model and initialize the router.

        Worker tasks are NOT started here — they are created lazily on the
        first call to generate() / chat() so that asyncio.create_task() is
        always called from within a running event loop.

        Args:
            api_key:     Groq API key. Falls back to settings.groq_api_key.
            temperature: Default sampling temperature. Falls back to settings.
            max_tokens:  Default max output tokens. Falls back to settings.
            timeout:     Request timeout in seconds. Falls back to settings.GROQ_TIMEOUT.
            router:      ModelRouter instance. Defaults to a new ModelRouter using
                         the shared tracker singleton. Pass explicitly in tests.
        """
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

        # Tracks the most recently used model for model_name / provider_name
        # properties. Starts with the top-priority STRONG model.
        self._active_model: str = settings.GROQ_MODEL_STRONG

        # Queue + worker state — workers are started lazily on first call
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._workers: list[asyncio.Task] = []
        self._workers_started: bool = False

        logger.info(
            "GroqModelPool initialized | models=%s | workers=%d | queue_max=%d",
            list(self._providers.keys()),
            _NUM_WORKERS,
            _QUEUE_MAX_SIZE,
        )

    # BaseLLM properties

    @property
    def provider_name(self) -> str:
        """Returns 'groq' — the pool always talks to Groq."""
        return "groq"

    @property
    def model_name(self) -> str:
        """Returns the model_id last successfully used by the pool.

        Useful for logging in BaseRAG and LLMResponse. On the first call
        (before any dispatch has succeeded) returns the top-priority STRONG
        model name.
        """
        return self._active_model

    # BaseLLM abstract method implementations

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Single-turn text generation routed to the best available pool model.

        Auto-detects role from kwargs['max_tokens']:
            ≤ 512 → FAST pool
            > 512 or absent → STRONG pool

        Args:
            prompt:   Input text prompt.
            **kwargs: Per-call overrides (temperature, max_tokens).

        Returns:
            LLMResponse from the selected model.

        Raises:
            LLMRateLimitError: If all models in both pools are exhausted, or
                               the request queue is full.
            LLMProviderError:  For any non-recoverable provider error.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self._enqueue(messages, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        """Multi-turn conversation routed to the best available pool model.

        Auto-detects role from kwargs['max_tokens'] using the same logic as
        generate().

        Args:
            messages: List of dicts [{"role": "...", "content": "..."}].
            **kwargs: Per-call overrides (temperature, max_tokens).

        Returns:
            LLMResponse from the selected model.

        Raises:
            ValueError:        If messages list is empty.
            LLMRateLimitError: If all models in both pools are exhausted, or
                               the request queue is full.
            LLMProviderError:  For any non-recoverable provider error.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")
        return await self._enqueue(messages, **kwargs)

    async def count_tokens(self, text: str) -> int:
        """Count tokens for the given text using the active model's encoder.

        Delegates to the provider for the currently active model. Falls back
        to the strong model's provider if no call has been dispatched yet.

        Args:
            text: Input text to count tokens for.

        Returns:
            Token count as integer. Returns 0 for empty input.
        """
        provider = self._providers.get(self._active_model)
        if provider is None:
            # Fallback to first available provider
            provider = next(iter(self._providers.values()))
        return await provider.count_tokens(text)

    async def is_available(self) -> bool:
        """Return True if at least one pool model responds to a health check.

        Tries each provider in order and returns True on the first success.
        Only returns False if every model in the pool fails the health check.

        Returns:
            True if any model is reachable, False if all fail.
        """
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

    # Queue + worker internals

    async def _ensure_workers_started(self) -> None:
        """Lazily start the worker tasks on the first LLM call.

        We cannot create asyncio.Tasks in __init__ because the event loop may
        not yet be running. Starting workers on first use guarantees
        asyncio.create_task() is always called from within a running loop.
        """
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
        """Enqueue a request and wait for a worker to fulfill it.

        Creates an asyncio.Future, wraps it in a _RequestItem, puts the item
        on the bounded queue, and suspends the caller until a worker resolves
        the future. This is the single entry point for all LLM calls in the
        pool — it replaces a direct call to _dispatch().

        Args:
            messages: OpenAI-formatted message list.
            **kwargs: Per-call overrides.

        Returns:
            LLMResponse from the dispatched model.

        Raises:
            LLMRateLimitError: Queue full (system overloaded) or all models
                               exhausted (propagated from worker).
            LLMProviderError:  Non-recoverable provider error (propagated from
                               worker).
        """
        await self._ensure_workers_started()

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        item = _RequestItem(messages=messages, kwargs=kwargs, future=future)

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise LLMRateLimitError(
                f"GroqModelPool request queue is full (max={_QUEUE_MAX_SIZE}). "
                "The system is overloaded — try again shortly."
            )

        return await future

    async def _worker(self, worker_id: int) -> None:
        """Persistent worker coroutine that processes requests from the queue.

        Each worker pulls one _RequestItem at a time, calls _dispatch() to
        route and execute the call, and resolves the item's future with
        either the result or the exception. Workers run indefinitely until
        the asyncio task is cancelled (e.g., at process shutdown).

        By capping in-flight calls to _NUM_WORKERS, we prevent the burst-429
        race condition where many concurrent callers all see stale remaining_rpm
        and all route to the same model simultaneously.

        Args:
            worker_id: Integer identifier for this worker (used in logging only).
        """
        logger.debug("GroqModelPool worker %d started", worker_id)

        while True:
            item: _RequestItem = await self._queue.get()
            try:
                response = await self._dispatch(item.messages, **item.kwargs)
                item.future.set_result(response)
            except Exception as exc:
                # Surface all exceptions to the caller via the future — workers
                # must never crash; they catch everything and keep running.
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._queue.task_done()

    # Private dispatch loop

    async def _dispatch(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Core dispatch loop — route, call, handle errors, retry or escalate.

        Determines the call role, asks the router for a model, dispatches to
        that model's provider, and updates the tracker. On 429 puts the model
        in cooldown and retries with the next available model. On HTTP 404
        "model not found" puts the model in a 24-hour cooldown and retries
        rather than propagating an error that would bring down the whole pool.
        Raises LLMRateLimitError only when the router returns None (all pools empty).

        Args:
            messages: OpenAI-formatted message list.
            **kwargs: Per-call overrides forwarded to the provider.

        Returns:
            LLMResponse from the first model that succeeds.

        Raises:
            LLMRateLimitError: If all models are in cooldown / exhausted.
            LLMProviderError:  For non-rate-limit, non-404 provider errors.
        """
        role = self._detect_role(kwargs.get("max_tokens"))
        est_tokens = await self._estimate_tokens(messages, kwargs.get("max_tokens"))

        # Retry loop: each 429 or 404 eliminates one model via cooldown and
        # the router automatically skips it on the next iteration.
        # The loop terminates when router.route() returns None (all pools empty).
        while True:
            model_id = await self._router.route(role=role, est_tokens=est_tokens)

            if model_id is None:
                # All models exhausted — propagate as rate limit error
                raise LLMRateLimitError(
                    "All Groq pool models are rate-limited or exhausted. "
                    "Try again after the shortest cooldown expires."
                )

            try:
                response = await self._call_provider(model_id, messages, **kwargs)

                # Record the model we just used successfully
                self._active_model = model_id

                logger.info(
                    "GroqModelPool dispatch succeeded | model=%s | role=%s | "
                    "tokens=%d | latency_ms=%.1f",
                    model_id, role, response.tokens_used, response.latency_ms,
                )
                return response

            except LLMRateLimitError as exc:
                # HTTP 429 — put model in cooldown and retry with the next model
                retry_after = self._parse_retry_after(str(exc))
                await self._router.on_429(model_id, retry_after=retry_after)
                # Loop back — router will skip the cooled-down model

            except LLMProviderError as exc:
                error_msg = str(exc).lower()
                if "does not exist" in error_msg or "model_not_found" in error_msg:
                    # HTTP 404 — this account does not have access to the model.
                    # Exclude it for the rest of the session (24-hour cooldown)
                    # and try the next model. Do NOT propagate — the pool stays up.
                    logger.warning(
                        "model=%s returned 404 (not found / no access) — "
                        "excluding from pool for 24 hours. "
                        "Remove it from _ALL_POOL_MODELS to silence this warning.",
                        model_id,
                    )
                    await self._router.on_429(model_id, retry_after=86_400)
                    # Loop back — router will skip the excluded model
                else:
                    # Auth errors, token limit exceeded, timeout, etc. —
                    # not recoverable by switching models; propagate immediately.
                    raise

            # All other exceptions propagate immediately — not model-specific

    # Provider call + header extraction

    async def _call_provider(
        self,
        model_id: str,
        messages: list[dict],
        **kwargs,
    ) -> LLMResponse:
        """Call a specific model's provider and update the rate limit tracker.

        Uses the provider's underlying _call_api() method directly so we can
        intercept the raw HTTP response for header extraction. Falls back to
        the public chat() method when direct access is unavailable.

        After a successful call:
            - Extracts x-ratelimit-* headers from the raw HTTP response.
            - Calls tracker.update_from_headers() with those headers.
            - Calls tracker.increment_daily() with the response's token count.

        Args:
            model_id: Exact Groq model ID to dispatch to.
            messages: OpenAI-formatted message list.
            **kwargs: Per-call overrides forwarded to the provider.

        Returns:
            LLMResponse from the provider.

        Raises:
            LLMRateLimitError: On HTTP 429 from Groq.
            Any LLMError subclass: On other provider-side failures.
        """
        provider = self._providers[model_id]

        # Use the internal _call_api and capture the raw response with headers.
        # We monkey-patch around the SDK's response parsing to get the raw object.
        response, raw_headers = await self._call_with_headers(provider, messages, **kwargs)

        # Update rate limit tracker from server-authoritative headers
        if raw_headers:
            await self._tracker.update_from_headers(model_id, raw_headers)

        # Update daily local counters
        await self._tracker.increment_daily(model_id, response.tokens_used)

        return response

    async def _call_with_headers(
        self,
        provider: GroqProvider,
        messages: list[dict],
        **kwargs,
    ) -> tuple[LLMResponse, dict[str, str]]:
        """Dispatch to the provider and extract response headers.

        The OpenAI Python SDK does not directly expose response headers on the
        returned completion object. We use the `with_raw_response` context on
        the AsyncCompletions resource to capture the raw httpx response,
        which carries the x-ratelimit-* headers.

        If `with_raw_response` is unavailable (SDK version gap), we fall back
        to the normal .chat() call and return an empty headers dict, degrading
        gracefully without crashing.

        Args:
            provider: GroqProvider instance to call.
            messages: OpenAI-formatted message list.
            **kwargs: Per-call overrides.

        Returns:
            Tuple of (LLMResponse, headers_dict). headers_dict is empty {}
            if header extraction failed.

        Raises:
            LLMRateLimitError: On HTTP 429.
            Any LLMError subclass: On other failures.
        """
        temperature = kwargs.get("temperature", provider._temperature)
        max_tokens  = kwargs.get("max_tokens",  provider._max_tokens)

        try:
            # with_raw_response returns an object whose .parse() gives the
            # normal SDK response and whose .headers gives the HTTP headers.
            start_time = time.monotonic()
            raw = await provider._client.chat.completions.with_raw_response.create(
                model=provider._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            # Parse into the normal completion object
            completion = raw.parse()

            # Convert httpx headers to a plain lowercase dict
            headers: dict[str, str] = dict(raw.headers)

            llm_response = provider._parse_response(completion, latency_ms)
            return llm_response, headers

        except RateLimitError as exc:
            # Re-raise as our internal error type so the dispatch loop handles it
            raise LLMRateLimitError(
                f"Groq 429 on model {provider._model}: {str(exc)[:200]}"
            ) from exc

        except (AttributeError, TypeError):
            # with_raw_response not available on this SDK version — fall back to
            # a normal chat() call; headers will not be extracted this time.
            logger.debug(
                "with_raw_response unavailable for model=%s; "
                "falling back to normal chat() — headers will not be extracted",
                provider._model,
            )
            response = await provider.chat(messages, **kwargs)
            return response, {}

        except Exception as exc:
            # All other SDK exceptions (auth, timeout, token limit, API error).
            # Translate via the provider's error handler — never make a second
            # HTTP call here, as the same error would just repeat.
            provider._handle_error(exc)
            raise  # Unreachable — _handle_error always raises; satisfies type checker

    # Static helpers

    @staticmethod
    def _detect_role(max_tokens: Optional[int]) -> CallRole:
        """Determine the call role from the max_tokens parameter.

        All existing RAG variant calls already pass explicit small max_tokens
        for lightweight operations, so this detection requires zero changes
        to SimpleRAG, CorrectiveRAG, or ChainRAG.

        Mapping:
            max_tokens is None → STRONG (open-ended generate())
            max_tokens > 512   → STRONG (large synthesis)
            max_tokens ≤ 512   → FAST   (eval, rewrite, classify)

        Args:
            max_tokens: The max_tokens kwarg value, or None if not provided.

        Returns:
            "FAST" or "STRONG".
        """
        if max_tokens is None or max_tokens > _FAST_MAX_TOKENS_THRESHOLD:
            return "STRONG"
        return "FAST"

    async def _estimate_tokens(
        self,
        messages: list[dict],
        max_tokens: Optional[int],
    ) -> int:
        """Estimate total tokens for a call (prompt + expected completion).

        Used by ModelRouter to check TPM/TPD headroom before routing. We count
        the prompt tokens via tiktoken and add max_tokens (or a conservative
        default) as the completion budget.

        Args:
            messages:   OpenAI-formatted message list.
            max_tokens: Expected output token budget from the caller.

        Returns:
            Estimated total tokens as an integer.
        """
        # Concatenate all message content for token counting
        full_text = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )

        # Use any provider's encoder — they all use the same tiktoken fallback
        provider = next(iter(self._providers.values()))
        prompt_tokens = await provider.count_tokens(full_text)

        # Add expected completion size; default to 512 if not specified
        completion_budget = max_tokens if max_tokens is not None else 512
        return prompt_tokens + completion_budget

    @staticmethod
    def _parse_retry_after(error_str: str) -> Optional[int]:
        """Extract Retry-After seconds from a 429 error message string.

        Groq embeds the retry-after duration in the error message body when
        returning 429. We do a simple string scan for common patterns.
        Returns None if no value is found — the tracker will use its default.

        Patterns checked (case-insensitive):
            "retry after Xs"      → X seconds
            "retry_after: X"      → X seconds

        Args:
            error_str: String representation of the LLMRateLimitError.

        Returns:
            Retry-After duration in seconds, or None.
        """
        # Pattern: "Please try again in 57.123s"
        match = re.search(r"try again in\s+([\d.]+)s", error_str, re.IGNORECASE)
        if match:
            try:
                return int(float(match.group(1))) + 1  # round up
            except ValueError:
                pass

        # Pattern: "retry_after: 60"
        match = re.search(r"retry.?after[:\s]+(\d+)", error_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        return None
