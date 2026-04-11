"""
Model router for the Groq multi-model pool.

Design:
    ModelRouter is the single decision point for "which model should handle
    this call?". It knows two pools (FAST and STRONG), each with a fixed
    priority order. On every call it:

        1. Identifies the call role (FAST vs STRONG) from the max_tokens kwarg.
        2. Iterates the pool in priority order.
        3. Skips models that fail any of the four availability checks
           (cooldown, RPM headroom, TPM headroom, RPD budget, TPD budget).
        4. Returns the first model_id that passes all checks.
        5. Returns None if no model is available in the primary pool.
           GroqModelPool then tries the other pool before giving up.

    Role auto-detection (zero changes to RAG variant code):
        max_tokens ≤ 512  → FAST   (eval, rewrite, classify, completeness checks)
        max_tokens > 512  → STRONG (final synthesis, multi-hop generation)
        max_tokens is None → STRONG (generate() default — large / open-ended)

    Pool definitions:
        FAST   : ["llama-3.1-8b-instant", "qwen/qwen3-32b"]
        STRONG : ["moonshotai/kimi-k2", "llama-3.3-70b-versatile",
                  "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct"]

        qwen3-32b appears in both pools — its budget is tracked once
        (single ModelRateLimitState keyed by model_id in RateLimitTracker).

    Headroom thresholds:
        Minute windows  : _MIN_RPM_HEADROOM (default 2) and _MIN_TPM_HEADROOM
                          (default 500). These prevent routing a model that has
                          only 1 request left in its minute window — leaving a
                          tiny buffer avoids last-second 429s.
        Daily ceilings  : derived from MODEL_RATE_LIMITS.rpd / tpd minus a
                          small guard margin (_RPD_GUARD, _TPD_GUARD).

    on_429(model_id, retry_after):
        Called by GroqModelPool when a model returns HTTP 429. Delegates to
        RateLimitTracker.on_429() so the model enters cooldown immediately.

Chain of Responsibility:
    GroqModelPool.chat() / .generate() calls ModelRouter.route(role, est_tokens)
    → returns model_id → GroqModelPool dispatches to that model's GroqProvider
    → on success, GroqModelPool calls tracker.update_from_headers() and
    tracker.increment_daily() → on 429, calls ModelRouter.on_429().

Dependencies:
    asyncio, typing (stdlib), llm.rate_limiter.rate_limit_tracker,
    llm.rate_limiter.model_limits, utils.logger.
"""

from typing import Literal, Optional

from llm.rate_limiter.rate_limit_tracker import RateLimitTracker, get_tracker
from llm.rate_limiter.model_limits import MODEL_RATE_LIMITS
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Call role — determined from max_tokens kwarg by GroqModelPool
# ---------------------------------------------------------------------------
CallRole = Literal["FAST", "STRONG"]

# ---------------------------------------------------------------------------
# Pool definitions — priority order matters (first = preferred)
# ---------------------------------------------------------------------------

# FAST pool: high-volume, low-latency tasks (eval, rewrite, classify)
# Llama 3.1 8B is the workhorse; Qwen3 32B is the capacity overflow fallback
_FAST_POOL: list[str] = [
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]

# STRONG pool: final answer generation, complex reasoning
# Priority: Kimi K2 (best quality) → Llama 3.3 70B → Qwen3 32B → Llama 4 Scout
_STRONG_POOL: list[str] = [
    "moonshotai/kimi-k2",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# ---------------------------------------------------------------------------
# Headroom constants — buffer to avoid last-slot 429s
# ---------------------------------------------------------------------------

# Minimum RPM remaining before a model is considered "at capacity" for routing.
# Routing to a model with 1 remaining request risks a 429 if another coroutine
# fires concurrently.
_MIN_RPM_HEADROOM: int = 2

# Minimum TPM remaining before a model is considered "at token capacity".
# 500 tokens is smaller than even the smallest eval prompt, so anything below
# this means the window is effectively exhausted.
_MIN_TPM_HEADROOM: int = 500

# Daily guard margins — we stop routing to a model when its locally-tracked
# daily usage exceeds (rpd - _RPD_GUARD) or (tpd - _TPD_GUARD).
# Keeps a buffer for in-flight requests that haven't been accounted for yet.
_RPD_GUARD: int = 5
_TPD_GUARD: int = 2_000


class ModelRouter:
    """Selects the best available Groq model for each LLM call.

    All route decisions are async because get_state() acquires an asyncio lock
    inside RateLimitTracker.

    Attributes:
        _tracker: Shared RateLimitTracker singleton.
    """

    def __init__(self, tracker: Optional[RateLimitTracker] = None) -> None:
        """Initialize the router with a shared (or injected) RateLimitTracker.

        Args:
            tracker: RateLimitTracker instance to use. Defaults to the
                     module-level singleton from get_tracker(). Pass an
                     explicit instance in tests to avoid shared state.
        """
        self._tracker: RateLimitTracker = tracker or get_tracker()

    # Public API

    async def route(
        self,
        role: CallRole,
        est_tokens: int = 1_000,
    ) -> Optional[str]:
        """Return the best available model_id for the given call role.

        Tries the primary pool first in priority order, then the secondary pool
        as a cross-pool overflow (e.g. FAST-role call routes to STRONG pool
        when FAST pool is exhausted). Returns None only when ALL models in both
        pools are unavailable — GroqModelPool should raise LLMRateLimitError
        in that case.

        Args:
            role:       "FAST" or "STRONG" — determines which pool is tried first.
            est_tokens: Estimated total tokens for this call (prompt + expected
                        completion). Used to check TPM and TPD headroom.
                        Defaults to 1,000 as a conservative estimate.

        Returns:
            model_id string of the selected model, or None if no model is available.
        """
        primary_pool = _FAST_POOL if role == "FAST" else _STRONG_POOL
        secondary_pool = _STRONG_POOL if role == "FAST" else _FAST_POOL

        # Try primary pool in priority order
        model = await self._pick_from_pool(primary_pool, est_tokens, pool_label=role)
        if model:
            return model

        # Cross-pool overflow — try the other pool when primary is exhausted
        other_label: CallRole = "STRONG" if role == "FAST" else "FAST"
        logger.warning(
            "Primary %s pool exhausted — attempting cross-pool overflow to %s pool",
            role, other_label,
        )
        return await self._pick_from_pool(secondary_pool, est_tokens, pool_label=other_label)

    async def on_429(
        self,
        model_id: str,
        retry_after: Optional[int] = None,
    ) -> None:
        """Record a 429 response for a model and put it in cooldown.

        Delegates to RateLimitTracker.on_429() which sets in_cooldown=True and
        resets remaining_rpm / remaining_tpm to 0 so the model is immediately
        skipped by future route() calls.

        Args:
            model_id:    Exact Groq model ID that returned 429.
            retry_after: Cooldown duration in seconds from the Retry-After
                         response header. Falls back to tracker default (60s).
        """
        logger.warning(
            "429 received for model=%s | retry_after=%s",
            model_id, f"{retry_after}s" if retry_after else "default",
        )
        await self._tracker.on_429(model_id, cooldown_seconds=retry_after)

    # Private helpers

    async def _pick_from_pool(
        self,
        pool: list[str],
        est_tokens: int,
        pool_label: str,
    ) -> Optional[str]:
        """Iterate a pool in priority order and return the first available model.

        Args:
            pool:       Ordered list of model_id strings to try.
            est_tokens: Estimated total tokens for the call.
            pool_label: Pool name string used only for logging.

        Returns:
            model_id of the first available model, or None if all are skipped.
        """
        for model_id in pool:
            if await self._is_model_available(model_id, est_tokens):
                logger.debug(
                    "Routing to model=%s (pool=%s, est_tokens=%d)",
                    model_id, pool_label, est_tokens,
                )
                return model_id

            logger.debug(
                "Skipping model=%s (pool=%s) — availability check failed",
                model_id, pool_label,
            )

        logger.warning("No available model in pool=%s", pool_label)
        return None

    async def _is_model_available(self, model_id: str, est_tokens: int) -> bool:
        """Check whether a model can accept a call right now.

        Applies four checks in order of cheapness:

            1. 429 cooldown — skip immediately if in active cooldown.
            2. RPM headroom — skip if remaining_rpm is known and below threshold.
            3. TPM headroom — skip if remaining_tpm is known and below est_tokens.
            4. Daily RPD budget — skip if locally-tracked daily requests exceed limit.
            5. Daily TPD budget — skip if locally-tracked daily tokens exceed limit.

        Checks 2 and 3 use server-authoritative minute values from headers.
        When remaining_rpm / remaining_tpm is None (first call, no header yet),
        those checks are skipped — we optimistically allow the call through
        and let the server enforce.

        Args:
            model_id:   Exact Groq model ID.
            est_tokens: Estimated total tokens for this call.

        Returns:
            True if the model passes all availability checks.
        """
        state = await self._tracker.get_state(model_id)

        # Check 1: Active 429 cooldown
        if state.in_cooldown and not state.cooldown_expired():
            logger.debug(
                "model=%s skipped — in 429 cooldown until %s",
                model_id,
                state.cooldown_until.strftime("%H:%M:%S") if state.cooldown_until else "?",
            )
            return False

        # If minute window has reset, the server-side remaining values are stale.
        # We treat them as unknown rather than falsely blocking the model.
        minute_fresh = state.is_minute_window_fresh()

        # Check 2: RPM headroom (only when we have a fresh, authoritative value)
        if minute_fresh and state.remaining_rpm is not None:
            if state.remaining_rpm < _MIN_RPM_HEADROOM:
                logger.debug(
                    "model=%s skipped — remaining_rpm=%d < threshold=%d",
                    model_id, state.remaining_rpm, _MIN_RPM_HEADROOM,
                )
                return False

        # Check 3: TPM headroom (only when we have a fresh, authoritative value)
        if minute_fresh and state.remaining_tpm is not None:
            if state.remaining_tpm < max(est_tokens, _MIN_TPM_HEADROOM):
                logger.debug(
                    "model=%s skipped — remaining_tpm=%d < needed=%d",
                    model_id, state.remaining_tpm, max(est_tokens, _MIN_TPM_HEADROOM),
                )
                return False

        # Check 4: Daily RPD budget (locally tracked)
        limits = MODEL_RATE_LIMITS.get(model_id)
        if limits is not None:
            rpd_ceiling = limits.rpd - _RPD_GUARD
            if state.used_rpd >= rpd_ceiling:
                logger.warning(
                    "model=%s skipped — daily request budget exhausted "
                    "(used_rpd=%d >= ceiling=%d)",
                    model_id, state.used_rpd, rpd_ceiling,
                )
                return False

            # Check 5: Daily TPD budget (locally tracked)
            tpd_ceiling = limits.tpd - _TPD_GUARD
            if state.used_tpd + est_tokens >= tpd_ceiling:
                logger.warning(
                    "model=%s skipped — daily token budget exhausted "
                    "(used_tpd=%d + est_tokens=%d >= ceiling=%d)",
                    model_id, state.used_tpd, est_tokens, tpd_ceiling,
                )
                return False

        return True
