"""Model router for the Groq multi-model pool — selects best available model from FAST/STRONG pools."""

from typing import Literal, Optional

from llm.rate_limiter.rate_limit_tracker import RateLimitTracker, get_tracker
from llm.rate_limiter.model_limits import MODEL_RATE_LIMITS
from utils.logger import get_logger

logger = get_logger(__name__)

# call role determined from max_tokens by GroqModelPool
CallRole = Literal["FAST", "STRONG"]

# fast pool: high-volume, low-latency tasks; priority order matters
_FAST_POOL: list[str] = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

# strong pool: final answer generation, complex reasoning
# kimi-k2-instruct is unlisted on groq — 404 if no access, _dispatch skips it for 24h
_STRONG_POOL: list[str] = [
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# headroom buffers to avoid last-slot 429s
_MIN_RPM_HEADROOM: int = 2
_MIN_TPM_HEADROOM: int = 500

# daily guard margins — stop routing when used > limit - guard
_RPD_GUARD: int = 5
_TPD_GUARD: int = 2_000


class ModelRouter:
    """Selects the best available Groq model for each LLM call."""

    def __init__(self, tracker: Optional[RateLimitTracker] = None) -> None:
        """Initialize the router with a shared (or injected) RateLimitTracker."""
        self._tracker: RateLimitTracker = tracker or get_tracker()

    async def route(
        self,
        role: CallRole,
        est_tokens: int = 1_000,
    ) -> Optional[str]:
        """Return the best available model_id for the call role, or None if all pools exhausted."""
        logger.debug("route() called | role=%s | est_tokens=%d", role, est_tokens)

        primary_pool = _FAST_POOL if role == "FAST" else _STRONG_POOL
        secondary_pool = _STRONG_POOL if role == "FAST" else _FAST_POOL

        model = await self._pick_from_pool(primary_pool, est_tokens, pool_label=role)
        if model:
            return model

        # cross-pool overflow when primary is exhausted
        other_label: CallRole = "STRONG" if role == "FAST" else "FAST"
        logger.warning(
            "Primary %s pool exhausted — attempting cross-pool overflow to %s pool",
            role, other_label,
        )
        result = await self._pick_from_pool(secondary_pool, est_tokens, pool_label=other_label)
        if result is None:
            logger.error(
                "All pools exhausted — no model available in either pool | "
                "role=%s | est_tokens=%d",
                role, est_tokens,
            )
        return result

    async def on_429(
        self,
        model_id: str,
        retry_after: Optional[int] = None,
    ) -> None:
        """Record a 429 response for a model and put it in cooldown."""
        logger.warning(
            "429 received for model=%s | retry_after=%s",
            model_id, f"{retry_after}s" if retry_after else "default",
        )
        await self._tracker.on_429(model_id, cooldown_seconds=retry_after)

    async def _pick_from_pool(
        self,
        pool: list[str],
        est_tokens: int,
        pool_label: str,
    ) -> Optional[str]:
        """Iterate a pool in priority order and return the first available model."""
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
        """Check cooldown, RPM/TPM headroom, and daily RPD/TPD budgets before routing."""
        state = await self._tracker.get_state(model_id)

        # check 1: active 429 cooldown
        if state.in_cooldown and not state.cooldown_expired():
            logger.debug(
                "model=%s skipped — in 429 cooldown until %s",
                model_id,
                state.cooldown_until.strftime("%H:%M:%S") if state.cooldown_until else "?",
            )
            return False

        # if minute window has reset, server-side remaining values are stale; treat as unknown
        minute_fresh = state.is_minute_window_fresh()

        # check 2: rpm headroom (only when fresh, authoritative value available)
        if minute_fresh and state.remaining_rpm is not None:
            if state.remaining_rpm < _MIN_RPM_HEADROOM:
                logger.debug(
                    "model=%s skipped — remaining_rpm=%d < threshold=%d",
                    model_id, state.remaining_rpm, _MIN_RPM_HEADROOM,
                )
                return False

        # check 3: tpm headroom (only when fresh, authoritative value available)
        if minute_fresh and state.remaining_tpm is not None:
            if state.remaining_tpm < max(est_tokens, _MIN_TPM_HEADROOM):
                logger.debug(
                    "model=%s skipped — remaining_tpm=%d < needed=%d",
                    model_id, state.remaining_tpm, max(est_tokens, _MIN_TPM_HEADROOM),
                )
                return False

        # check 4: daily rpd budget (locally tracked)
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

            # check 5: daily tpd budget (locally tracked)
            tpd_ceiling = limits.tpd - _TPD_GUARD
            if state.used_tpd + est_tokens >= tpd_ceiling:
                logger.warning(
                    "model=%s skipped — daily token budget exhausted "
                    "(used_tpd=%d + est_tokens=%d >= ceiling=%d)",
                    model_id, state.used_tpd, est_tokens, tpd_ceiling,
                )
                return False

        return True
