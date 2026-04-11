"""
Singleton tracker that maintains per-model rate limit state for the Groq Model Pool.

Design:
    RateLimitTracker is the single source of truth for "how much budget does
    each model have right now?". It is used by ModelRouter (read) and
    GroqModelPool (write) — nothing else should touch it directly.

    Two types of updates flow in:

        1. update_from_headers(model_id, headers):
               Called after every successful Groq response. Parses the
               x-ratelimit-* headers and updates the per-minute remaining
               counts (remaining_rpm, remaining_tpm, rpm_reset_at, tpm_reset_at).
               These are authoritative — they come from the server.

        2. increment_daily(model_id, tokens_used):
               Called after every successful Groq response (alongside header
               update). Increments used_rpd by 1 and used_tpd by tokens_used.
               These are locally approximated — Groq does not expose remaining
               RPD/TPD in headers.

        3. on_429(model_id, cooldown_seconds):
               Called when a model returns HTTP 429. Puts the model into a
               timed cooldown. ModelRouter will not route to it until the
               cooldown expires.

    Midnight UTC reset:
        reset_daily_if_needed() is called at the start of update_from_headers()
        and increment_daily(). It checks whether the current UTC date has
        advanced past the date stored in state.rpd_date, and zeroes out
        used_rpd / used_tpd if so. This prevents daily-budget exhaustion from
        carrying over past midnight UTC.

    Thread / async safety:
        All state mutations are protected by a single asyncio.Lock per tracker
        instance. This is safe for asyncio concurrency (multiple coroutines,
        single event loop). It is NOT safe for multi-process use, but the RAG
        pipeline is always single-process.

    Singleton pattern:
        get_tracker() returns the module-level _tracker singleton. This ensures
        all parts of the application share one state dict — if GroqModelPool
        creates multiple GroqProvider instances for different models, they all
        read/write the same tracker.

Chain of Responsibility:
    GroqModelPool calls update_from_headers() and increment_daily() after each
    successful response, and on_429() after each 429 → RateLimitTracker
    updates ModelRateLimitState in _state dict → ModelRouter reads via
    get_state() to make routing decisions.

Dependencies:
    asyncio, datetime (stdlib), llm.rate_limiter.rate_limit_state,
    llm.rate_limiter.model_limits, utils.logger.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from llm.rate_limiter.rate_limit_state import ModelRateLimitState
from utils.logger import get_logger

logger = get_logger(__name__)

# Default cooldown after a 429 response if we cannot parse a Retry-After header.
# 60 seconds is conservative enough to avoid immediate re-429, short enough
# to not lose significant throughput on a 30-RPM model.
_DEFAULT_COOLDOWN_SECONDS = 60


class RateLimitTracker:
    """Maintains live per-model rate limit state for the Groq Model Pool.

    Attributes:
        _state: Dict mapping model_id → ModelRateLimitState.
        _lock:  asyncio.Lock protecting all mutations to _state.
    """

    def __init__(self) -> None:
        """Initialize an empty tracker with a single asyncio lock."""
        self._state: dict[str, ModelRateLimitState] = {}
        self._lock = asyncio.Lock()

    # Write path — called by GroqModelPool after each API call

    async def update_from_headers(
        self,
        model_id: str,
        headers: dict[str, str],
    ) -> None:
        """Update per-minute remaining counters from Groq response headers.

        Groq includes these headers on every response:
            x-ratelimit-remaining-requests  → remaining_rpm
            x-ratelimit-remaining-tokens    → remaining_tpm
            x-ratelimit-reset-requests      → when RPM window resets (ISO-8601 or Xs/ms)
            x-ratelimit-reset-tokens        → when TPM window resets (ISO-8601 or Xs/ms)

        Missing or unparseable headers are silently ignored — the tracker
        retains whatever value was last successfully parsed.

        Args:
            model_id: Exact Groq model ID e.g. 'llama-3.1-8b-instant'.
            headers:  Response headers dict from the HTTP response. Header
                      keys are expected in lowercase (OpenAI SDK normalises them).
        """
        async with self._lock:
            state = self._get_or_create_state(model_id)
            self._reset_daily_if_needed(state)

            # remaining requests this minute
            raw_rem_req = headers.get("x-ratelimit-remaining-requests")
            if raw_rem_req is not None:
                try:
                    state.remaining_rpm = int(raw_rem_req)
                except ValueError:
                    logger.debug(
                        "Could not parse x-ratelimit-remaining-requests='%s' for model=%s",
                        raw_rem_req, model_id,
                    )

            # remaining tokens this minute
            raw_rem_tok = headers.get("x-ratelimit-remaining-tokens")
            if raw_rem_tok is not None:
                try:
                    state.remaining_tpm = int(raw_rem_tok)
                except ValueError:
                    logger.debug(
                        "Could not parse x-ratelimit-remaining-tokens='%s' for model=%s",
                        raw_rem_tok, model_id,
                    )

            # RPM window reset time
            raw_reset_req = headers.get("x-ratelimit-reset-requests")
            if raw_reset_req is not None:
                state.rpm_reset_at = self._parse_reset_time(raw_reset_req, model_id, "reset-requests")

            # TPM window reset time
            raw_reset_tok = headers.get("x-ratelimit-reset-tokens")
            if raw_reset_tok is not None:
                state.tpm_reset_at = self._parse_reset_time(raw_reset_tok, model_id, "reset-tokens")

            # Clear cooldown on a successful call — server accepted the request
            if state.in_cooldown and state.cooldown_expired():
                state.in_cooldown = False
                state.cooldown_until = None
                logger.info("Cooldown cleared for model=%s (successful response received)", model_id)

            logger.debug("Header update for model=%s | %s", model_id, state)

    async def increment_daily(self, model_id: str, tokens_used: int) -> None:
        """Increment the daily request and token accumulators for a model.

        Called alongside update_from_headers() after every successful call.
        The daily accumulators are locally tracked because Groq does not expose
        remaining-RPD or remaining-TPD in response headers.

        Args:
            model_id:    Exact Groq model ID.
            tokens_used: Total tokens consumed by the call (prompt + completion).
        """
        async with self._lock:
            state = self._get_or_create_state(model_id)
            self._reset_daily_if_needed(state)
            state.used_rpd += 1
            state.used_tpd += tokens_used
            logger.debug(
                "Daily counter updated | model=%s | used_rpd=%d | used_tpd=%d",
                model_id, state.used_rpd, state.used_tpd,
            )

    async def on_429(
        self,
        model_id: str,
        cooldown_seconds: Optional[int] = None,
    ) -> None:
        """Put a model into cooldown after receiving a 429 response.

        The model will be excluded from routing until the cooldown expires.
        ModelRouter.on_429() is the caller; it passes the Retry-After value
        from the response header when available.

        Args:
            model_id:         Exact Groq model ID that returned 429.
            cooldown_seconds: How long to pause routing to this model. Defaults
                              to _DEFAULT_COOLDOWN_SECONDS when not provided.
        """
        secs = cooldown_seconds if cooldown_seconds is not None else _DEFAULT_COOLDOWN_SECONDS
        until = datetime.now(timezone.utc) + timedelta(seconds=secs)

        async with self._lock:
            state = self._get_or_create_state(model_id)
            state.in_cooldown = True
            state.cooldown_until = until
            # Reset minute remaining to 0 so ModelRouter skips this model immediately
            state.remaining_rpm = 0
            state.remaining_tpm = 0

        logger.warning(
            "Model %s in 429 cooldown for %ds (until %s UTC)",
            model_id, secs, until.strftime("%H:%M:%S"),
        )

    # Read path — called by ModelRouter

    async def get_state(self, model_id: str) -> ModelRateLimitState:
        """Return the current rate limit state for a model.

        Creates a blank ModelRateLimitState for models not yet seen (first call
        before any response header has been received). In that case
        remaining_rpm and remaining_tpm are None, which ModelRouter interprets
        as "unknown — assume available" to allow the first request through.

        Args:
            model_id: Exact Groq model ID.

        Returns:
            ModelRateLimitState for the model (never None).
        """
        async with self._lock:
            return self._get_or_create_state(model_id)

    # Private helpers

    def _get_or_create_state(self, model_id: str) -> ModelRateLimitState:
        """Return existing state or create a fresh one for this model.

        Must be called while holding self._lock.

        Args:
            model_id: Exact Groq model ID.

        Returns:
            ModelRateLimitState for model_id (created if absent).
        """
        if model_id not in self._state:
            self._state[model_id] = ModelRateLimitState()
            logger.debug("Initialized rate limit state for model=%s", model_id)
        return self._state[model_id]

    def _reset_daily_if_needed(self, state: ModelRateLimitState) -> None:
        """Zero out daily accumulators if the UTC date has advanced.

        Groq's RPD window resets at midnight UTC. We detect the rollover by
        comparing today's UTC date string to the date stored in state.rpd_date.

        Must be called while holding self._lock.

        Args:
            state: The ModelRateLimitState to conditionally reset.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.rpd_date != today:
            logger.info(
                "Midnight UTC rollover detected — resetting daily counters "
                "(used_rpd=%d, used_tpd=%d → 0)",
                state.used_rpd, state.used_tpd,
            )
            state.used_rpd = 0
            state.used_tpd = 0
            state.rpd_date = today

    @staticmethod
    def _parse_reset_time(
        raw: str,
        model_id: str,
        header_name: str,
    ) -> Optional[datetime]:
        """Parse a Groq reset-time header into a timezone-aware UTC datetime.

        Groq sends reset times in one of two formats:
            "60s"   — seconds from now (e.g. "60s", "1.5s")
            "500ms" — milliseconds from now (e.g. "500ms")

        If neither format parses, the value is silently ignored.

        Args:
            raw:         Raw header value string.
            model_id:    Model ID for logging context.
            header_name: Header field name for logging context.

        Returns:
            timezone-aware UTC datetime, or None if parsing failed.
        """
        now = datetime.now(timezone.utc)
        raw = raw.strip()

        try:
            if raw.endswith("ms"):
                ms = float(raw[:-2])
                return now + timedelta(milliseconds=ms)
            if raw.endswith("s"):
                sec = float(raw[:-1])
                return now + timedelta(seconds=sec)
        except ValueError:
            pass

        logger.debug(
            "Could not parse x-ratelimit-%s='%s' for model=%s",
            header_name, raw, model_id,
        )
        return None


# Module-level singleton — all callers share one state dict
_tracker: Optional[RateLimitTracker] = None


def get_tracker() -> RateLimitTracker:
    """Return the module-level RateLimitTracker singleton.

    Initializes the tracker on first call. Subsequent calls return the same
    instance, ensuring GroqModelPool and ModelRouter share one state dict
    regardless of how many times they import this module.

    Returns:
        The shared RateLimitTracker instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = RateLimitTracker()
        logger.info("RateLimitTracker singleton initialized")
    return _tracker
