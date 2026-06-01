"""Singleton tracker that maintains per-model rate limit state for the Groq Model Pool."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from llm.rate_limiter.rate_limit_state import ModelRateLimitState
from utils.logger import get_logger

logger = get_logger(__name__)

# default cooldown after 429 when retry-after is unparseable
_DEFAULT_COOLDOWN_SECONDS = 60


class RateLimitTracker:
    """Maintains live per-model rate limit state for the Groq Model Pool."""

    def __init__(self) -> None:
        """Initialize an empty tracker with a single asyncio lock."""
        self._state: dict[str, ModelRateLimitState] = {}
        self._lock = asyncio.Lock()

    async def update_from_headers(
        self,
        model_id: str,
        headers: dict[str, str],
    ) -> None:
        """Update per-minute remaining counters from Groq's x-ratelimit-* response headers."""
        async with self._lock:
            state = self._get_or_create_state(model_id)
            self._reset_daily_if_needed(state)

            raw_rem_req = headers.get("x-ratelimit-remaining-requests")
            if raw_rem_req is not None:
                try:
                    state.remaining_rpm = int(raw_rem_req)
                except ValueError:
                    logger.debug(
                        "Could not parse x-ratelimit-remaining-requests='%s' for model=%s",
                        raw_rem_req, model_id,
                    )

            raw_rem_tok = headers.get("x-ratelimit-remaining-tokens")
            if raw_rem_tok is not None:
                try:
                    state.remaining_tpm = int(raw_rem_tok)
                except ValueError:
                    logger.debug(
                        "Could not parse x-ratelimit-remaining-tokens='%s' for model=%s",
                        raw_rem_tok, model_id,
                    )

            raw_reset_req = headers.get("x-ratelimit-reset-requests")
            if raw_reset_req is not None:
                state.rpm_reset_at = self._parse_reset_time(raw_reset_req, model_id, "reset-requests")

            raw_reset_tok = headers.get("x-ratelimit-reset-tokens")
            if raw_reset_tok is not None:
                state.tpm_reset_at = self._parse_reset_time(raw_reset_tok, model_id, "reset-tokens")

            # clear cooldown on a successful call
            if state.in_cooldown and state.cooldown_expired():
                state.in_cooldown = False
                state.cooldown_until = None
                logger.info("Cooldown cleared for model=%s (successful response received)", model_id)

            logger.debug("Header update for model=%s | %s", model_id, state)

    async def increment_daily(self, model_id: str, tokens_used: int) -> None:
        """Increment locally-tracked daily request and token accumulators."""
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
        """Put a model into cooldown after receiving a 429 response."""
        secs = cooldown_seconds if cooldown_seconds is not None else _DEFAULT_COOLDOWN_SECONDS
        until = datetime.now(timezone.utc) + timedelta(seconds=secs)

        async with self._lock:
            state = self._get_or_create_state(model_id)
            state.in_cooldown = True
            state.cooldown_until = until
            # zero minute remaining so router skips this model immediately
            state.remaining_rpm = 0
            state.remaining_tpm = 0

        logger.warning(
            "Model %s in 429 cooldown for %ds (until %s UTC)",
            model_id, secs, until.strftime("%H:%M:%S"),
        )

    async def get_state(self, model_id: str) -> ModelRateLimitState:
        """Return the current rate limit state for a model; creates a blank state on first sight."""
        async with self._lock:
            return self._get_or_create_state(model_id)

    def _get_or_create_state(self, model_id: str) -> ModelRateLimitState:
        """Return existing state or create a fresh one; caller must hold self._lock."""
        if model_id not in self._state:
            self._state[model_id] = ModelRateLimitState()
            logger.debug("Initialized rate limit state for model=%s", model_id)
        return self._state[model_id]

    def _reset_daily_if_needed(self, state: ModelRateLimitState) -> None:
        """Zero out daily accumulators on midnight UTC rollover; caller must hold self._lock."""
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
