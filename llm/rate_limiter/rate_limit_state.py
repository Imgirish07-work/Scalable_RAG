"""
Per-model rate limit state container for the Groq Model Pool.

Design:
    ModelRateLimitState holds the real-time usage picture for one model across
    all four rate limit dimensions. Two data sources are combined:

        1. Server-authoritative (from Groq response headers):
               remaining_rpm  — requests left this minute
               remaining_tpm  — tokens left this minute
               rpm_reset_at   — UTC datetime when the minute window resets
               tpm_reset_at   — UTC datetime when the token-minute window resets

        2. Locally-tracked (incremented on every successful call; reset at midnight UTC):
               used_rpd       — requests consumed today
               used_tpd       — tokens consumed today (prompt + completion)

    The 429 cooldown fields are set by ModelRouter.on_429() when the server
    explicitly refuses a request. They override all other state until the
    cooldown expires, ensuring the pool does not hammer a model that just
    returned 429.

    This class is a plain dataclass (not frozen) because RateLimitTracker
    mutates it in-place on every header update and every 429 event.

Chain of Responsibility:
    RateLimitTracker holds one ModelRateLimitState per model_id in a dict →
    ModelRouter reads state via RateLimitTracker.get_state() to decide
    availability → GroqModelPool calls update_from_headers() after each
    successful response and on_429() after each 429 error.

Dependencies:
    dataclasses, datetime (stdlib only).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelRateLimitState:
    """Live rate limit state for a single model.

    Fields updated from Groq response headers (per-minute, authoritative):
        remaining_rpm:  Requests remaining in the current minute window.
                        None means we have not yet received a header for this model.
        remaining_tpm:  Tokens remaining in the current minute window.
                        None means we have not yet received a header for this model.
        rpm_reset_at:   UTC datetime when the RPM window resets and remaining_rpm
                        reverts to the full RPM budget. None until first response.
        tpm_reset_at:   UTC datetime when the TPM window resets and remaining_tpm
                        reverts to the full TPM budget. None until first response.

    Fields tracked locally (per-day, approximate):
        used_rpd:   Requests made today since midnight UTC (or since process start).
        used_tpd:   Tokens consumed today (prompt_tokens + completion_tokens).
        rpd_date:   The UTC date on which used_rpd / used_tpd were last reset.
                    Used by RateLimitTracker.reset_daily_if_needed() to detect
                    midnight UTC rollovers.

    Fields for 429 cooldown (set reactively when server refuses a request):
        in_cooldown:      True while this model is in post-429 backoff.
        cooldown_until:   UTC datetime when the cooldown expires and the model
                          becomes eligible for routing again.
    """

    # Server-authoritative per-minute state (from Groq headers)
    remaining_rpm: int | None = None
    remaining_tpm: int | None = None
    rpm_reset_at: datetime | None = None
    tpm_reset_at: datetime | None = None

    # Locally-tracked per-day accumulators
    used_rpd: int = 0
    used_tpd: int = 0
    # The UTC date this model's daily counters were last reset (YYYY-MM-DD string)
    rpd_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    # 429 reactive cooldown
    in_cooldown: bool = False
    cooldown_until: datetime | None = None

    # Convenience helpers

    def is_minute_window_fresh(self) -> bool:
        """Return True if the RPM and TPM windows have not yet reset.

        When both reset timestamps are in the future the server-reported
        remaining_rpm / remaining_tpm values are still valid. Once either
        window has passed, remaining values should be treated as unreliable
        until the next header arrives.

        Returns:
            True if both minute windows are still open (i.e., reset times
            are in the future relative to now UTC).
        """
        now = datetime.now(timezone.utc)
        rpm_ok = self.rpm_reset_at is not None and now < self.rpm_reset_at
        tpm_ok = self.tpm_reset_at is not None and now < self.tpm_reset_at
        return rpm_ok and tpm_ok

    def cooldown_expired(self) -> bool:
        """Return True if the 429 cooldown period has elapsed.

        Called by ModelRouter._is_model_available() to decide whether a model
        that previously returned 429 is ready to receive requests again.

        Returns:
            True if not in cooldown OR the cooldown_until timestamp has passed.
        """
        if not self.in_cooldown:
            return True
        if self.cooldown_until is None:
            return True
        return datetime.now(timezone.utc) >= self.cooldown_until

    def __repr__(self) -> str:
        """Compact representation for logging — avoids dumping full datetime objects."""
        reset_str = (
            self.rpm_reset_at.strftime("%H:%M:%S") if self.rpm_reset_at else "?"
        )
        cooldown_str = (
            self.cooldown_until.strftime("%H:%M:%S")
            if self.cooldown_until else "none"
        )
        return (
            f"ModelRateLimitState("
            f"rem_rpm={self.remaining_rpm}, rem_tpm={self.remaining_tpm}, "
            f"used_rpd={self.used_rpd}, used_tpd={self.used_tpd}, "
            f"rpm_reset={reset_str}, "
            f"cooldown={'ON until ' + cooldown_str if self.in_cooldown else 'OFF'}"
            f")"
        )
