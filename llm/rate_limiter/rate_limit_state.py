"""Per-model rate limit state container for the Groq Model Pool; mutated in-place by RateLimitTracker."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelRateLimitState:
    """Live rate limit state for a single model: server-authoritative minute + local daily counters."""

    # server-authoritative per-minute state (from groq headers)
    remaining_rpm: int | None = None
    remaining_tpm: int | None = None
    rpm_reset_at: datetime | None = None
    tpm_reset_at: datetime | None = None

    # locally-tracked per-day accumulators
    used_rpd: int = 0
    used_tpd: int = 0
    rpd_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    # 429 reactive cooldown
    in_cooldown: bool = False
    cooldown_until: datetime | None = None

    def is_minute_window_fresh(self) -> bool:
        """Return True if both minute windows are still open (reset times in future)."""
        now = datetime.now(timezone.utc)
        rpm_ok = self.rpm_reset_at is not None and now < self.rpm_reset_at
        tpm_ok = self.tpm_reset_at is not None and now < self.tpm_reset_at
        return rpm_ok and tpm_ok

    def cooldown_expired(self) -> bool:
        """Return True if the 429 cooldown period has elapsed."""
        if not self.in_cooldown:
            return True
        if self.cooldown_until is None:
            return True
        return datetime.now(timezone.utc) >= self.cooldown_until

    def __repr__(self) -> str:
        """Compact representation for logging."""
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
