"""In-process LLM provider health tracker with cooldown-based auto-recovery."""

import time
from threading import Lock

from utils.logger import get_logger

logger = get_logger(__name__)

# 60s allows one retry per minute through intermittent corporate proxy blocks
_COOLDOWN_SECONDS: float = 60.0


class _ProviderHealthTracker:
    """In-process health state tracker for LLM providers; thread-safe via Lock."""

    def __init__(self) -> None:
        self._failed_at: dict[str, float] = {}
        self._lock = Lock()

    def mark_failed(self, provider: str) -> None:
        """Record a call failure and start or extend the cooldown window."""
        with self._lock:
            is_new = provider not in self._failed_at
            self._failed_at[provider] = time.monotonic()

        if is_new:
            logger.warning(
                "LLM provider '%s' marked unavailable — "
                "routing to fallback for %.0fs cooldown.",
                provider,
                _COOLDOWN_SECONDS,
            )

    def mark_recovered(self, provider: str) -> None:
        """Clear the failure state after a successful call."""
        with self._lock:
            was_failed = self._failed_at.pop(provider, None) is not None

        if was_failed:
            logger.info("LLM provider '%s' recovered after successful call.", provider)

    def is_available(self, provider: str) -> bool:
        """Return True if the provider is healthy or its cooldown has expired."""
        with self._lock:
            failed_at = self._failed_at.get(provider)
            if failed_at is None:
                return True
            elapsed = time.monotonic() - failed_at
            if elapsed >= _COOLDOWN_SECONDS:
                del self._failed_at[provider]
                expired = True
            else:
                expired = False

        if expired:
            logger.info(
                "LLM provider '%s' cooldown expired — attempting recovery.",
                provider,
            )
            return True

        return False


# singleton shared by all BaseRAG instances
provider_health = _ProviderHealthTracker()
