"""
In-process health tracker for LLM providers.

Design:
    Module-level singleton pattern. A single _ProviderHealthTracker instance
    (provider_health) is shared across all BaseRAG instances in the process.
    Failed providers are held in cooldown for _COOLDOWN_SECONDS, during which
    the pipeline skips them and routes directly to the configured fallback.
    Auto-recovery: once the cooldown expires the provider is retried on the
    next request without any external reset or background task.
    Thread-safe via threading.Lock.

Chain of Responsibility:
    BaseRAG.generate() queries is_available() before each LLM call →
    on failure calls mark_failed() → LLMFactory selects fallback provider →
    on success calls mark_recovered() to clear the cooldown.

Dependencies:
    threading.Lock, utils.logger.
"""

import time
from threading import Lock

from utils.logger import get_logger

logger = get_logger(__name__)

# How long to skip a failed provider before attempting recovery.
# 60s allows one retry per minute while a corporate proxy intermittently blocks.
_COOLDOWN_SECONDS: float = 60.0


class _ProviderHealthTracker:
    """In-process health state tracker for LLM providers.

    Maintains a dict of provider names to their last-failure monotonic timestamps.
    is_available() auto-clears expired entries so no background task is needed.
    Repeated failures during an active cooldown reset the clock, preventing
    premature recovery while the underlying issue persists.
    """

    def __init__(self) -> None:
        self._failed_at: dict[str, float] = {}
        self._lock = Lock()

    def mark_failed(self, provider: str) -> None:
        """Record a call failure and start or extend the cooldown window.

        Logs only on the first failure in a cooldown window to suppress
        per-query log spam when a provider is persistently unavailable.

        Args:
            provider: Provider name string e.g. 'groq', 'gemini'.
        """
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
        """Clear the failure state after a successful call.

        Args:
            provider: Provider name string e.g. 'groq', 'gemini'.
        """
        with self._lock:
            was_failed = self._failed_at.pop(provider, None) is not None

        if was_failed:
            logger.info("LLM provider '%s' recovered after successful call.", provider)

    def is_available(self, provider: str) -> bool:
        """Return True if the provider is healthy or its cooldown has expired.

        Auto-clears the failure record on expiry, giving the provider one retry
        attempt per _COOLDOWN_SECONDS without requiring an external reset.

        Args:
            provider: Provider name string e.g. 'groq', 'gemini'.

        Returns:
            True if provider is healthy or cooldown has expired.
            False if provider is in an active cooldown window.
        """
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


# Singleton shared by all BaseRAG instances in the process
provider_health = _ProviderHealthTracker()
