"""
LLM provider health tracker.

Tracks which LLM providers are currently available based on recent call
failures. When a provider fails (Zscaler block, timeout, network error),
it is marked as unavailable for _COOLDOWN_SECONDS. During that window the
pipeline skips it entirely and routes directly to the configured fallback,
avoiding repeated timeout waits on every query.

Auto-recovery: once _COOLDOWN_SECONDS passes the provider is retried.
A successful call clears the failure state immediately.

Design: module-level singleton — one shared instance across all BaseRAG
instances in the process. Thread-safe via threading.Lock.

Note: Zscaler HTML responses are already sanitised in openai_provider.py
(_handle_error / _call_api). This module only tracks health *state* and
produces clean routing log messages — no HTML ever reaches these logs.
"""

import time
from threading import Lock

from utils.logger import get_logger

logger = get_logger(__name__)

# How long to skip a failed provider before attempting recovery.
# 60s gives one retry per minute while a corporate proxy blocks the provider.
_COOLDOWN_SECONDS: float = 60.0


class _ProviderHealthTracker:
    """In-process health state for LLM providers.

    Failed providers are held in a dict keyed by provider name with the
    monotonic timestamp of their last failure. is_available() auto-clears
    expired entries so no background task is needed.
    """

    def __init__(self) -> None:
        self._failed_at: dict[str, float] = {}
        self._lock = Lock()

    def mark_failed(self, provider: str) -> None:
        """Record a call failure for provider.

        Provider will be skipped for _COOLDOWN_SECONDS. Repeated calls
        during an existing cooldown extend the timestamp so the clock
        resets — prevents premature recovery while the issue persists.
        Logs only on the first failure to avoid per-query log spam.
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
        """Clear failure state after a successful call."""
        with self._lock:
            was_failed = self._failed_at.pop(provider, None) is not None

        if was_failed:
            logger.info("LLM provider '%s' recovered after successful call.", provider)

    def is_available(self, provider: str) -> bool:
        """Return True if provider is healthy or cooldown has expired.

        Auto-clears the failure record on expiry so the provider gets one
        retry attempt per _COOLDOWN_SECONDS without any external reset.
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


# Module-level singleton shared by all BaseRAG instances in the process.
provider_health = _ProviderHealthTracker()
