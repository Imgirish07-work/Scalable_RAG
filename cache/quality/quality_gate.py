"""
Quality gate — filters out bad LLM responses before caching.

Prevents cache poisoning by rejecting responses that are:
    - Empty or whitespace-only text
    - Too few completion tokens (likely errors, refusals, or truncated)
    - Suspiciously fast (likely API errors returning instantly)

Runs on the write path ONLY — zero impact on read latency.
All checks are sync CPU comparisons on fields already in LLMResponse.

Sync — pure CPU, zero I/O (Rule 2).

Usage:
    gate = QualityGate(min_tokens=20, min_latency_ms=100.0)
    passed, reason = gate.check(llm_response)
    if not passed:
        logger.info("Rejected: %s", reason)
"""

from typing import Optional

from utils.logger import get_logger
from llm.models.llm_response import LLMResponse

logger = get_logger(__name__)

# Phrases that indicate the LLM could not answer from the provided context.
# Responses containing any of these (case-insensitive) must not be cached —
# they are document-specific failures that will poison future cache hits.
_NEGATIVE_PATTERNS: list[str] = [
    "i don't have enough information",
    "i do not have enough information",
    "not mentioned in the provided",
    "not found in the provided",
    "cannot answer based on",
    "not covered in the provided",
    "the provided documents do not",
    "the document does not contain",
    "no information available",
    "insufficient information",
    "not available in the context",
    "context does not contain",
    "i cannot find",
    "unable to find",
]


class QualityGate:
    """Validates LLM responses before cache entry.

    Attributes:
        _min_tokens: Minimum completion tokens to cache.
        _min_latency_ms: Minimum latency to cache (filters instant errors).
    """

    def __init__(
        self,
        min_tokens: int = 20,
        min_latency_ms: float = 100.0,
    ) -> None:
        """Initialize quality gate with thresholds.

        Args:
            min_tokens: Minimum completion_tokens. Below this → rejected.
            min_latency_ms: Minimum latency_ms. Below this → rejected.
        """
        self._min_tokens = min_tokens
        self._min_latency_ms = min_latency_ms

        logger.info(
            "QualityGate initialized: min_tokens=%d, min_latency_ms=%.0f",
            self._min_tokens,
            self._min_latency_ms,
        )

    def check(self, response: LLMResponse) -> tuple[bool, Optional[str]]:
        """Run all quality checks on an LLM response.

        Returns (True, None) if the response passes all checks.
        Returns (False, reason) if any check fails.

        Args:
            response: The LLMResponse to evaluate.

        Returns:
            Tuple of (passed: bool, reason: Optional[str]).
        """
        if not response.text or not response.text.strip():
            reason = "empty response text"
            logger.debug("Quality gate: %s", reason)
            return False, reason

        if response.completion_tokens < self._min_tokens:
            reason = (
                f"too few tokens ({response.completion_tokens} < {self._min_tokens})"
            )
            logger.debug("Quality gate: %s", reason)
            return False, reason

        if response.latency_ms < self._min_latency_ms:
            reason = (
                f"suspiciously fast ({response.latency_ms:.0f}ms < {self._min_latency_ms:.0f}ms)"
            )
            logger.debug("Quality gate: %s", reason)
            return False, reason

        # Negative response pattern check — block polite refusals from being cached.
        # A "I don't have enough information" response is query-specific and will
        # poison every future cache hit for this query until TTL expires.
        lowered = response.text.lower()
        for pattern in _NEGATIVE_PATTERNS:
            if pattern in lowered:
                reason = f"negative response pattern detected: '{pattern}'"
                logger.info("Quality gate rejected (cache poison prevention): %s", reason)
                return False, reason

        return True, None

    def passes(self, response: LLMResponse) -> bool:
        """Simple boolean check — wraps check() for convenience.

        Args:
            response: The LLMResponse to evaluate.

        Returns:
            True if the response passes all quality checks.
        """
        passed, _ = self.check(response)
        return passed

    @property
    def thresholds(self) -> dict:
        """Return current thresholds for observability."""
        return {
            "min_tokens": self._min_tokens,
            "min_latency_ms": self._min_latency_ms,
        }