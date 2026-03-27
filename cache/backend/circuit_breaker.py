"""
Circuit breaker for async backend operations.

Prevents cascading failures when an external dependency (Redis, Qdrant)
becomes unhealthy. Instead of letting every request hang for the full
timeout duration, the circuit breaker short-circuits after N consecutive
failures and skips the backend entirely for a configurable reset period.

Three states:
    CLOSED  — normal operation, all requests pass through
    OPEN    — backend is unhealthy, all requests are immediately rejected
    HALF_OPEN — reset period elapsed, next request is a probe:
                if it succeeds → CLOSED
                if it fails    → OPEN again

State transitions:
    CLOSED → OPEN:
        Triggered when consecutive_failures >= failure_threshold.

    OPEN → HALF_OPEN:
        Triggered when current_time >= last_failure_time + reset_seconds.
        The next call becomes a probe.

    HALF_OPEN → CLOSED:
        Probe call succeeded.

    HALF_OPEN → OPEN:
        Probe call failed, reset the timer.

Usage:
    breaker = CircuitBreaker(name="redis", failure_threshold=5, reset_seconds=60.0)

    if not breaker.allow_request():
        return None  # Skip backend

    try:
        result = await redis.get(key)
        breaker.record_success()
        return result
    except Exception:
        breaker.record_failure()
        raise

Sync — state checks are CPU-only (Rule 2).
No locks needed — single event loop + atomic attribute assignments.
"""

import time

from utils.logger import get_logger

logger = get_logger(__name__)

class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Async-safe circuit breaker for external dependencies.

    Attributes:
        _name: Human-readable name for logging (e.g. 'redis', 'qdrant').
        _failure_threshold: Consecutive failures before opening the circuit.
        _reset_seconds: Seconds to wait before probing a tripped circuit.
        _state: Current circuit state.
        _consecutive_failures: Running failure counter (resets on success).
        _last_failure_time: Timestamp of most recent failure.
        _total_trips: Lifetime count of CLOSED → OPEN transitions.
        _total_rejected: Lifetime count of requests rejected by open circuit.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_seconds: float = 60.0,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds

        self._state : CircuitState = CircuitState.CLOSED
        self._consecutive_failures : int = 0
        self._last_failure_time : float = 0.0
        self._total_trips : int = 0
        self._total_rejected : int = 0

        logger.info(
            "CircuitBreaker '%s' initialized: threshold=%d, reset=%.1fs",
            self._name,
            self._failure_threshold,
            self._reset_seconds,
        )

    @property   
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        """Current state, accounting for automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._reset_seconds:
                logger.info(
                    "CircuitBreaker '%s': OPEN → HALF_OPEN (reset period elapsed)",
                    self._name,
                )
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if the request should proceed (CLOSED or HALF_OPEN probe).
            False if the circuit is OPEN (request should be skipped).
        """
        current = self.state

        if current == CircuitState.CLOSED:
            return True
        
        if current == CircuitState.HALF_OPEN:
            logger.debug(
                "CircuitBreaker '%s': allowing probe request (HALF_OPEN)",
                self._name,
            )
            return True

        # OPEN state
        self._total_rejected += 1
        if self._total_rejected % 50 == 0:
            logger.warning(
                "CircuitBreaker '%s': OPEN — request rejected "
                "(total_rejected=%d, resets in %.1fs)",
                self._name,
                self._total_rejected,
                max(
                    0,
                    self._reset_seconds
                    - (time.monotonic() - self._last_failure_time),
                ),
            )
        return False

    def record_success(self) -> None:
        """Record a successful operation — resets failure state."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "CircuitBreaker '%s': HALF_OPEN → CLOSED (probe succeeded)",
                self._name,
            )
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed operation — may trip the circuit."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._total_trips += 1
            logger.warning(
                "CircuitBreaker '%s': HALF_OPEN → OPEN (probe failed, trip #%d)",
                self._name,
                self._total_trips,
            )
            return

        if(
            self._state == CircuitState.CLOSED
            and self._consecutive_failures >= self._failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._total_trips += 1
            logger.warning(
                "CircuitBreaker '%s': CLOSED → OPEN "
                "(failures=%d >= threshold=%d, trip #%d)",
                self._name,
                self._consecutive_failures,
                self._failure_threshold,
                self._total_trips,
            )
            return

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        logger.info("CircuitBreaker '%s': manually reset to CLOSED", self._name)

    def stats(self) -> dict:
        """Return circuit breaker statistics for observability."""
        return {
            "name": self._name,
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._failure_threshold,
            "reset_seconds": self._reset_seconds,
            "total_trips": self._total_trips,
            "total_rejected": self._total_rejected,
        }