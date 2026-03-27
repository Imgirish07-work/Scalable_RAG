"""
Design rule:
    Cache failures must NEVER crash the request pipeline.
    Every exception here is caught inside cache_manager.py and
    converted to a silent cache miss with a warning log.
"""

class CacheError(Exception):
    """ Base exception for all cache operations. """
    def __init__(self, message: str = "Cache operation failed"):
        self.message = message
        super().__init__(self.message)


class CacheConnectionError(CacheError):
    """Redis unreachable, Qdrant timeout, or backend unavailable."""

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Connection failed for backend '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheSerializationError(CacheError):
    """Failed to serialize or deserialize a cache entry."""

    def __init__(self, operation: str, message: str = "") -> None:
        self.operation = operation
        msg = f"Serialization failed during '{operation}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheKeyError(CacheError):
    """Failed to generate a cache key from the input."""

    def __init__(self, strategy: str, message: str = "") -> None:
        self.strategy = strategy
        msg = f"Key generation failed for strategy '{strategy}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheCapacityError(CacheError):
    """Backend storage limit reached and eviction failed."""

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Capacity exceeded for backend '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheBackendError(CacheError):
    """Generic backend-level error not covered by specific exceptions."""

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Backend error in '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)