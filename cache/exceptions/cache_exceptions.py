"""
Custom exception hierarchy for cache layer errors.

Design:
    All cache exceptions inherit from CacheError so callers can catch
    the entire hierarchy with a single except clause. Specific subclasses
    carry context fields (backend name, operation name) to make log
    messages actionable without parsing the message string.

    Design rule:
        Cache failures must NEVER crash the request pipeline.
        Every exception here is caught inside cache_manager.py and
        converted to a silent cache miss with a warning log.

Chain of Responsibility:
    Raised by backend implementations (MemoryCacheBackend,
    RedisCacheBackend), serializers (JSONSerializer), and strategies
    (ExactCacheStrategy, SemanticCacheStrategy). Caught and suppressed
    by CacheManager before reaching the RAG pipeline.

Dependencies:
    None (stdlib Exception only)
"""


class CacheError(Exception):
    """Base exception for all cache operations."""

    def __init__(self, message: str = "Cache operation failed"):
        self.message = message
        super().__init__(self.message)


class CacheConnectionError(CacheError):
    """Redis unreachable, Qdrant timeout, or backend unavailable.

    Attributes:
        backend: Name of the backend that failed to connect.
    """

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Connection failed for backend '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheSerializationError(CacheError):
    """Failed to serialize or deserialize a cache entry.

    Attributes:
        operation: The operation that failed ('serialize' or 'deserialize').
    """

    def __init__(self, operation: str, message: str = "") -> None:
        self.operation = operation
        msg = f"Serialization failed during '{operation}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheKeyError(CacheError):
    """Failed to generate a cache key from the input.

    Attributes:
        strategy: The strategy that failed key generation.
    """

    def __init__(self, strategy: str, message: str = "") -> None:
        self.strategy = strategy
        msg = f"Key generation failed for strategy '{strategy}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheCapacityError(CacheError):
    """Backend storage limit reached and eviction failed.

    Attributes:
        backend: Name of the backend that hit capacity.
    """

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Capacity exceeded for backend '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class CacheBackendError(CacheError):
    """Generic backend-level error not covered by specific exceptions.

    Attributes:
        backend: Name of the backend where the error occurred.
    """

    def __init__(self, backend: str, message: str = "") -> None:
        self.backend = backend
        msg = f"Backend error in '{backend}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)
