"""
Abstract base class for all cache storage backends.

Design:
    ABC defining the async contract that every backend must implement.
    MemoryBackend (L1) and RedisBackend (L2) are the two concrete
    implementations. All methods are async so the event loop is never
    blocked by storage I/O, regardless of backend type.

    Design contract:
        - get() returns None on miss, never raises on a missing key
        - set() silently overwrites existing keys
        - delete() is idempotent — deleting a missing key is a no-op
        - clear() removes ALL entries (use with caution in production)
        - All methods raise CacheBackendError on internal failures
        - The caller (cache_manager.py) wraps all calls in try/except

Chain of Responsibility:
    Created and injected by CacheManager. Called by ExactCacheStrategy
    and SemanticCacheStrategy during lookup and write operations.
    CircuitBreaker wraps every call in RedisCacheBackend.

Dependencies:
    abc (stdlib only — no third-party dependencies at this level)
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseCacheBackend(ABC):
    """Async interface for cache storage backends.

    Attributes:
        name: Human-readable backend identifier used in logs and metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend, e.g. 'l1_memory' or 'l2_redis'."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Retrieve a cached value by key.

        Args:
            key: Cache key (SHA-256 hash or vector ID).

        Returns:
            Serialized cache entry as JSON string, or None on miss.
        """

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store a value with TTL.

        Args:
            key: Cache key.
            value: Serialized cache entry (JSON string from serializer).
            ttl_seconds: Time-to-live in seconds. Entry auto-expires after TTL.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a single entry. Returns True if key existed.

        Args:
            key: Cache key to delete.

        Returns:
            True if the key was found and deleted, False if missing.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists without retrieving the value.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists and has not expired.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries from this backend.

        Returns:
            Number of entries removed.
        """

    @abstractmethod
    async def size(self) -> int:
        """Return the current number of entries in this backend."""

    @abstractmethod
    async def stats(self) -> dict:
        """Return backend-specific statistics.

        Returns:
            Dict with at minimum: name, size, backend_type.
            Implementations add their own fields (e.g. max_size for memory,
            connection_pool_size for Redis).
        """

    @abstractmethod
    async def close(self) -> None:
        """Graceful shutdown — release connections, flush if needed."""
