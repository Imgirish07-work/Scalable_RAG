"""
L1 in-memory LRU cache backend — fastest layer in the cache hierarchy.

Design:
    Uses collections.OrderedDict for O(1) LRU eviction:
        - get() moves accessed key to end (most recently used)
        - set() appends to end, evicts from front (least recently used)
        - Bounded by CACHE_L1_MAX_SIZE entries

    Concurrency:
        asyncio.Lock protects all dict mutations. This is correct because
        FastAPI runs on a single event loop with concurrent coroutines.
        asyncio.Lock is the right primitive (not threading.Lock).
        Individual operations are fast enough that lock contention is negligible.

    TTL enforcement:
        Each entry stores its expiry timestamp alongside the value.
        get() checks expiry before returning — expired entries return None.
        Expired entries are lazily deleted on access (no background sweeper).
        evict_expired() can be called periodically for proactive cleanup.

    Memory:
        Stores raw JSON strings (not deserialized objects) — same as Redis.
        Typical entry: 500-2000 bytes (LLMResponse JSON).
        1000 entries ≈ 1-2 MB RAM — negligible for any server.

Chain of Responsibility:
    Instantiated by CacheManager. Passed to ExactCacheStrategy as the
    first backend in the lookup list. Not shared across processes — L2
    Redis handles cross-instance sharing.

Dependencies:
    asyncio, collections.OrderedDict (stdlib only)
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger
from cache.backend.base_backend import BaseCacheBackend
from cache.exceptions.cache_exceptions import CacheBackendError

logger = get_logger(__name__)


class _CacheSlot:
    """Internal storage slot — holds value + expiry metadata.

    Not a Pydantic model on purpose — this is a hot-path internal
    data structure. Plain __slots__ class avoids Pydantic overhead
    (validation, schema generation) on every read/write.

    Attributes:
        value: Serialized cache entry (JSON string).
        expires_at: UTC datetime when this slot becomes invalid.
        created_at: UTC datetime when this slot was written.
    """

    __slots__ = ("value", "expires_at", "created_at")

    def __init__(self, value: str, expires_at: datetime) -> None:
        self.value = value
        self.expires_at = expires_at
        self.created_at = datetime.now(timezone.utc)

    @property
    def is_expired(self) -> bool:
        """True if the current UTC time has passed expires_at."""
        return datetime.now(timezone.utc) >= self.expires_at


class MemoryCacheBackend(BaseCacheBackend):
    """L1 in-memory LRU cache backend.

    Attributes:
        _max_size: Maximum number of entries before LRU eviction.
        _store: OrderedDict holding _CacheSlot values, ordered by access time.
        _lock: asyncio.Lock for concurrent access protection.
        _total_evictions: Counter for observability.
        _total_expired_removals: Counter for lazy TTL deletions.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize L1 memory backend.

        Args:
            max_size: Maximum entries. When exceeded, the least recently
                      used entry is evicted. Must be > 0.

        Raises:
            ValueError: If max_size <= 0.
        """

        if max_size <= 0:
            raise ValueError(f"max_size must be greater than 0, got {max_size}")

        self._max_size = max_size
        self._store : OrderedDict[str, _CacheSlot] = OrderedDict()
        self._lock = asyncio.Lock()

        self._total_evictions : int = 0
        self._total_expired_removals : int = 0

        logger.info(
            "MemoryCacheBackend initialized: max_size=%d", self._max_size
        )

    @property
    def name(self) -> str:
        """Backend identifier used in logs and metrics."""
        return "l1_memory"

    async def get(self, key: str) -> Optional[str]:
        """Retrieve a value by key, enforcing TTL and updating LRU order.

        If the key exists but is expired, it is lazily deleted and
        None is returned (treated as a miss).

        If the key exists and is valid, it is moved to the end of
        the OrderedDict (most recently used position).

        Args:
            key: Cache key (SHA-256 hex digest).

        Returns:
            Serialized cache entry (JSON string), or None on miss/expired.
        """
        async with self._lock:
            slot = self._store.get(key)
            if slot is None:
                return None
            if slot.is_expired:
                del self._store[key]
                self._total_expired_removals += 1
                logger.debug(
                    "L1 expired entry removed on access: key=%s",
                    key[:16] + "...",
                )
                return None

            self._store.move_to_end(key)
            return slot.value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store a value with TTL, applying LRU eviction if at capacity.

        If the key already exists, it is overwritten and moved to
        the most recently used position.

        If the store is at max_size after insertion, the least recently used entry (front of OrderedDict) is evicted.

        Args:
            key: Cache key.
            value: Serialized cache entry (JSON string).
            ttl_seconds: Time-to-live in seconds.

        Raises:
            CacheBackendError: If an unexpected error occurs during write.
        """
        if ttl_seconds <= 0:
            logger.debug(
                "L1 set skipped — TTL <= 0: key=%s, ttl=%d",
                key[:16] + "...",
                ttl_seconds,
            )
            return

        # Compute absolute expiry before acquiring the lock to minimize contention
        expires_at = datetime.now(timezone.utc)
        try:
            from datetime import timedelta
            expires_at += timedelta(seconds=ttl_seconds)
        except OverflowError:
            logger.warning(
                "L1 TTL overflow, capping at 30 days: key=%s, ttl=%d",
                key[:16] + "...",
                ttl_seconds,
            )
            from datetime import timedelta
            expires_at += timedelta(days=30)
        _slot = _CacheSlot(value=value, expires_at=expires_at)

        async with self._lock:
            if key in self._store:
                # Overwrite existing entry in-place, keeping LRU order fresh
                self._store.move_to_end(key)
                self._store[key] = _slot
                logger.debug(
                    "L1 entry updated: key=%s, ttl=%ds",
                    key[:16] + "...",
                    ttl_seconds,
                )
                return

            self._store[key] = _slot
            if len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._total_evictions += 1
                logger.debug(
                    "L1 LRU eviction: evicted=%s, size=%d/%d",
                    evicted_key[:16] + "...",
                    len(self._store),
                    self._max_size,
                )

    async def delete(self, key: str) -> bool:
        """Delete a single entry by key.

        Args:
            key: Cache key to delete.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        async with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("L1 entry deleted: key=%s", key[:16] + "...")
                return True

            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists and has not expired.

        This is a read operation that does NOT update LRU order.
        Used by ExactCacheStrategy.find_similar() to check key
        existence without affecting eviction priority.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists and is not expired.
        """
        async with self._lock:
            slot = self._store.get(key)

            if slot is None:
                return False

            if slot.is_expired:
                del self._store[key]
                self._total_expired_removals += 1
                return False

            return True

    async def clear(self) -> int:
        """Remove all entries from the L1 cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info("L1 cache cleared: removed=%d entries", count)
            return count

    async def size(self) -> int:
        """Return current number of entries (including potentially expired).

        For an accurate count excluding expired entries, call
        evict_expired() first. This method is fast and does not
        scan for expiry.

        Returns:
            Number of entries in the store.
        """
        async with self._lock:
            return len(self._store)

    async def evict_expired(self) -> int:
        """Proactively scan and remove all expired entries.

        Unlike the lazy deletion in get()/exists(), this scans the
        entire store. Use sparingly — O(n) operation. Suitable for
        periodic cleanup in a background task.

        Returns:
            Number of expired entries removed.
        """
        async with self._lock:
            expired_keys = [
                k for k, slot in self._store.items() if slot.is_expired
            ]

            for k in expired_keys:
                del self._store[k]

            if expired_keys:
                self._total_expired_removals += len(expired_keys)
                logger.info(
                    "L1 expired sweep: removed=%d, remaining=%d",
                    len(expired_keys),
                    len(self._store),
                )

            return len(expired_keys)

    async def stats(self) -> dict:
        """Return L1 backend statistics for observability.

        Returns:
            Dict with size, capacity, eviction counts, and utilization.
        """
        async with self._lock:
            current_size = len(self._store)

        utilization = (
            round((current_size / self._max_size) * 100, 1)
            if self._max_size > 0
            else 0.0
        )

        return {
            "name": self.name,
            "backend_type": "in_memory_lru",
            "current_size": current_size,
            "max_size": self._max_size,
            "utilization_pct": utilization,
            "total_evictions": self._total_evictions,
            "total_expired_removals": self._total_expired_removals,
        }

    async def close(self) -> None:
        """Graceful shutdown — clear the store and release resources.

        For L1 memory backend, this just clears the dict.
        No connections to close, no flush needed.
        """
        await self.clear()
        logger.info("MemoryCacheBackend closed")
