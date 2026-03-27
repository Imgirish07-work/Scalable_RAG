"""Cache exception hierarchy."""

from cache.exceptions.cache_exceptions import (
    CacheError,
    CacheConnectionError,
    CacheSerializationError,
    CacheKeyError,
    CacheCapacityError,
    CacheBackendError,
)

__all__ = [
    "CacheError",
    "CacheConnectionError",
    "CacheSerializationError",
    "CacheKeyError",
    "CacheCapacityError",
    "CacheBackendError",
]