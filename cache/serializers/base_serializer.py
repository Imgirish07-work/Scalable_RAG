"""
Abstract base class for cache value serializers.

Design:
    Defines the two-method interface that converts CacheEntry objects
    to/from strings for storage in cache backends. Backends store raw
    strings — they never interact with Pydantic models directly.
    Separating serialization from storage allows backends and serializers
    to evolve independently.

    Implementations:
        JSONSerializer    — human-readable, uses Pydantic v2 native JSON
        (Future) MsgpackSerializer — compact binary, ~30-40% smaller, faster parse

Chain of Responsibility:
    Instantiated by CacheManager (currently always JSONSerializer).
    Called by CacheManager on every read (deserialize) and write
    (serialize) operation before data reaches the backend.

Dependencies:
    abc, cache.models.cache_entry (stdlib + internal)
"""

from abc import ABC, abstractmethod
from cache.models.cache_entry import CacheEntry


class BaseCacheSerializer(ABC):
    """Interface for serializing/deserializing cache entries.

    Attributes:
        name: Serializer identifier for logging (e.g. 'json', 'msgpack').
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Serializer name for logging (e.g. 'json', 'msgpack')."""

    @abstractmethod
    def serialize(self, entry: CacheEntry) -> str:
        """Convert a CacheEntry to a string for backend storage.

        Args:
            entry: The CacheEntry to serialize.

        Returns:
            String representation suitable for backend.set().

        Raises:
            CacheSerializationError: If serialization fails.
        """

    @abstractmethod
    def deserialize(self, data: str) -> CacheEntry:
        """Convert a stored string back to a CacheEntry.

        Args:
            data: Raw string from backend.get().

        Returns:
            Reconstructed CacheEntry with all metadata.

        Raises:
            CacheSerializationError: If deserialization fails (corrupt data,
                schema mismatch, invalid JSON, etc).
        """
