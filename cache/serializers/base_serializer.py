"""
Abstract base class for cache serializers.

Responsible for converting CacheEntry <-> string for storage in backends.
Backends store raw strings — they never see Pydantic models directly.

Sync — serialization is CPU only (Rule 2). No I/O involved.

Implementations:
    JSONSerializer — human-readable, debuggable, uses Pydantic v2 native JSON
    (Future) MsgpackSerializer — compact binary, 30-40% smaller, faster parse
"""

from abc import ABC, abstractmethod
from cache.models.cache_entry import CacheEntry

class BaseCacheSerializer(ABC):
    """Interface for serializing/deserializing cache entries."""

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