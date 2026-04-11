"""
JSON serializer for CacheEntry objects using Pydantic v2 native serialization.

Design:
    Implements BaseCacheSerializer using Pydantic v2's model_dump_json()
    for serialization and model_validate_json() for deserialization.
    Both methods use Pydantic's Rust-based core for performance. No manual
    dict construction or json.dumps() calls are needed.

    Performance:
        model_dump_json() uses Rust-based serializer internally (~fast)
        model_validate_json() validates all fields on reconstruction (~0.1ms)
        Typical LLM response serializes to 500-2000 bytes as JSON

    Validation on deserialize catches:
        - Corrupt JSON (parse error from backend)
        - Missing required fields (schema mismatch after code upgrade)
        - Invalid field values (negative token counts, bad datetime strings)

Chain of Responsibility:
    Instantiated by CacheManager. Called on every read path
    (deserialize raw string from backend) and every write path
    (serialize CacheEntry before calling backend.set()).

Dependencies:
    pydantic (via CacheEntry), utils.logger
"""

from utils.logger import get_logger
from cache.models.cache_entry import CacheEntry
from cache.serializers.base_serializer import BaseCacheSerializer
from cache.exceptions.cache_exceptions import CacheSerializationError

logger = get_logger(__name__)


class JSONSerializer(BaseCacheSerializer):
    """Pydantic v2 JSON serializer for CacheEntry objects."""

    @property
    def name(self) -> str:
        """Serializer identifier."""
        return "json"

    def serialize(self, entry: CacheEntry) -> str:
        """Serialize CacheEntry to JSON string.

        Uses Pydantic v2 model_dump_json() which internally calls
        the Rust-based serializer for speed. Datetime fields are
        serialized as ISO 8601 strings automatically.

        Args:
            entry: CacheEntry to serialize.

        Returns:
            Compact JSON string (no indentation for storage efficiency).

        Raises:
            CacheSerializationError: If Pydantic serialization fails.
        """
        try:
            return entry.model_dump_json()
        except Exception as e:
            logger.exception(
                "Failed to serialize cache entry: key=%s", entry.cache_key
            )
            raise CacheSerializationError(
                operation="serialize",
                message=f"JSON serialization failed for key '{entry.cache_key}': {e}",
            ) from e

    def deserialize(self, data: str) -> CacheEntry:
        """Deserialize JSON string back to CacheEntry.

        Uses Pydantic v2 model_validate_json() which validates all fields
        on reconstruction. This catches:
            - Corrupt JSON (parse error)
            - Missing required fields (schema mismatch after upgrade)
            - Invalid field values (negative token counts, bad dates)

        Args:
            data: Raw JSON string from backend.get().

        Returns:
            Validated CacheEntry instance.

        Raises:
            CacheSerializationError: If JSON parsing or validation fails.
        """
        try:
            return CacheEntry.model_validate_json(data)
        except Exception as e:
            # Truncate preview to avoid flooding logs with large payloads
            preview = data[:100] if len(data) > 100 else data
            logger.exception(
                "Failed to deserialize cache entry: data_preview=%s", preview
            )
            raise CacheSerializationError(
                operation="deserialize",
                message=f"JSON deserialization failed: {e}",
            ) from e
