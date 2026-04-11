"""
Exact-match cache key strategy using SHA-256 hashing.

Design:
    Generates a deterministic 64-character SHA-256 hex digest from the
    normalized query and model parameters. Two queries produce the same
    key if and only if they are identical after normalization.

    Normalization pipeline (via QueryNormalizerChain):
        raw query → whitespace → case → punctuation → unicode → params
        → build_cache_fingerprint() → hash_text() → SHA-256 hex digest

    Zero false positives — identical keys always mean identical content.
    Zero semantic understanding — "Explain RAG" misses even if
    "What is RAG?" is cached. SemanticCacheStrategy handles that case.

    make_key() is sync (CPU only).
    find_similar() is async (checks backend existence — may involve I/O).
    index_entry() is a no-op (the backend key IS the index).

Chain of Responsibility:
    Instantiated by CacheManager with a normalizer and an ordered list
    of backends (L1 first, L2 second). CacheManager calls make_key()
    on both the read and write paths, and find_similar() as a secondary
    lookup step when direct get() returns None.

Dependencies:
    utils.helpers.hash_text, cache.normalizers, cache.backend
"""

from typing import Optional

from utils.logger import get_logger
from utils.helpers import hash_text
from cache.strategies.base_strategy import BaseCacheStrategy, SimilarityMatch
from cache.normalizers.query_normalizer import QueryNormalizerChain
from cache.backend.base_backend import BaseCacheBackend
from cache.exceptions.cache_exceptions import CacheKeyError

logger = get_logger(__name__)


class ExactCacheStrategy(BaseCacheStrategy):
    """SHA-256 hash-based exact-match cache strategy.

    Attributes:
        _normalizer: Query normalization chain instance.
        _backends: List of backends to check for key existence
                   (ordered by priority: L1 first, then L2).
    """

    def __init__(
        self,
        normalizer: Optional[QueryNormalizerChain] = None,
        backends: Optional[list[BaseCacheBackend]] = None,
    ) -> None:
        """Initialize exact strategy with normalizer and backends.

        Args:
            normalizer: Custom normalizer chain. If None, uses default
                        production chain.
            backends: Ordered list of backends to check in find_similar().
                      Typically [l1_memory_backend, l2_redis_backend].
                      If None, find_similar() always returns None.
        """
        self._normalizer = normalizer or QueryNormalizerChain()
        self._backends: list[BaseCacheBackend] = backends or []
        logger.info(
            "ExactCacheStrategy initialized: backends=%s",
            [b.name for b in self._backends],
        )

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "exact"

    def make_key(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> str:
        """Generate SHA-256 cache key from normalized inputs.

        The key is deterministic: identical normalized inputs always
        produce the same 64-character hex digest. Sync — CPU only.

        Args:
            query: Raw user query.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of system prompt.

        Returns:
            64-character SHA-256 hex digest.

        Raises:
            CacheKeyError: If normalization or hashing fails.
        """
        try:
            fingerprint = self._normalizer.build_cache_fingerprint(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
            )

            if not fingerprint:
                raise ValueError("Empty fingerprint after normalization")

            key = hash_text(fingerprint)

            logger.debug(
                "Exact key generated: query='%s' → key=%s",
                query[:60],
                key[:16] + "...",
            )

            return key

        except Exception as e:
            logger.exception("Failed to generate exact cache key")
            raise CacheKeyError(
                strategy="exact",
                message=f"Key generation failed: {e}",
            ) from e

    async def find_similar(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> Optional[SimilarityMatch]:
        """Check if the exact key exists in any backend.

        Searches backends in priority order (L1 → L2). Returns on
        first match. If no backends are configured, returns None.

        For exact strategy, similarity_score is always 1.0 on hit
        and tier is always 'direct' — there is no ambiguity in
        exact matching.

        Args:
            query: Raw user query.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of system prompt.

        Returns:
            SimilarityMatch with score=1.0 if key exists, None otherwise.
        """
        try:
            key = self.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
            )
        except CacheKeyError:
            return None

        for backend in self._backends:
            try:
                if await backend.exists(key):
                    logger.debug(
                        "Exact match found: backend=%s, key=%s",
                        backend.name,
                        key[:16] + "...",
                    )
                    return SimilarityMatch(
                        cache_key=key,
                        similarity_score=1.0,
                        tier="direct",
                    )
            except Exception:
                logger.exception(
                    "Backend '%s' failed during exact key lookup, skipping",
                    backend.name,
                )
                continue

        logger.debug(
            "Exact match not found: query='%s', key=%s",
            query[:60],
            key[:16] + "...",
        )
        return None

    async def index_entry(
        self,
        query: str,
        cache_key: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> None:
        """No-op for exact strategy.

        Exact keys don't need a separate index — the key IS the
        index. The backend's set() call is sufficient.

        This method exists to satisfy the BaseCacheStrategy interface.
        SemanticCacheStrategy overrides this with Qdrant vector upsert.
        """
        pass

    def get_normalized_query(self, query: str) -> str:
        """Expose the normalized form of a query for external use.

        Useful for:
            - Logging what the cache actually matched against
            - Debugging normalization issues
            - Building the query_hash field in CacheEntry

        Args:
            query: Raw user query.

        Returns:
            Normalized query string.
        """
        return self._normalizer.normalize(query)

    def get_query_hash(self, query: str) -> str:
        """Get SHA-256 hash of the normalized query alone (no params).

        Used for the query_hash field in CacheEntry, which supports
        deduplication on the write path — two entries with the same
        query_hash are answering the same question regardless of
        model or temperature differences.

        Args:
            query: Raw user query.

        Returns:
            64-character SHA-256 hex digest of normalized query.
        """
        normalized = self._normalizer.normalize(query)
        return hash_text(normalized)
