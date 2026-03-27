"""
Exact-match cache key strategy.

Generates a deterministic SHA-256 hash from the normalized query
and model parameters. Two queries produce the same key if and only
if they are identical after normalization.

Flow:
    raw query
      → QueryNormalizerChain.build_cache_fingerprint()
          → whitespace → case → punctuation → unicode → params
      → hash_text(fingerprint)  (from utils/helpers.py)
      → SHA-256 hex digest (64 chars)

This strategy is fast (sub-millisecond), deterministic, and has
zero false positives. The tradeoff is zero semantic understanding:
    "What is RAG?" → HIT
    "what is rag"  → HIT (normalization handles this)
    "Explain RAG"  → MISS (different words, same meaning)

For semantic matching, see SemanticCacheStrategy (Phase 6).

make_key() is sync (CPU only — Rule 2).
find_similar() is async (checks backend — Rule 1).
index_entry() is a no-op (exact keys don't need a separate index).

Dependencies:
    - QueryNormalizerChain (cache/normalizers/)
    - hash_text() (utils/helpers.py)
    - BaseCacheBackend (injected for key existence checks)
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
        return "exact"

    def make_key(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> str:
        """Generate SHA-256 cache key from normalized inputs.

        Sync — CPU only (Rule 2).

        The key is deterministic: identical normalized inputs always
        produce the same 64-character hex digest.

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

        Async — checks backends which may involve I/O (Rule 1).

        Searches backends in priority order (L1 → L2). Returns on
        first match. If no backends are configured, returns None.

        For exact strategy, similarity_score is always 1.0 on hit
        and tier is always "direct" — there's no ambiguity in
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