"""
Semantic cache strategy using BGE embeddings and Qdrant vector similarity.

Design:
    Catches paraphrased queries that exact matching misses by embedding
    each query with the shared BGE model and searching an in-memory Qdrant
    collection for the nearest cached vector. Tiered cosine thresholds
    gate what confidence level qualifies as a hit.

    Cached:  "What is RAG?"
    Hit:     "Explain RAG to me"       (cosine ~0.96, tier=high)
    Hit:     "Can you define RAG?"     (cosine ~0.95, tier=high)
    Miss:    "What is RLM?"            (cosine ~0.72, below threshold)

    Tiered thresholds (configurable):
        DIRECT  — cosine >= 0.98 → serve directly, high confidence
        HIGH    — cosine >= 0.93 → serve with monitoring flag
        MISS    — cosine <  0.93 → no match

    Embedding reuse:
        Uses get_embeddings() from vectorstore/embeddings.py.
        Same lru_cache instance as RAG retrieval — zero duplicate model
        loading. embed_query() returns list[float] already L2-normalized.

    Qdrant collection stores:
        vector  — BGE embedding of the query (384-dim for bge-small)
        payload — {cache_key, query_text, model_name, system_prompt_hash}

    Normalization:
        Uses a LIGHTER normalizer chain (whitespace only) for embedding —
        over-normalizing degrades semantic quality.
        make_key() uses the FULL normalizer chain (same as exact strategy).

Chain of Responsibility:
    Instantiated by CacheManager._initialize_semantic() when
    CACHE_STRATEGY='semantic'. CacheManager calls find_similar() on the
    read path after exact miss, and index_entry() on the write path.
    All Qdrant I/O is offloaded via asyncio.to_thread() (sync client).

Dependencies:
    qdrant_client, vectorstore.embeddings.get_embeddings,
    cache.normalizers.query_normalizer, utils.helpers.hash_text
"""

import asyncio
import uuid
from typing import Optional

from utils.logger import get_logger
from utils.helpers import hash_text
from vectorstore.embeddings import get_embeddings, get_embedding_dimension
from cache.strategies.base_strategy import BaseCacheStrategy, SimilarityMatch
from cache.normalizers.query_normalizer import (
    QueryNormalizerChain,
    WhitespaceNormalizer,
)
from cache.exceptions.cache_exceptions import CacheKeyError

logger = get_logger(__name__)


class SemanticCacheStrategy(BaseCacheStrategy):
    """BGE embedding + Qdrant vector similarity cache strategy.

    Reuses the existing embedding model from vectorstore/embeddings.py.
    No duplicate model loading — get_embeddings() is lru_cached globally.

    Attributes:
        _embed_normalizer: Light normalizer for embedding (whitespace only).
        _key_normalizer: Full normalizer for SHA-256 key generation.
        _embedding_dim: Embedding vector dimension (from get_embedding_dimension()).
        _collection: Qdrant collection name for cache vectors.
        _threshold_direct: Cosine threshold for direct serve (>= 0.98).
        _threshold_high: Cosine threshold for high-confidence serve (>= 0.93).
        _client: QdrantClient instance.
        _initialized: Whether initialize() has been called.
    """

    def __init__(
        self,
        collection_name: str = "cache_semantic",
        threshold_direct: float = 0.98,
        threshold_high: float = 0.93,
        qdrant_client=None,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        use_memory: bool = True,
    ) -> None:
        """Sync constructor — stores config, no I/O.

        Args:
            collection_name: Qdrant collection for cache vectors.
            threshold_direct: Cosine >= this → direct serve.
            threshold_high: Cosine >= this → high-confidence serve.
            qdrant_client: Optional pre-built QdrantClient (for testing).
            qdrant_url: Qdrant server URL (if no client provided).
            qdrant_api_key: Qdrant API key (if no client provided).
            use_memory: If True and no client/url, use in-memory Qdrant.
        """
        self._embed_normalizer = QueryNormalizerChain(
            steps=[WhitespaceNormalizer()]
        )
        self._key_normalizer = QueryNormalizerChain()
        self._embedding_dim: int = 0
        self._collection = collection_name
        self._threshold_direct = threshold_direct
        self._threshold_high = threshold_high
        self._client = qdrant_client or self._create_client(
            qdrant_url, qdrant_api_key, use_memory
        )
        self._initialized: bool = False
        logger.info(
            "SemanticCacheStrategy created: collection='%s', "
            "thresholds=(direct=%.2f, high=%.2f)",
            self._collection,
            self._threshold_direct,
            self._threshold_high,
        )

    @staticmethod
    def _create_client(url: str, api_key: str, use_memory: bool):
        """Create a QdrantClient based on provided config.

        Args:
            url: Qdrant server URL. If non-empty, connects to remote server.
            api_key: API key for authenticated remote Qdrant.
            use_memory: If True and url is empty, use in-memory Qdrant.

        Returns:
            Configured QdrantClient instance.
        """
        from qdrant_client import QdrantClient

        if url:
            logger.info("Semantic cache Qdrant: url=%s", url)
            return QdrantClient(url=url, api_key=api_key or None)

        if use_memory:
            logger.info("Semantic cache Qdrant: in-memory mode")
            return QdrantClient(location=":memory:")

        logger.info("Semantic cache Qdrant: localhost default")
        return QdrantClient()

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "semantic"

    # Initialization

    async def initialize(self) -> None:
        """Create the Qdrant collection and verify embedding model.

        Must be called once before find_similar() or index_entry().
        Safe to call multiple times — subsequent calls are no-ops.

        Loads the embedding model via get_embeddings() (lru_cached —
        if already loaded by RAG layer, this is instant).
        Gets dimension via get_embedding_dimension() (from map or computed).

        Uses asyncio.to_thread() because QdrantClient is sync.
        """
        if self._initialized:
            return

        try:
            self._embedding_dim = await asyncio.to_thread(get_embedding_dimension)
            logger.info(
                "Embedding dimension resolved: dim=%d", self._embedding_dim
            )

            await asyncio.to_thread(self._ensure_model_loaded)
            await asyncio.to_thread(self._setup_collection)

            self._initialized = True
            logger.info(
                "SemanticCacheStrategy initialized: collection='%s', dim=%d",
                self._collection,
                self._embedding_dim,
            )
        except Exception as e:
            logger.exception("Semantic strategy initialization failed")
            raise

    def _ensure_model_loaded(self) -> None:
        """Force the embedding model to load if not already cached.

        get_embeddings() is lru_cached — calling it here ensures the
        model is ready before any search/index operation. If the RAG
        layer already called it, this is a no-op (returns cached instance).
        """
        get_embeddings()

    def _setup_collection(self) -> None:
        """Create cache_semantic collection if missing. Sync — runs in thread."""
        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        existing = {c.name for c in collections}

        # Skip creation if the collection already exists from a prior run
        if self._collection in existing:
            logger.info(
                "Semantic cache collection '%s' already exists",
                self._collection,
            )
            return

        # Create with cosine distance to match L2-normalized BGE output
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._embedding_dim,
                distance=Distance.COSINE,
            )
        )
        logger.info(
            "Created semantic cache collection: name='%s', dim=%d, distance=cosine",
            self._collection,
            self._embedding_dim,
        )

    # Embeddings

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string using the shared BGE model.

        Uses LIGHT normalization (whitespace only) before embedding.
        The model handles semantic understanding — over-normalizing
        degrades quality.

        Calls get_embeddings().embed_query() from vectorstore/embeddings.py.
        Returns list[float] already L2-normalized (normalize_embeddings=True
        is set in the embeddings module).

        Sync — CPU only, runs inside asyncio.to_thread() from async callers.

        Args:
            query: Raw user query.

        Returns:
            Embedding vector as list[float] (384-dim for bge-small).
        """
        normalized = self._embed_normalizer.normalize(query)

        if not normalized:
            return [0.0] * self._embedding_dim

        embeddings = get_embeddings()
        vector = embeddings.embed_query(normalized)
        return vector

    async def _async_embed_query(self, query: str) -> list[float]:
        """Async wrapper for embedding — offloads CPU-bound work to thread pool."""
        return await asyncio.to_thread(self._embed_query, query)

    # Key generation

    def make_key(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> str:
        """Generate SHA-256 cache key — identical to exact strategy.

        The semantic strategy still needs deterministic keys for L1/L2
        storage. Uses the FULL normalizer chain (not the light one used
        for embedding).

        Sync — CPU only.

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
            fingerprint = self._key_normalizer.build_cache_fingerprint(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
            )
            if not fingerprint:
                raise ValueError("Empty fingerprint after normalization")
            key = hash_text(fingerprint)
            logger.debug(
                "Semantic key generated: query='%s' → key=%s",
                query[:60],
                key[:16] + "...",
            )
            return key
        except Exception as e:
            logger.exception("Failed to generate semantic cache key")
            raise CacheKeyError(
                strategy="semantic",
                message=f"Key generation failed: {e}",
            ) from e

    # Similarity search

    async def find_similar(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> Optional[SimilarityMatch]:
        """Search for a semantically similar cached query in Qdrant.

        Flow:
            1. Embed the query with BGE via get_embeddings() (~5-15ms)
            2. Search Qdrant for nearest neighbor (~1-5ms)
            3. Apply tiered threshold logic
            4. Return SimilarityMatch with cache_key if above threshold

        Args:
            query: Raw user query.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of system prompt.

        Returns:
            SimilarityMatch if found above threshold, None otherwise.
        """
        if not self._initialized:
            logger.warning("Semantic strategy not initialized, skipping")
            return None

        try:
            query_vector = await self._async_embed_query(query)

            # Skip search for degenerate empty-query vectors
            if all(v == 0.0 for v in query_vector):
                logger.debug("Semantic search skipped — zero vector from empty query")
                return None

            # Qdrant search runs in a thread — client is sync
            results = await asyncio.to_thread(
                self._search_qdrant,
                query_vector=query_vector,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
            )

            if not results:
                logger.debug(
                    "Semantic miss: query='%s', no results above threshold",
                    query[:60],
                )
                return None

            # Apply tiered threshold to the single top result
            top = results[0]
            score = top.score
            cache_key = top.payload.get("cache_key", "")

            if not cache_key:
                logger.warning("Semantic match has empty cache_key, ignoring")
                return None

            tier = self._classify_tier(score)

            if tier == "miss":
                logger.debug(
                    "Semantic miss: query='%s', score=%.4f (below threshold)",
                    query[:60],
                    score,
                )
                return None

            logger.info(
                "Semantic hit: query='%s', score=%.4f, tier=%s, key=%s",
                query[:60],
                score,
                tier,
                cache_key[:16] + "...",
            )

            return SimilarityMatch(
                cache_key=cache_key,
                similarity_score=score,
                tier=tier,
            )
        except Exception as e:
            logger.exception(
                "Semantic find_similar failed: query='%s'", query[:60]
            )
            return None

    def _search_qdrant(
        self,
        query_vector: list[float],
        model_name: str,
        temperature: float,
        system_prompt_hash: str,
    ) -> list:
        """Sync Qdrant search — runs inside asyncio.to_thread().

        Searches for the single nearest neighbor above the high threshold.
        Filters by model_name to prevent cross-model cache pollution.

        Args:
            query_vector: BGE embedding vector.
            model_name: Filter — only match same model.
            temperature: Filter — only match same temperature.
            system_prompt_hash: Filter — only match same system prompt.

        Returns:
            List of ScoredPoint results (0 or 1 items).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must_conditions = [
            FieldCondition(
                key="model_name",
                match=MatchValue(value=model_name.strip().lower()),
            )
        ]

        if system_prompt_hash:
            must_conditions.append(
                FieldCondition(
                    key="system_prompt_hash",
                    match=MatchValue(value=system_prompt_hash),
                )
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=Filter(
                must=must_conditions
            ),
            limit=1,
            score_threshold=self._threshold_high,
        )

        return response.points

    def _classify_tier(self, score: float) -> str:
        """Classify a cosine similarity score into a confidence tier.

        Args:
            score: Cosine similarity (0.0 to 1.0).

        Returns:
            'direct', 'high', or 'miss'.
        """
        if score >= self._threshold_direct:
            return "direct"
        elif score >= self._threshold_high:
            return "high"
        else:
            return "miss"

    # Index entry (write path)

    async def index_entry(
        self,
        query: str,
        cache_key: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> None:
        """Embed a query and upsert into Qdrant for future similarity lookups.

        Called by CacheManager.set() after writing to L1/L2.

        Flow:
            1. Embed query with get_embeddings().embed_query()
            2. Upsert point into Qdrant: (vector, {cache_key, query_text, model_name})

        Args:
            query: The user's prompt that produced the cached response.
            cache_key: The key this entry is stored under in L1/L2.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of system prompt.
        """
        if not self._initialized:
            logger.warning("Semantic strategy not initialized, skipping index")
            return

        try:
            query_vector = await self._async_embed_query(query)

            # Skip indexing for degenerate empty-query vectors
            if all(v == 0.0 for v in query_vector):
                logger.debug("Semantic index skipped — zero vector from empty query")
                return

            payload = {
                "cache_key": cache_key,
                "query_text": query[:500],
                "model_name": model_name.strip().lower(),
                "temperature": float(temperature),
                "system_prompt_hash": system_prompt_hash,
            }

            await asyncio.to_thread(
                self._upsert_point,
                query_vector=query_vector,
                payload=payload,
            )

            logger.debug(
                "Semantic index: query='%s', key=%s",
                query[:60],
                cache_key[:16] + "...",
            )

        except Exception:
            logger.exception(
                "Semantic index_entry failed: query='%s'", query[:60]
            )

    def _upsert_point(self, query_vector: list[float], payload: dict) -> None:
        """Sync Qdrant upsert — runs inside asyncio.to_thread().

        Uses UUID as point ID. Duplicate queries get separate points.
        Qdrant returns nearest by vector similarity, so duplicates are
        harmless — the closest match always wins on search.
        """
        from qdrant_client.models import PointStruct

        point_id = str(uuid.uuid4())

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload=payload,
                )
            ]
        )

    # Helpers

    def get_query_hash(self, query: str) -> str:
        """SHA-256 hash of normalized query for deduplication.

        Args:
            query: Raw user query.

        Returns:
            64-character SHA-256 hex digest.
        """
        normalized = self._key_normalizer.normalize(query)
        return hash_text(normalized)

    async def get_collection_stats(self) -> dict:
        """Return Qdrant collection statistics.

        Returns:
            Dict with collection name, point count, status, and embedding dim.
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            info = await asyncio.to_thread(
                self._client.get_collection, self._collection
            )
            return {
                "collection": self._collection,
                "points_count": info.points_count,
                "status": str(info.status),
                "embedding_dim": self._embedding_dim,
            }
        except Exception as e:
            return {"collection": self._collection, "error": str(e)}

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            if hasattr(self._client, "close"):
                self._client.close()
            logger.info("SemanticCacheStrategy closed")
        except Exception as e:
            logger.exception("Error closing semantic strategy Qdrant client")
