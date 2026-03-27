"""
Hybrid retriever — wraps QdrantStore for dense + SPLADE sparse search.

Design:
    - Combines dense vector similarity (BGE embeddings) with SPLADE sparse
      matching for keyword-aware retrieval.
    - Uses QdrantStore's existing hybrid search capability — no new
      embedding or sparse encoding logic.
    - Falls back to dense-only search if sparse component fails, logging
      a warning instead of raising. This makes hybrid retrieval resilient
      to SPLADE initialization failures.

When to use:
    - Queries containing exact terms that must match (product IDs, legal
      clause numbers, medical codes like ICD-10, chemical formulas).
    - Queries where semantic similarity alone misses keyword-critical
      documents (e.g., "HIPAA section 164.502" vs "privacy rules").
    - Documents with specialized vocabulary where dense embeddings
      may not capture domain-specific term importance.

Cost:
    - ~1.5-2x latency vs dense-only (two index lookups + fusion).
    - No extra embedding model — SPLADE uses FastEmbedSparse already
      initialized in QdrantStore.

Integration:
    - QdrantStore must have been initialized with hybrid=True and sparse
      embeddings configured. If not, hybrid search falls back to dense.
    - RAGConfig.retrieval_mode="hybrid" selects this retriever via RAGFactory.
"""

import time

from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from rag.retrieval.base_retriever import BaseRetriever
from rag.exceptions.rag_exceptions import RAGRetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridRetriever(BaseRetriever):
    """Hybrid dense + SPLADE retrieval via existing QdrantStore.

    Wraps QdrantStore.hybrid_search() with fallback to dense-only
    if the sparse component is unavailable or fails.

    Attributes:
        _store: QdrantStore instance injected via constructor.
        _dense_weight: Weight for dense scores in fusion (0.0-1.0).
        _sparse_weight: Weight for sparse scores in fusion (0.0-1.0).
    """
    def __init__(
        self,
        store: object,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """Initialize hybrid retriever.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
                Must implement async hybrid_search() and similarity_search().
            dense_weight: Weight for dense vector scores in fusion.
                Higher = more semantic matching. Default 0.7.
            sparse_weight: Weight for sparse SPLADE scores in fusion.
                Higher = more keyword matching. Default 0.3.

        Raises:
            ValueError: If weights are negative or don't sum to ~1.0.
        """
        super().__init__(store)
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError(
                f"Weights must be non-negative. "
                f"Got dense={dense_weight}, sparse={sparse_weight}"
            )
        weight_sum = dense_weight + sparse_weight
        if not 0.9 <= weight_sum <= 1.1:
            raise ValueError(
                f"Weights should sum to ~1.0. "
                f"Got dense={dense_weight} + sparse={sparse_weight} = {weight_sum}"
            )
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

        logger.info(
            "HybridRetriever initialized | dense_weight=%.2f | sparse_weight=%.2f",
            self._dense_weight,
            self._sparse_weight,
        )

    @property
    def retriever_type(self) -> str:
        """Return retriever type identifier.

        Returns:
            The string 'hybrid'.
        """
        return "hybrid"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using dense + SPLADE hybrid search.

        Attempts hybrid search first. Falls back to dense-only if:
            - QdrantStore doesn't have hybrid_search() method
            - Sparse embeddings are not initialized
            - Hybrid search raises an exception

        Args:
            query: Search query string.
            top_k: Maximum number of chunks to return (1-50).
            filters: Optional metadata filters for scoped retrieval.

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).
            Empty list if no results found.

        Raises:
            RAGRetrievalError: If both hybrid and dense fallback fail.
        """
        if not query or not query.strip():
            logger.warning("Empty query received, returning empty results")
            return []

        qdrant_filter = self.build_qdrant_filter(filters)

        start = time.perf_counter()

        docs = await self._try_hybrid_search(query, top_k, qdrant_filter)

        if docs is None:
            # Hybrid unavailable or failed — fall back to dense
            docs = await self._fallback_dense_search(query, top_k, qdrant_filter)

        elapsed_ms = (time.perf_counter() - start)*1000

        chunks  = self._convert_documents(docs)

        logger.info(
            "Hybrid retrieval complete | query_len=%d | top_k=%d | "
            "results=%d | latency=%.1f ms",
            len(query),
            top_k,
            len(chunks),
            elapsed_ms,
        )

        return chunks

    async def _try_hybrid_search(
        self, 
        query: str,
        top_k: int,
        qdrant_filter: dict | None,
    ) -> list | None:
        """Attempt hybrid search via QdrantStore.

        Args:
            query: Search query string.
            top_k: Maximum results.
            qdrant_filter: Qdrant payload filter dict or None.

        Returns:
            List of LangChain Documents, or None if hybrid is unavailable.
        """

        if not hasattr(self._store, "hybrid_search"):
            logger.warning(
                "QdrantStore does not have hybrid_search method, "
                "falling back to dense"
            )
            return None

        try:
            docs = await self._store.hybrid_search(
                query=query,
                k=top_k,
                filter=qdrant_filter,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
            )
            return docs
        except Exception as e:
            logger.warning(
                "Hybrid search failed, falling back to dense | error=%s",
                str(e),
            )
            return None

    async def _fallback_dense_search(
        self,
        query: str,
        top_k: int,
        qdrant_filter: dict | None,
    ) -> list:
        """Fallback to dense-only search when hybrid is unavailable.

        Args:
            query: Search query string.
            top_k: Maximum results.
            qdrant_filter: Qdrant payload filter dict or None.

        Returns:
            List of LangChain Documents.

        Raises:
            RAGRetrievalError: If dense search also fails.
        """

        try:
            docs = await self._store.similarity_search(
                query=query,
                k=top_k,
                filter=qdrant_filter,
            )
            logger.info(
                "Dense fallback succeeded | results=%d",
                len(docs),
            )
            return docs
        except Exception as e:
            logger.error(
                "Dense fallback also failed | error=%s",
                str(e),
            )
            raise RAGRetrievalError(
                f"Both hybrid and dense retrieval failed: {e}",
                details={
                    "retriever": self.retriever_type,
                    "query_length": len(query),
                    "top_k": top_k,
                    "has_filters": qdrant_filter is not None,
                },
            ) from e