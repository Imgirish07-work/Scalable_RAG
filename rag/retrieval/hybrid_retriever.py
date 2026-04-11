"""
Hybrid retrieval combining dense vectors and SPLADE sparse vectors.

Design:
    Wraps QdrantStore.hybrid_search_with_vectors() to combine dense
    embedding similarity (BGE) with SPLADE sparse keyword matching via
    Reciprocal Rank Fusion (RRF). Falls back to dense-only if the sparse
    component is unavailable or raises — this makes hybrid retrieval
    resilient to SPLADE initialization failures without surfacing them
    to the caller.

    Use hybrid over dense when queries contain exact terms that must match
    (product codes, legal clause numbers, medical codes like ICD-10,
    chemical formulas) or when domain-specific vocabulary reduces the
    quality of dense embeddings alone.

    Cost: ~1.5–2x latency vs dense-only due to two index lookups + fusion.
    No extra embedding model — SPLADE uses FastEmbedSparse already initialized
    in QdrantStore.

Chain of Responsibility:
    Instantiated by RAGFactory.create_retriever(mode="hybrid")
    → called by BaseRAG variants via retrieve()
    → QdrantStore.hybrid_search_with_vectors() (primary)
    → QdrantStore.similarity_search_with_vectors() (fallback)
    → returns list[RetrievedChunk].

Dependencies:
    rag.retrieval.base_retriever (BaseRetriever)
    rag.exceptions.rag_exceptions (RAGRetrievalError)
    vectorstore.qdrant_store (QdrantStore, via duck typing)
"""

import time

from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from rag.retrieval.base_retriever import BaseRetriever
from rag.exceptions.rag_exceptions import RAGRetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid dense + SPLADE retrieval via QdrantStore.

    Wraps QdrantStore.hybrid_search_with_vectors() with automatic fallback
    to dense-only search if the sparse component is unavailable or fails.

    Attributes:
        _store: QdrantStore instance injected via constructor.
        _dense_weight: Weight applied to dense scores during fusion (0.0–1.0).
        _sparse_weight: Weight applied to sparse scores during fusion (0.0–1.0).
    """

    def __init__(
        self,
        store: object,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """Initialize the hybrid retriever.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
                Must implement async hybrid_search_with_vectors() and
                similarity_search_with_vectors() for the dense fallback.
            dense_weight: Fusion weight for dense vector scores.
                Higher values favour semantic matching. Default 0.7.
            sparse_weight: Fusion weight for sparse SPLADE scores.
                Higher values favour exact keyword matching. Default 0.3.

        Raises:
            ValueError: If any weight is negative, or if they do not sum to ~1.0.
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
        """Retrieve chunks using dense + SPLADE hybrid search.

        Attempts hybrid search first. Falls back to dense-only when:
            - QdrantStore does not have hybrid_search_with_vectors().
            - Sparse embeddings are not initialized.
            - Hybrid search raises any exception.

        Args:
            query: Search query string.
            top_k: Maximum number of chunks to return (1–50).
            filters: Optional metadata filters for scoped retrieval.

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).
            Empty list if no results are found.

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
            # Hybrid unavailable or failed — fall back to dense-only
            docs = await self._fallback_dense_search(query, top_k, qdrant_filter)

        elapsed_ms = (time.perf_counter() - start) * 1000

        chunks = self._convert_documents(docs)

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
            top_k: Maximum results to return.
            qdrant_filter: Qdrant payload filter dict or None.

        Returns:
            List of LangChain Documents, or None if hybrid is unavailable
            or raises.
        """
        # Respect the store's configured search mode — if it was initialised in
        # dense-only mode, skip the hybrid path entirely so we never pay for a
        # doomed prefetch attempt followed by a fallback penalty.
        store_mode = getattr(self._store, "search_mode", "hybrid")
        if store_mode == "dense":
            logger.debug("Store search_mode=dense — routing directly to dense search")
            return None

        if not hasattr(self._store, "hybrid_search_with_vectors"):
            logger.warning(
                "QdrantStore does not have hybrid_search_with_vectors method, "
                "falling back to dense"
            )
            return None

        try:
            docs = await self._store.hybrid_search_with_vectors(
                query=query,
                k=top_k,
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
        """Fall back to dense-only search when hybrid is unavailable.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            qdrant_filter: Qdrant payload filter dict or None.

        Returns:
            List of LangChain Documents from dense search.

        Raises:
            RAGRetrievalError: If dense search also fails.
        """
        try:
            docs = await self._store.similarity_search_with_vectors(
                query=query,
                k=top_k,
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
