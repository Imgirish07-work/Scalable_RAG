"""
Dense retriever — wraps QdrantStore for dense-only vector search.

Design:
    - Thin wrapper around the existing QdrantStore.similarity_search().
    - No new embedding code — QdrantStore uses get_embeddings() internally.
    - Converts LangChain Documents → RetrievedChunks at the boundary.
    - MetadataFilters converted to Qdrant payload filters via base class.

When to use dense vs hybrid:
    - Dense: Good default. Works well when user queries and document
      language overlap semantically. Faster than hybrid.
    - Hybrid: Better when exact keywords matter (product codes, legal
      terms, medical terminology). Adds SPLADE sparse matching.

The retrieval_mode field in RAGConfig controls which retriever
RAGFactory injects into BaseRAG.
"""

import time

from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from rag.retrieval.base_retriever import BaseRetriever
from rag.exceptions.rag_exceptions import RAGRetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)

class DenseRetriever(BaseRetriever):
    """Dense-only retrieval via existing QdrantStore.

    Wraps QdrantStore.similarity_search() with MetadataFilter
    conversion and LangChain → RetrievedChunk transformation.

    Attributes:
        _store: QdrantStore instance injected via constructor.
    """
    def __init__(self, store: object):
        """Initialize dense retriever.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
                Must implement async similarity_search(query, k, filter).
        """
        super().__init__(store)
        logger.info("DenseRetriever initialized")

    @property
    def retriever_type(self) -> str:
        """Return retriever type identifier.

        Returns:
            The string 'dense'.
        """
        return "dense"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using dense vector similarity.

        Calls QdrantStore.similarity_search() which:
            1. Embeds the query using get_embeddings() (shared BGE instance)
            2. Searches the Qdrant collection for nearest vectors
            3. Returns LangChain Documents with relevance_score in metadata

        Args:
            query: Search query string.
            top_k: Maximum number of chunks to return (1-50).
            filters: Optional metadata filters for scoped retrieval.

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).
            Empty list if no results found.

        Raises:
            RAGRetrievalError: If the vector store query fails.
        """
        if not query or not query.strip():
            logger.warning("Empty query received, returning empty results")
            return []

        qdrant_filter = self.build_qdrant_filter(filters)

        start = time.perf_counter()
        
        try:
            # Use with-vectors variant so MMR can skip re-embedding entirely.
            # Falls back gracefully to similarity_search if the method is
            # not available (e.g. non-Qdrant store injected in tests).
            if hasattr(self._store, "similarity_search_with_vectors"):
                docs = await self._store.similarity_search_with_vectors(
                    query=query,
                    k=top_k,
                )
            else:
                docs = await self._store.similarity_search(
                    query=query,
                    k=top_k,
                    score_threshold=0.0,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000

            chunks = self._convert_documents(docs)

            logger.info(
                "Dense retrieval complete | query_len=%d | top_k=%d | "
                "results=%d | latency=%.1f ms",
                len(query),
                top_k,
                len(chunks),
                elapsed_ms,
            )

            return chunks
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Dense retrieval failed | query_len=%d | top_k=%d | "
                "latency=%.1f ms | error=%s",
                len(query),
                top_k,
                elapsed_ms,
                str(e),
            )
            raise RAGRetrievalError(
                f"Dense retrieval failed: {e}",
                details={
                    "retriever": self.retriever_type,
                    "query_length": len(query),
                    "top_k": top_k,
                    "has_filters": filters is not None,
                },
            ) from e

