"""
Dense vector retrieval from Qdrant using embedding similarity.

Design:
    Thin wrapper around QdrantStore.similarity_search_with_vectors(). No
    new embedding code — QdrantStore uses get_embeddings() internally.
    Converts LangChain Documents to RetrievedChunks at the retriever
    boundary. MetadataFilters are converted to Qdrant payload filters via
    the shared BaseRetriever.build_qdrant_filter() method.

    Use dense over hybrid when queries and documents share semantic
    vocabulary and exact keyword matching is not critical. Dense is faster
    (~1.5–2x) because it performs only one index lookup.

Chain of Responsibility:
    Instantiated by RAGFactory.create_retriever(mode="dense")
    → called by SimpleRAG.retrieve() / CorrectiveRAG._do_retrieval()
    → QdrantStore.similarity_search_with_vectors()
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


class DenseRetriever(BaseRetriever):
    """Dense-only retrieval using QdrantStore vector similarity.

    Wraps QdrantStore.similarity_search_with_vectors() with MetadataFilter
    conversion and LangChain Document → RetrievedChunk transformation.
    Falls back to similarity_search() when the with-vectors variant is
    unavailable (e.g., non-Qdrant store injected in tests).

    Attributes:
        _store: QdrantStore instance injected via constructor.
    """

    def __init__(self, store: object):
        """Initialize the dense retriever.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
                Must implement async similarity_search_with_vectors(query, k).
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

        Calls QdrantStore.similarity_search_with_vectors() which:
            1. Embeds the query using the shared BGE get_embeddings() instance.
            2. Searches the Qdrant collection for nearest vectors.
            3. Returns LangChain Documents with relevance_score and vector
               in metadata.

        Falls back to similarity_search() when the with-vectors variant
        is not available on the injected store.

        Args:
            query: Search query string.
            top_k: Maximum number of chunks to return (1–50).
            filters: Optional metadata filters for scoped retrieval.

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).
            Empty list if no results are found.

        Raises:
            RAGRetrievalError: If the vector store query fails.
        """
        if not query or not query.strip():
            logger.warning("Empty query received, returning empty results")
            return []

        qdrant_filter = self.build_qdrant_filter(filters)

        start = time.perf_counter()

        try:
            # Prefer the with-vectors variant so MMR can skip re-embedding.
            # Falls back gracefully when the store doesn't expose this method.
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
