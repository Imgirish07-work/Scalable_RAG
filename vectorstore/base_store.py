"""
Abstract base class for all vector store backends.

Defines the async interface that QdrantStore (and any future backends
like Pinecone, Weaviate, pgvector) must implement.

All methods are async — FastAPI ready, event loop never blocked.

Design contract:
    - add_documents() embeds page_content (or embed_content) and stores with metadata
    - similarity_search() returns Documents with clean page_content for LLM use
    - Metadata is NEVER embedded — only stored as payload
    - doc_id and user_id must be present in metadata for every document
    - close() must be called on shutdown to prevent connection leaks

Pipeline position:
    DocumentCleaner → StructurePreserver → Chunker
        → VectorStore.add_documents()     ← write
        → VectorStore.similarity_search() ← read
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Async interface for vector store backends.

    Implementations must handle:
        - Connection management (pooling, reconnection)
        - Collection/index creation
        - Embedding via the configured model
        - Metadata payload storage and filtering
        - Graceful shutdown
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Async initialization — create connections, collections, indexes.

        Must be called once before add_documents() or similarity_search().
        Safe to call multiple times (idempotent).

        Raises:
            Exception: If connection or collection creation fails.
        """
        ...

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents in the vector store.

        Embedding source:
            Uses metadata['embed_content'] if available (richer vector
            with title + section context from Chunker). Falls back to
            page_content if embed_content is not set.

        Metadata requirements:
            doc_id      : str → document identifier (FK to documents table)
            user_id     : str → owner identifier (for per-user filtering)
            source      : str → filename or URL
            page        : int → page number
            chunk_index : int → position within document

        Args:
            documents: List of Document objects from Chunker.

        Returns:
            List of assigned point IDs (one per document).

        Raises:
            Exception: If embedding or storage fails.
        """
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter_user_id: Optional[str] = None,
    ) -> List[Document]:
        """Return top-k semantically similar documents.

        Returned documents have clean page_content suitable for LLM
        consumption (no embed_content prefix). Relevance scores are
        attached to metadata['relevance_score'] when score_threshold
        is provided.

        Args:
            query: Search text to embed and compare.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0.0-1.0).
                             None = return all top-k without filtering.
            filter_user_id: When set, returns only that user's documents.
                            None = search entire collection.

        Returns:
            List of matching Documents with metadata.

        Raises:
            Exception: If search fails.
        """
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """Permanently delete the entire collection/index.

        Use with extreme caution — this is irreversible.
        All vectors and metadata are lost.

        Raises:
            Exception: If deletion fails.
        """
        ...

    @abstractmethod
    async def get_collection_stats(self) -> dict:
        """Return collection statistics for observability.

        Returns:
            Dict with at minimum: backend, collection_name,
            document_count, embedding_model, search_mode.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Graceful shutdown — release connections and resources.

        Must be called on application shutdown to prevent leaks.
        Safe to call multiple times.
        """
        ...