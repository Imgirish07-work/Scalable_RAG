"""Abstract async interface for vector store backends."""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Async interface contract for all vector store backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create connections, collections, and indexes."""
        ...

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents in the vector store."""
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter_user_id: Optional[str] = None,
        filter_collection: Optional[str] = None,
    ) -> List[Document]:
        """Return top-k semantically similar documents for the given query."""
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """Permanently delete the entire collection."""
        ...

    @abstractmethod
    async def get_collection_stats(self) -> dict:
        """Return collection statistics for observability."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release connections and resources on shutdown."""
        ...
