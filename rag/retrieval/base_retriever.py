"""
Abstract base class defining the retriever interface (Strategy pattern).

Design:
    Strategy pattern: BaseRAG.__init__ takes a BaseRetriever. The retriever
    is swappable at runtime and mockable for tests — variants never know the
    concrete type. DenseRetriever and HybridRetriever both implement this
    contract by wrapping the existing QdrantStore. MetadataFilter conversion
    from RAGRequest format to Qdrant payload filter format is shared logic
    in the base class, keeping it out of each concrete retriever.

    Retrieval mode (dense vs hybrid) is orthogonal to RAG variant.
    Strategy decouples the two axes — changing retrieval doesn't touch variant
    code.

Chain of Responsibility:
    Instantiated by RAGFactory.create_retriever() → injected into BaseRAG
    → called by BaseRAG.retrieve() hook → returns list[RetrievedChunk]
    to the rank step.

Dependencies:
    rag.models.rag_request (MetadataFilter)
    rag.models.rag_response (RetrievedChunk)
"""

from abc import ABC, abstractmethod

from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# Maps RAGRequest filter operator names to Qdrant payload filter field names
_QDRANT_OPERATOR_MAP = {
    "eq": "value",
    "neq": "value",
    "gt": "gt",
    "gte": "gte",
    "lt": "lt",
    "lte": "lte",
    "in": "value",
}


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies.

    Subclasses must implement:
        - retrieve() — fetch relevant chunks from the vector store.
        - retriever_type (property) — return a string identifier.

    Shared logic provided:
        - build_qdrant_filter() — convert MetadataFilter list to Qdrant format.
        - _convert_documents() — convert LangChain Documents to RetrievedChunks.

    Attributes:
        _store: QdrantStore instance injected via constructor.
    """

    def __init__(self, store: object):
        """Initialize the retriever with a vector store.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
                Typed as object to avoid circular imports — validated
                at runtime via duck typing.
        """
        self._store = store

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
        user_id: str = "",
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from the vector store.

        Args:
            query: Search query string.
            top_k: Maximum number of chunks to return.
            filters: Optional metadata filters for scoped retrieval.
            user_id: Authenticated user ID for per-user document filtering.
                Empty string disables user scoping (returns all docs).

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).
            Empty list if no results found.

        Raises:
            RAGRetrievalError: If retrieval fails after retries.
        """
        ...

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        """Return retriever type identifier.

        Returns:
            String identifier e.g. 'dense', 'hybrid'.
        """

    # Shared methods

    def build_qdrant_filter(
        self, filters: list[MetadataFilter] | None,
    ) -> dict | None:
        """Convert a MetadataFilter list to Qdrant payload filter format.

        Multiple filters are combined with AND logic (Qdrant 'must' clause).

        Qdrant filter format:
            {
                "must": [
                    {"key": "source_file", "match": {"value": "report.pdf"}},
                    {"key": "page_number", "range": {"gte": 5}},
                ]
            }

        Args:
            filters: List of MetadataFilter from RAGRequest, or None.

        Returns:
            Qdrant filter dict, or None if no filters are provided.
        """
        if not filters:
            return None

        conditions = []
        for f in filters:
            condition = self._build_single_condition(f)
            if condition is not None:
                conditions.append(condition)

        if not conditions:
            return None

        return {"must": conditions}

    def _build_single_condition(self, f: MetadataFilter) -> dict | None:
        """Convert a single MetadataFilter to a Qdrant condition dict.

        Args:
            f: Single MetadataFilter instance.

        Returns:
            Qdrant condition dict, or None if the operator is unrecognised.
        """
        op = f.operator

        if op == "eq":
            return {"key": f.field, "match": {"value": f.value}}

        if op == "neq":
            # Qdrant uses must_not for negation — handled at caller level
            # For simplicity, wrap as a match that gets negated
            return {"key": f.field, "match": {"value": f.value}, "_negate": True}

        if op == "in":
            # Qdrant match accepts list for "any of" semantics
            if isinstance(f.value, list):
                return {"key": f.field, "match": {"any": f.value}}
            return {"key": f.field, "match": {"value": f.value}}

        if op in ("gt", "gte", "lt", "lte"):
            return {"key": f.field, "range": {op: f.value}}

        logger.warning(
            "Unknown filter operator, skipping | field=%s | operator=%s",
            f.field,
            op,
        )
        return None

    def _convert_documents(self, docs: list) -> list[RetrievedChunk]:
        """Convert LangChain Documents to RetrievedChunk instances.

        Extracts relevance_score from metadata when present (set by
        QdrantStore after a scored search). Also forwards the pre-fetched
        embedding vector when available so MMR can use it directly
        without re-embedding.

        Args:
            docs: List of LangChain Document objects from QdrantStore.

        Returns:
            List of RetrievedChunk instances.
        """
        chunks = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            score = meta.get("relevance_score", 0.0)

            # Clamp score to the valid range — Qdrant occasionally returns
            # values slightly outside [0, 1] due to floating-point rounding.
            if isinstance(score, (int, float)):
                score = max(0.0, min(1.0, float(score)))
            else:
                score = 0.0

            # Pass pre-fetched vector if present; MMR falls back to re-embedding
            vector = meta.get("vector")

            chunk = RetrievedChunk.from_document(
                doc, relevance_score=score, vector=vector
            )
            chunks.append(chunk)

        return chunks

    def __repr__(self) -> str:
        """Human-readable representation for logging.

        Returns:
            String like 'DenseRetriever(type=dense)'.
        """
        return f"{self.__class__.__name__}(type={self.retriever_type})"
