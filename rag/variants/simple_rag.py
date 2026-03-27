"""
SimpleRAG — baseline RAG variant.

Design:
    - Overrides ONLY retrieve(). Everything else uses BaseRAG defaults:
      pre_process (normalize/refine), rank (MMR), assemble_context
      (token-bounded), generate (grounded LLM call), cache (automatic).
    - This is the production default. For 80% of queries — factual
      lookups, document Q&A, summarization — SimpleRAG is the right
      answer. It's not a stepping stone; it's the workhorse.
    - Must pass tests before any other variant is built. Every bug
      found here (cache integration, doc format, prompt template shape)
      is cheaper to fix now than after CorrectiveRAG is built on top.

Pipeline flow (inherited from BaseRAG.query()):
    1. Cache check       → hit? return immediately
    2. pre_process()     → normalize query, resolve pronouns if history
    3. retrieve()        → ★ SimpleRAG: call retriever directly
    4. rank()            → MMR diversification
    5. assemble_context() → token-bounded assembly
    6. generate()        → grounded LLM call with context
    7. Cache write       → store for future hits

Cost: 1 retrieval call + 1 LLM call (+ 1 LLM call for query
refinement only if conversation_history is present).

Integration:
    - BaseRetriever.retrieve() from rag/retrieval/
    - All other steps inherited from BaseRAG
"""

from rag.base_rag import BaseRAG
from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from rag.retrieval.base_retriever import BaseRetriever
from rag.exceptions.rag_exceptions import RAGRetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class SimpleRAG(BaseRAG):
    """Baseline RAG variant — direct retrieval, no evaluation or retry.

    Overrides only retrieve(). Delegates directly to the injected
    BaseRetriever. The simplest possible RAG pipeline.

    When to use:
        - Direct factual questions ("What does section 4.2 say?")
        - Document summarization ("Summarize the Q3 report")
        - Concept explanations ("Explain the CAP theorem")
        - Any query where retrieval quality is expected to be good

    When to use CorrectiveRAG instead:
        - High-stakes queries where wrong answers are costly
        - Domains where retrieval often returns similar-but-wrong chunks
        - Queries where you need confidence that retrieval was relevant
    """

    @property
    def variant_name(self) -> str:
        """Return the variant identifier.

        Returns:
            The string 'simple'.
        """
        return "simple"

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks via direct retriever call.

        No evaluation, no retry, no query rewriting. Delegates
        directly to the injected BaseRetriever.

        Args:
            query: Processed query string (output of pre_process).
            top_k: Maximum number of chunks to retrieve.
            filters: Optional metadata filters from RAGConfig.

        Returns:
            List of RetrievedChunk ordered by relevance (highest first).

        Raises:
            RAGRetrievalError: If the retriever fails.
        """
        logger.info(
            "SimpleRAG retrieving | query_len=%d | top_k=%d | "
            "has_filters=%s",
            len(query),
            top_k,
            filters is not None,
        )

        chunks = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        logger.info(
            "SimpleRAG retrieval complete | results=%d | "
            "top_score=%.3f | bottom_score=%.3f",
            len(chunks),
            chunks[0].relevance_score if chunks else 0.0,
            chunks[-1].relevance_score if chunks else 0.0,
        )

        return chunks