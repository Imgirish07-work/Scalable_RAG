"""
SimpleRAG — the baseline single-pass RAG variant.

Design:
    Overrides only retrieve(). All other pipeline steps use BaseRAG defaults:
    pre_process (normalize/refine via conversation context), rank (MMR),
    assemble_context (token-bounded), generate (grounded LLM call), and
    cache (automatic on both read and write).

    SimpleRAG is the production default — the workhorse for 80% of queries.
    Factual lookups, document Q&A, and summarization all perform well here.
    It is not a stepping stone; it is a complete, production-ready variant.

    Minimum cost: 1 retrieval call + 1 LLM generation call. An extra LLM
    call is incurred only when conversation_history is present (query
    refinement in pre_process).

Chain of Responsibility:
    Created by RAGFactory → BaseRAG.query() calls retrieve()
    → delegates directly to BaseRetriever.retrieve()
    → returns list[RetrievedChunk] to the rank step.

Dependencies:
    rag.base_rag (BaseRAG)
    rag.retrieval.base_retriever (BaseRetriever)
    rag.exceptions.rag_exceptions (RAGRetrievalError)
"""

from rag.base_rag import BaseRAG
from rag.models.rag_request import MetadataFilter
from rag.models.rag_response import RetrievedChunk
from rag.retrieval.base_retriever import BaseRetriever
from rag.exceptions.rag_exceptions import RAGRetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class SimpleRAG(BaseRAG):
    """Baseline RAG variant — direct retrieval with no evaluation or retry.

    Overrides only retrieve() to delegate directly to the injected
    BaseRetriever. The simplest possible RAG pipeline.

    When to use SimpleRAG:
        - Direct factual questions ("What does section 4.2 say?")
        - Document summarization ("Summarize the Q3 report")
        - Concept explanations ("Explain the CAP theorem")
        - Any query where retrieval quality is expected to be good

    For complex multi-part questions, the agent layer (AgentOrchestrator)
    decomposes them into sub-queries — each sub-query uses this variant
    for retrieval-only execution.
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
        request=None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks via direct retriever call.

        No evaluation, no retry, no query rewriting. Delegates directly to
        the injected BaseRetriever (dense or hybrid). The request parameter
        is accepted for interface consistency but not used by SimpleRAG.

        Args:
            query: Processed query string (output of pre_process).
            top_k: Maximum number of chunks to retrieve.
            filters: Optional metadata filters from RAGConfig.
            request: Unused. Accepted for BaseRAG interface compliance.

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
