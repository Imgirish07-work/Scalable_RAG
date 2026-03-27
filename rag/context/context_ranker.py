"""
Context ranker — post-retrieval reranking and diversification.

Design:
    - Takes retrieved chunks and reranks them before context assembly.
    - Supports three strategies (selected via RAGConfig.rerank_strategy):
        - 'none': Pass through unchanged (retrieval order preserved).
        - 'mmr': Maximal Marginal Relevance — balances relevance with
          diversity to reduce redundant chunks in the context.
        - 'cross_encoder': Placeholder for future cross-encoder reranking.
          Falls back to 'none' with a warning until implemented.
    - MMR is the default strategy. It's free (no extra LLM or API calls)
      and prevents the common failure mode where top-5 retrieval results
      are all from the same paragraph, wasting context window tokens on
      near-duplicate information.

Why MMR matters:
    Dense retrieval often returns chunks that are semantically very similar
    to each other (e.g., 3 out of 5 chunks from the same section). MMR
    re-scores chunks to penalize similarity to already-selected chunks,
    promoting diversity. The result: more information per token of context.

MMR formula:
    score(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)

    lambda = 1.0 → pure relevance (same as 'none')
    lambda = 0.0 → pure diversity (ignores relevance)
    lambda = 0.7 → recommended default (relevance-weighted with diversity)

Integration:
    - RetrievedChunk from rag/models/rag_response.py
    - get_embeddings() from vectorstore/embeddings.py for MMR similarity
    - Called by BaseRAG.rank() between retrieve() and assemble_context()
"""

import asyncio
from typing import Literal

import numpy as np

from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# Default MMR lambda — relevance vs diversity tradeoff
_DEFAULT_MMR_LAMBDA = 0.7


class ContextRanker:
    """Post-retrieval reranking with pluggable strategies.

    Reranks chunks after retrieval to optimize context quality before
    assembly. Strategy is selected at construction time but can be
    overridden per-call.

    Attributes:
        _strategy: Default reranking strategy.
        _mmr_lambda: Relevance-diversity tradeoff for MMR (0.0-1.0).
        _embeddings_fn: Callable that returns the embedding model.
            Used by MMR to compute inter-chunk similarity.
    """

    def __init__(
        self,
        strategy: str = "mmr",
        mmr_lambda: float = _DEFAULT_MMR_LAMBDA,
        embeddings_fn: object | None = None,
    ) -> None:
        """Initialize context ranker.

        Args:
            strategy: Reranking strategy: 'none', 'mmr', 'cross_encoder'.
            mmr_lambda: Relevance-diversity tradeoff for MMR.
                1.0 = pure relevance, 0.0 = pure diversity.
                Default 0.7 gives strong relevance with diversity bonus.
            embeddings_fn: Callable returning embedding model with
                embed_documents() method. Required for MMR strategy.
                Pass get_embeddings from vectorstore/embeddings.py.

        Raises:
            ValueError: If mmr_lambda is outside 0.0-1.0 range.
        """
        if not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError(
                f"mmr_lambda must be between 0.0 and 1.0. Got {mmr_lambda}"
            )

        self._strategy = strategy.strip().lower()
        self._mmr_lambda = mmr_lambda
        self._embeddings_fn = embeddings_fn

        logger.info(
            "ContextRanker initialized | strategy=%s | mmr_lambda=%.2f",
            self._strategy,
            self._mmr_lambda,
        )

    async def rank(
        self,
        chunks: list[RetrievedChunk],
        query: str,
        strategy: str | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using the configured strategy.

        Args:
            chunks: Retrieved chunks to rerank, ordered by retrieval score.
            query: Original query string (used by MMR for query embedding).
            strategy: Override the default strategy for this call.
                If None, uses the strategy set at construction time.

        Returns:
            Reranked list of RetrievedChunk. Same chunks, different order.
            Empty input returns empty output.
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks

        active_strategy = (strategy or self._strategy).strip().lower()

        if active_strategy == "none":
            return self._rank_none(chunks)

        if active_strategy == "mmr":
            return await self._rank_mmr(chunks, query)

        if active_strategy == "cross_encoder":
            logger.warning(
                "cross_encoder reranking not yet implemented, "
                "falling back to none"
            )
            return self._rank_none(chunks)

        logger.warning(
            "Unknown rerank strategy '%s', falling back to none",
            active_strategy,
        )
        return self._rank_none(chunks)

    def _rank_none(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Pass-through ranking — preserves retrieval order.

        Args:
            chunks: Retrieved chunks in original order.

        Returns:
            Same list unchanged.
        """
        logger.info("Rerank strategy=none | chunks=%d | order preserved", len(chunks))
        return chunks

    async def _rank_mmr(
        self,
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance reranking.

        Iteratively selects chunks that are both relevant to the query
        AND different from already-selected chunks.

        Algorithm:
            1. Embed all chunk contents + the query.
            2. Select the most relevant chunk first.
            3. For each remaining slot, score candidates as:
               lambda * relevance - (1-lambda) * max_similarity_to_selected
            4. Select the highest-scoring candidate.

        Falls back to no reranking if embeddings are unavailable.

        Args:
            chunks: Retrieved chunks to rerank.
            query: Original query for relevance scoring.

        Returns:
            Reranked chunks with diversity-relevance balance.
        """
        if self._embeddings_fn is None:
            logger.warning(
                "MMR requested but no embeddings function provided, "
                "falling back to none"
            )
            return self._rank_none(chunks)

        try:
            embeddings_model = self._embeddings_fn()

            # Embed all chunks and the query in one batch
            texts = [chunk.content for chunk in chunks]
            texts_with_query = texts + [query]

            # embed_documents is sync (CPU-bound), wrap in to_thread
            all_embeddings = await asyncio.to_thread(
                embeddings_model.embed_documents, texts_with_query
            )

            chunk_embeddings = np.array(all_embeddings[:-1])
            query_embedding = np.array(all_embeddings[-1])

            # Compute relevance scores (cosine similarity to query)
            relevance_scores = self._cosine_similarity_batch(
                query_embedding, chunk_embeddings
            )

            # Run MMR selection
            selected_indices = self._mmr_select(
                relevance_scores=relevance_scores,
                chunk_embeddings=chunk_embeddings,
                n_results=len(chunks),
            )

            reranked = [chunks[i] for i in selected_indices]

            logger.info(
                "MMR reranking complete | chunks=%d | lambda=%.2f",
                len(reranked),
                self._mmr_lambda,
            )

            return reranked

        except Exception as exc:
            logger.warning(
                "MMR reranking failed, falling back to none | error=%s",
                str(exc),
            )
            return self._rank_none(chunks)

    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        chunk_embeddings: np.ndarray,
        n_results: int,
    ) -> list[int]:
        """Core MMR selection algorithm.

        Greedily selects chunks that maximize the MMR criterion:
            score = lambda * relevance - (1-lambda) * max_sim_to_selected

        Args:
            relevance_scores: Array of query-chunk cosine similarities.
            chunk_embeddings: Matrix of chunk embedding vectors.
            n_results: Number of chunks to select.

        Returns:
            List of selected indices in MMR order.
        """
        n_chunks = len(relevance_scores)
        selected = []
        remaining = list(range(n_chunks))

        # First selection: highest relevance
        best_idx = int(np.argmax(relevance_scores))
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Subsequent selections: MMR criterion
        for _ in range(min(n_results - 1, len(remaining))):
            if not remaining:
                break

            best_score = -float("inf")
            best_candidate = remaining[0]

            for candidate in remaining:
                # Relevance component
                relevance = float(relevance_scores[candidate])

                # Diversity component — max similarity to any selected chunk
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._cosine_similarity(
                        chunk_embeddings[candidate],
                        chunk_embeddings[sel_idx],
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr_score = (
                    self._mmr_lambda * relevance
                    - (1 - self._mmr_lambda) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _cosine_similarity_batch(
        self,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between a query and multiple documents.

        Args:
            query_vec: Query embedding vector (1D array).
            doc_vecs: Document embedding matrix (2D, one row per doc).

        Returns:
            1D array of cosine similarity scores.
        """
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(doc_vecs))

        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        # Avoid division by zero for zero-norm documents
        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)

        return np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity score (0.0-1.0 for normalized embeddings).
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))