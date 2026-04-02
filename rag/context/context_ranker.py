"""
Context ranker — post-retrieval reranking and diversification.

Strategies:
    none         — pass-through, retrieval order preserved.
    mmr          — Maximal Marginal Relevance: diversity + relevance balance.
    cross_encoder — ms-marco-MiniLM reranker for accurate relevance scoring,
                   followed by MMR for diversity on the final set.

MMR bug fix (vs original implementation):
    Original: re-embedded ALL chunk texts + query on every query call (~3s).
    Fixed:    uses chunk.relevance_score (already the cosine-sim from Qdrant)
              as the query-chunk similarity — zero re-embedding for that step.
              Only embeds chunks for inter-chunk similarity (the part MMR
              actually needs embeddings for), saving the query embed entirely.

Cross-encoder flow:
    retrieve top RERANKER_COARSE_TOP_K (e.g. 10)
        → cross-encoder scores all (query, chunk) pairs
        → returns top_k with updated relevance_score
        → MMR applied on top_k for diversity (fast, pre-computed embeddings)
"""

import asyncio
from typing import List

import numpy as np

from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MMR_LAMBDA = 0.7


class ContextRanker:
    """
    Post-retrieval reranking with pluggable strategies.

    Attributes:
        _strategy:      Default reranking strategy.
        _mmr_lambda:    Relevance-diversity tradeoff for MMR (0.0-1.0).
        _embeddings_fn: Callable returning the embedding model (for MMR).
        _reranker:      Optional CrossEncoderReranker instance.
        _top_k:         Final number of chunks to return.
    """

    def __init__(
        self,
        strategy: str = "mmr",
        mmr_lambda: float = _DEFAULT_MMR_LAMBDA,
        embeddings_fn: object | None = None,
        reranker: object | None = None,
        top_k: int = 5,
    ) -> None:
        if not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError(f"mmr_lambda must be 0.0-1.0, got {mmr_lambda}")

        self._strategy      = strategy.strip().lower()
        self._mmr_lambda    = mmr_lambda
        self._embeddings_fn = embeddings_fn
        self._reranker      = reranker
        self._top_k         = top_k

        logger.info(
            "ContextRanker initialized | strategy=%s | mmr_lambda=%.2f | "
            "reranker=%s",
            self._strategy,
            self._mmr_lambda,
            "enabled" if reranker else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def retrieval_top_k(self) -> int:
        """How many chunks to fetch from the vector store before reranking.

        For cross_encoder: returns RERANKER_COARSE_TOP_K (e.g. 10) so the
        reranker has enough candidates to work with.
        For all other strategies: returns the final top_k directly.
        """
        if self._strategy == "cross_encoder" and self._reranker is not None:
            from config.settings import settings
            return getattr(settings, "RERANKER_COARSE_TOP_K", self._top_k * 2)
        return self._top_k

    async def rank(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        strategy: str | None = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using the configured strategy.

        Args:
            chunks:   Retrieved chunks ordered by Qdrant score.
            query:    Original query string.
            strategy: Override default strategy for this call.

        Returns:
            Reranked list — same chunks, different order / scores.
        """
        if not chunks:
            return []
        if len(chunks) == 1:
            return chunks

        active = (strategy or self._strategy).strip().lower()

        if active == "none":
            return self._rank_none(chunks)

        if active == "mmr":
            return await self._rank_mmr(chunks, query)

        if active == "cross_encoder":
            return await self._rank_cross_encoder(chunks, query)

        logger.warning("Unknown strategy '%s', falling back to none", active)
        return self._rank_none(chunks)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _rank_none(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        logger.info("Rerank=none | chunks=%d | order preserved", len(chunks))
        return chunks

    async def _rank_cross_encoder(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """
        Cross-encoder rerank → MMR for diversity.

        Step 1: Cross-encoder scores all (query, chunk) pairs and returns
                top_k with updated relevance_score.
        Step 2: MMR on top_k to remove near-duplicate chunks.
        """
        if self._reranker is None:
            logger.warning(
                "cross_encoder strategy selected but no reranker injected. "
                "Falling back to MMR."
            )
            return await self._rank_mmr(chunks, query)

        # Step 1 — cross-encoder: run in thread (CPU-bound)
        reranked = await asyncio.to_thread(
            self._reranker.rerank, query, chunks, self._top_k
        )

        # Step 2 — MMR for diversity on the already-reranked small set
        if len(reranked) > 1:
            reranked = await self._rank_mmr(reranked, query)

        return reranked

    async def _rank_mmr(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """
        Maximal Marginal Relevance reranking.

        MMR bug fix — what changed vs original:
            BEFORE: embed_documents(chunks + [query]) → re-embeds everything
                    from scratch on every query call (~3 seconds wasted).
            AFTER:  chunk.relevance_score IS the cosine-sim to the query
                    (already computed by Qdrant). Use it directly.
                    Only embed chunk texts for inter-chunk similarity.
                    This eliminates the query re-embedding entirely and
                    uses pre-existing Qdrant scores for the relevance term.

        Falls back to no reranking if neither pre-fetched vectors nor
        an embeddings_fn is available.
        """
        has_vectors = bool(chunks) and chunks[0].vector is not None

        if not has_vectors and self._embeddings_fn is None:
            logger.warning(
                "MMR requested but no pre-fetched vectors and no embeddings_fn. "
                "Falling back to none."
            )
            return self._rank_none(chunks)

        try:
            # ── Relevance term: use Qdrant cosine-sim scores directly ───
            # relevance_score is already the cosine similarity between the
            # query vector and each chunk vector, computed during retrieval.
            relevance_scores = np.array(
                [chunk.relevance_score for chunk in chunks], dtype=np.float32
            )

            if has_vectors:
                # ── FAST PATH: use pre-fetched Qdrant vectors (0ms) ─────
                # Vectors were returned by similarity_search_with_vectors()
                # alongside the search results — no re-embedding needed.
                chunk_embeddings = np.array(
                    [chunk.vector for chunk in chunks], dtype=np.float32
                )
                logger.info(
                    "MMR: using pre-fetched Qdrant vectors | chunks=%d | "
                    "embedding=skipped",
                    len(chunks),
                )
            else:
                # ── FALLBACK: embed chunk texts (slower path) ────────────
                # Used when chunks come from a non-Qdrant source or from
                # code paths that don't carry pre-fetched vectors.
                embeddings_model = self._embeddings_fn()
                chunk_texts = [chunk.content for chunk in chunks]
                chunk_embeddings_raw = await asyncio.to_thread(
                    embeddings_model.embed_documents, chunk_texts
                )
                chunk_embeddings = np.array(chunk_embeddings_raw, dtype=np.float32)

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
            logger.warning("MMR failed, falling back to none | error=%s", exc)
            return self._rank_none(chunks)

    # ------------------------------------------------------------------
    # MMR core algorithm
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        chunk_embeddings: np.ndarray,
        n_results: int,
    ) -> list[int]:
        """
        Greedy MMR selection.

        score(d) = lambda * relevance(d, query)
                 - (1-lambda) * max_sim(d, already_selected)
        """
        n_chunks = len(relevance_scores)
        selected: list[int] = []
        remaining = list(range(n_chunks))

        # First pick: highest relevance
        best_idx = int(np.argmax(relevance_scores))
        selected.append(best_idx)
        remaining.remove(best_idx)

        for _ in range(min(n_results - 1, len(remaining))):
            if not remaining:
                break

            best_score = -float("inf")
            best_candidate = remaining[0]

            for candidate in remaining:
                relevance = float(relevance_scores[candidate])

                # Diversity: penalise similarity to already-selected chunks
                max_sim = max(
                    self._cosine_similarity(
                        chunk_embeddings[candidate], chunk_embeddings[sel]
                    )
                    for sel in selected
                )

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

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
