"""Post-retrieval reranking strategy: none, mmr, cross_encoder."""

import asyncio
from typing import List

import numpy as np

from rag.models.rag_response import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MMR_LAMBDA = 0.7


class ContextRanker:
    """Post-retrieval reranker with pluggable MMR and cross-encoder strategies."""

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

    @property
    def retrieval_top_k(self) -> int:
        """How many chunks to fetch from the vector store before reranking."""
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
        """Rerank chunks using the configured or overridden strategy."""
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

    def _rank_none(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Pass-through — return chunks in the original retrieval order."""
        logger.info("Rerank=none | chunks=%d | order preserved", len(chunks))
        return chunks

    async def _rank_cross_encoder(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """Cross-encoder rerank followed by MMR for diversity."""
        if self._reranker is None:
            logger.warning(
                "cross_encoder strategy selected but no reranker injected. "
                "Falling back to MMR."
            )
            return await self._rank_mmr(chunks, query)

        # cross-encoder inference is cpu-bound
        reranked = await asyncio.to_thread(
            self._reranker.rerank, query, chunks, self._top_k
        )

        if len(reranked) > 1:
            reranked = await self._rank_mmr(reranked, query)

        return reranked

    async def _rank_mmr(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """Maximal Marginal Relevance reranking."""
        has_vectors = bool(chunks) and chunks[0].vector is not None

        if not has_vectors and self._embeddings_fn is None:
            logger.warning(
                "MMR requested but no pre-fetched vectors and no embeddings_fn. "
                "Falling back to none."
            )
            return self._rank_none(chunks)

        try:
            # relevance_score is the cosine sim from qdrant during retrieval
            relevance_scores = np.array(
                [chunk.relevance_score for chunk in chunks], dtype=np.float32
            )

            if has_vectors:
                # use pre-fetched qdrant vectors, no re-embedding needed
                chunk_embeddings = np.array(
                    [chunk.vector for chunk in chunks], dtype=np.float32
                )
                logger.info(
                    "MMR: using pre-fetched Qdrant vectors | chunks=%d | "
                    "embedding=skipped",
                    len(chunks),
                )
            else:
                # fallback: embed chunk texts when pre-fetched vectors are absent
                embeddings_model = await asyncio.to_thread(self._embeddings_fn)
                chunk_texts = [chunk.content for chunk in chunks]
                chunk_embeddings_raw = await asyncio.to_thread(
                    embeddings_model.embed_documents, chunk_texts
                )
                chunk_embeddings = np.array(chunk_embeddings_raw, dtype=np.float32)

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

    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        chunk_embeddings: np.ndarray,
        n_results: int,
    ) -> list[int]:
        """Greedy MMR selection returning chunk indices in ranked order."""
        n_chunks = len(relevance_scores)
        selected: list[int] = []
        remaining = list(range(n_chunks))

        # first pick: highest relevance, no diversity penalty
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

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
