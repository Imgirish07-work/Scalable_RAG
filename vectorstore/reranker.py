"""
Cross-encoder reranker — ms-marco-MiniLM-L-6-v2.

Design:
    - Takes coarse retrieval results (top 10) and reranks them using a
      cross-encoder model that reads query + chunk together via full attention.
    - Returns the top_k highest-scoring chunks with updated relevance scores.
    - Runs entirely locally (GPU if available, CPU fallback) — no API calls, no extra cost.

Why cross-encoder beats bi-encoder for reranking:
    - Bi-encoder (BGE): embeds query and chunk independently, compares vectors.
      Fast but approximate — misses relevance signals that need both texts.
    - Cross-encoder (MiniLM): reads [query, chunk] together via transformer
      attention. Much more accurate but too slow to run on all chunks — so
      we run it only on the coarse top-k from Qdrant.

Integration:
    - Injected into ContextRanker at construction time via rag_factory.py.
    - ContextRanker calls reranker.rerank() when strategy='cross_encoder'.
    - Returns List[RetrievedChunk] with relevance_score replaced by
      cross-encoder logit score (normalized 0.0-1.0 via sigmoid).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from rag.models.rag_response import RetrievedChunk
from config.settings import settings
from utils.logger import get_logger
from vectorstore.embeddings import _ONNX_PROVIDERS

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_reranker_path() -> str:
    """Resolve local reranker model path relative to project root."""
    local_path = settings.RERANKER_MODEL_PATH
    if local_path:
        absolute = (_PROJECT_ROOT / local_path).resolve()
        if absolute.is_dir():
            logger.info(f"Using local reranker model from: {absolute}")
            return str(absolute)
        logger.warning(f"Reranker path '{absolute}' not found.")
    return local_path or ""


class CrossEncoderReranker:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Loads the model once and keeps it in memory. Reranks (query, chunk)
    pairs using a sequence classification forward pass. Scores are
    sigmoid-normalized to 0.0-1.0 range so they are comparable to
    the cosine similarity scores from Qdrant retrieval.

    Inference backend (auto-detected at init):
        1. ONNX Runtime — if onnx/model.onnx exists inside model_path (~3-6x faster).
        2. PyTorch     — fallback when ONNX file is absent.
    """

    def __init__(self, model_path: str, batch_size: int = 32) -> None:
        from transformers import AutoTokenizer

        self._batch_size = batch_size

        logger.info(f"Loading cross-encoder reranker from: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        onnx_path = Path(model_path)/"onnx" /"model.onnx"
        if onnx_path.exists():
            self._load_onnx(str(onnx_path))
        else:
            logger.warning(
                "ONNX model not found at '%s' — falling back to PyTorch.", onnx_path
            )
            self._load_pytorch(model_path)

        logger.info("CrossEncoderReranker loaded successfully.")

    def _load_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = settings.RERANKER_INTRA_OP_THREADS

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=_ONNX_PROVIDERS,
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        self._backend = "onnx"

        logger.info(
            "ONNX cross-encoder loaded | inputs=%s | output=%s | provider=%s",
            self._input_names,
            self._output_name,
            self._session.get_providers()[0],
        )

    def _load_pytorch(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._model.to(self._device)
        self._torch = torch
        self._backend = "pytorch"
        logger.info("PyTorch cross-encoder loaded | device=%s", self._device)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks by cross-encoder relevance score.

        Args:
            query:  Original user query.
            chunks: Coarse retrieval results (top 10-15 from Qdrant).
            top_k:  Number of chunks to return after reranking.

        Returns:
            top_k chunks sorted by cross-encoder score (highest first),
            with relevance_score updated to the normalized cross-encoder score.
        """
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))

        # Pre-filter: drop bottom candidates by RRF rank before cross-encoding.
        # Chunks arrive sorted by hybrid retrieval score (highest first).
        # Bottom-N by hybrid rank have never been rescued by the cross-encoder
        # in practice — both dense and sparse retrieval found them weak.
        prefilter_n = min(settings.RERANKER_PREFILTER_TOP_N, len(chunks))
        candidates = chunks[:prefilter_n]

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self._score_pairs(pairs)

        # Pair each chunk with its score, sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        # ── Relative score filtering ────────────────────────────────────────
        # Production-grade pattern: threshold adapts to each query's score
        # distribution instead of using a fixed absolute value.
        #
        # Rule: keep chunk if score >= max(top_score × RATIO, MIN_ABS_FLOOR)
        #   RATIO=0.4  → chunk must reach 40% of the best chunk's score
        #   MIN_ABS_FLOOR=0.1 → safety net when all chunks score low
        #
        # Examples:
        #   top=0.984, ratio=0.4 → threshold=0.394  (drops 0.231 ✓)
        #   top=0.200, ratio=0.4 → threshold=0.100  (floor overrides 0.08)
        #   top=0.000, ratio=0.4 → threshold=0.100  (all filtered, keep top-1)
        #
        # Always keeps the top-1 chunk so context is never empty here.
        # Pipeline-level RERANKER_SCORE_THRESHOLD handles the "even top-1 is
        # irrelevant" case upstream in base_rag.py.
        top_score = scored[0][0] if scored else 0.0
        top_scored = scored[:top_k]

        dynamic_threshold = max(
            top_score * settings.RERANKER_SCORE_RATIO,
            settings.RERANKER_MIN_ABS_FLOOR,
        )

        filtered = [(s, c) for s, c in top_scored if s >= dynamic_threshold]
        if not filtered:
            filtered = [top_scored[0]]
        dropped = len(top_scored) - len(filtered)

        # Stamp each returned chunk with its cross-encoder score so the
        # pipeline can detect low-confidence retrievals upstream (base_rag.py).
        # Keep original relevance_score (Qdrant cosine similarity) — it is used
        # for confidence calculation and MMR diversity scoring.
        result = [
            chunk.model_copy(update={"reranker_score": score})
            for score, chunk in filtered
        ]

        bottom_score = filtered[-1][0] if filtered else 0.0

        logger.info(
            "CrossEncoder rerank complete | backend=%s | fetched=%d | "
            "prefiltered=%d | top_k=%d | returned=%d | filtered_low=%d | "
            "threshold=%.3f | top_score=%.3f | bottom_score=%.3f",
            self._backend,
            len(chunks),
            len(candidates),
            top_k,
            len(result),
            dropped,
            dynamic_threshold,
            top_score,
            bottom_score,
        )

        return result

    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """
        Run cross-encoder forward pass on all (query, chunk) pairs.

        Processes in batches to avoid OOM on long chunk lists.
        Applies sigmoid to convert raw logits to 0.0-1.0 range.
        Dispatches to ONNX or PyTorch backend based on what was loaded.

        Args:
            pairs: List of [query, chunk_text] pairs.

        Returns:
            List of float scores in the same order as pairs.
        """
        if self._backend == "onnx":
            return self._score_pairs_onnx(pairs)
        return self._score_pairs_pytorch(pairs)

    def _score_pairs_onnx(self, pairs: list[list[str]]) -> list[float]:
        import numpy as np

        all_scores: list[float] = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            feed = {k: v for k, v in encoded.items() if k in self._input_names}
            outputs = self._session.run([self._output_name], feed)

            logits = outputs[0].squeeze(-1)          # (batch,) or scalar
            if logits.ndim == 0:
                logits = logits[np.newaxis]           # ensure 1-D

            # Sigmoid → 0.0-1.0
            scores = (1.0 / (1.0 + np.exp(-logits))).tolist()

            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

        return all_scores

    def _score_pairs_pytorch(self, pairs: list[list[str]]) -> list[float]:
        import torch

        all_scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]

                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                logits = self._model(**encoded).logits.squeeze(-1)

                normalized = torch.sigmoid(logits).tolist()

                if isinstance(normalized, float):
                    normalized = [normalized]

                all_scores.extend(normalized)

        return all_scores


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker | None:
    """
    Return a cached CrossEncoderReranker instance.

    Returns None if reranker is disabled or model path is missing.
    Called once at startup, cached for the process lifetime.
    """
    if not settings.RERANKER_ENABLED:
        logger.info("Reranker disabled (RERANKER_ENABLED=false).")
        return None

    model_path = _resolve_reranker_path()
    if not model_path:
        logger.warning("RERANKER_ENABLED=true but RERANKER_MODEL_PATH is not set.")
        return None

    try:
        return CrossEncoderReranker(
            model_path=model_path,
            batch_size=settings.RERANKER_BATCH_SIZE,
        )
    except Exception as e:
        logger.exception(f"Failed to load reranker: {e}")
        return None
