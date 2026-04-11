"""
Cross-encoder reranker for post-retrieval relevance scoring.

Design:
    Loads ms-marco-MiniLM-L-6-v2 once at startup and keeps it in memory.
    Scores (query, chunk) pairs via a sequence-classification forward pass,
    applies sigmoid normalisation to produce 0.0-1.0 scores, and returns
    the top-k highest-scoring chunks.

    Why cross-encoder beats bi-encoder for reranking:
        Bi-encoder (BGE): embeds query and chunk independently, then compares
        vectors. Fast but approximate — misses relevance signals that require
        reading both texts together.
        Cross-encoder (MiniLM): reads [query, chunk] together via full
        transformer attention. Much more accurate but too slow to run on all
        chunks — so it runs only on the coarse top-k from Qdrant.

    Inference backend (auto-detected at init):
        1. ONNX Runtime — when onnx/model.onnx (standard) or model.onnx
           (CUDA-native export) exists inside the model directory.
        2. PyTorch       — fallback when no ONNX file is found.

Chain of Responsibility:
    Injected into ContextRanker at construction time via rag_factory.py.
    ContextRanker calls reranker.rerank() when strategy='cross_encoder'.
    Reads config from config.settings; uses _ONNX_PROVIDERS from embeddings.

Dependencies:
    onnxruntime (optional), torch (optional), transformers, config.settings,
    vectorstore.embeddings (_ONNX_PROVIDERS), rag.models.rag_response
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
    """Return the absolute model path to load, choosing CUDA-native export when available.

    Selection priority:
        1. CUDA-native export (RERANKER_MODEL_PATH_CUDA) — selected when CUDA
           is available and the directory exists. No Memcpy nodes, ~3x faster.
        2. Standard export (RERANKER_MODEL_PATH) — device-agnostic fallback.
           Works on both GPU (with Memcpy overhead) and CPU-only machines.

    Returns:
        Absolute path string to the selected model directory, or empty string
        if neither path resolves to an existing directory.
    """
    import torch

    cuda_available = torch.cuda.is_available()

    # Prefer CUDA-native export when GPU is present.
    if cuda_available and settings.RERANKER_MODEL_PATH_CUDA:
        cuda_absolute = (_PROJECT_ROOT / settings.RERANKER_MODEL_PATH_CUDA).resolve()
        if cuda_absolute.is_dir():
            logger.info(
                "Reranker: CUDA-native model selected | path=%s", cuda_absolute
            )
            return str(cuda_absolute)
        logger.warning(
            "Reranker: CUDA path configured but not found — "
            "falling back to standard model | path=%s",
            cuda_absolute,
        )

    # Standard path — works on CPU and GPU (with Memcpy if GPU).
    standard_path = settings.RERANKER_MODEL_PATH
    if standard_path:
        standard_absolute = (_PROJECT_ROOT / standard_path).resolve()
        if standard_absolute.is_dir():
            if cuda_available:
                logger.info(
                    "Reranker: standard model selected (CUDA available but "
                    "no CUDA export found) | path=%s", standard_absolute
                )
            else:
                logger.info(
                    "Reranker: standard model selected (CPU mode) | path=%s",
                    standard_absolute,
                )
            return str(standard_absolute)
        logger.warning("Reranker: standard path not found | path=%s", standard_absolute)

    return ""


class CrossEncoderReranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Loads the model once and keeps it in memory for the process lifetime.
    Reranks (query, chunk) pairs using a sequence classification forward pass.
    Scores are sigmoid-normalised to 0.0-1.0 so they are comparable to the
    cosine similarity scores produced by Qdrant retrieval.

    Attributes:
        _batch_size: Maximum (query, chunk) pairs per forward pass.
        _tokenizer: HuggingFace tokenizer for the cross-encoder model.
        _backend: Active inference backend string ('onnx' or 'pytorch').
        _session: ONNX Runtime InferenceSession (ONNX backend only).
        _input_names: ONNX graph input node names (ONNX backend only).
        _output_name: ONNX graph output node name (ONNX backend only).
        _model: PyTorch sequence classification model (PyTorch backend only).
        _device: Torch device string (PyTorch backend only).
    """

    def __init__(self, model_path: str, batch_size: int = 32) -> None:
        """Load the tokeniser and select the inference backend.

        Checks for an ONNX file in two locations:
            1. {model_path}/onnx/model.onnx — standard convention (model dir with
               ONNX in a subdirectory, as used by the standard model export).
            2. {model_path}/model.onnx — optimum-cli direct-export convention
               (used when exporting with --device cuda to a dedicated directory).

        Args:
            model_path: Absolute path to the local model directory.
            batch_size: Maximum (query, chunk) pairs per forward pass.
        """
        from transformers import AutoTokenizer

        self._batch_size = batch_size

        logger.info(f"Loading cross-encoder reranker from: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        _root = Path(model_path)
        onnx_path = next(
            (p for p in (_root / "onnx" / "model.onnx", _root / "model.onnx")
             if p.exists()),
            None,
        )
        if onnx_path is not None:
            self._load_onnx(str(onnx_path))
        else:
            logger.warning(
                "ONNX model not found in '%s' — falling back to PyTorch.", model_path
            )
            self._load_pytorch(model_path)

        logger.info("CrossEncoderReranker loaded successfully.")

    def _load_onnx(self, onnx_path: str) -> None:
        """Initialise the ONNX Runtime inference session for the cross-encoder.

        Args:
            onnx_path: Absolute path to the .onnx model file.
        """
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
        """Initialise the PyTorch sequence classification model.

        Args:
            model_path: Absolute path to the local model directory.
        """
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
        """Rerank retrieval candidates by cross-encoder relevance score.

        Steps:
            1. Pre-filter: retain only the top RERANKER_PREFILTER_TOP_N
               candidates by hybrid retrieval rank (bottom-ranked chunks are
               rarely rescued by the cross-encoder in practice).
            2. Score all (query, chunk) pairs via the cross-encoder.
            3. Apply relative + absolute score filtering to drop low-quality
               chunks while always retaining at least the top-1 result.
            4. Stamp each returned chunk with its cross-encoder score in
               reranker_score so the pipeline can detect low-confidence
               retrievals upstream (base_rag.py).

        Score filtering rule:
            keep chunk if score >= max(top_score × RATIO, MIN_ABS_FLOOR)
            RATIO=0.4 → chunk must reach 40% of the best chunk's score.
            MIN_ABS_FLOOR=0.1 → safety net when all chunks score low.
            Examples:
                top=0.984, ratio=0.4 → threshold=0.394  (drops 0.231)
                top=0.200, ratio=0.4 → threshold=0.100  (floor overrides 0.08)
                top=0.000, ratio=0.4 → threshold=0.100  (all filtered → keep top-1)

        Pipeline-level RERANKER_SCORE_THRESHOLD handles the case where even
        the top-1 chunk is irrelevant (checked upstream in base_rag.py).

        Args:
            query: Original user query.
            chunks: Coarse retrieval results (top 10-15 from Qdrant), sorted
                    by hybrid retrieval score descending.
            top_k: Number of chunks to return after reranking.

        Returns:
            Up to top_k chunks sorted by cross-encoder score descending, with
            relevance_score unchanged (Qdrant cosine similarity, used for
            confidence and MMR scoring) and reranker_score set to the
            cross-encoder sigmoid score.
        """
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))

        # Pre-filter: drop bottom candidates before cross-encoding.
        # Chunks arrive sorted by hybrid retrieval score (highest first).
        prefilter_n = min(settings.RERANKER_PREFILTER_TOP_N, len(chunks))
        candidates = chunks[:prefilter_n]

        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self._score_pairs(pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        # Relative score filtering — threshold adapts to the query's distribution.
        top_score = scored[0][0] if scored else 0.0
        top_scored = scored[:top_k]

        dynamic_threshold = max(
            top_score * settings.RERANKER_SCORE_RATIO,
            settings.RERANKER_MIN_ABS_FLOOR,
        )

        filtered = [(s, c) for s, c in top_scored if s >= dynamic_threshold]
        if not filtered:
            # Always keep at least one chunk so context is never empty here.
            filtered = [top_scored[0]]
        dropped = len(top_scored) - len(filtered)

        # Stamp reranker_score; preserve original relevance_score for MMR.
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
        """Run the cross-encoder forward pass on all (query, chunk) pairs.

        Processes pairs in batches to avoid OOM on long lists. Applies
        sigmoid to convert raw logits to the 0.0-1.0 range. Dispatches to
        the ONNX or PyTorch backend based on what was loaded at init.

        Args:
            pairs: List of [query, chunk_text] pairs.

        Returns:
            List of float scores in the same order as pairs.
        """
        if self._backend == "onnx":
            return self._score_pairs_onnx(pairs)
        return self._score_pairs_pytorch(pairs)

    def _score_pairs_onnx(self, pairs: list[list[str]]) -> list[float]:
        """Score pairs using the ONNX Runtime session.

        Args:
            pairs: List of [query, chunk_text] pairs.

        Returns:
            Sigmoid-normalised float scores in the same order as pairs.
        """
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

            logits = outputs[0].squeeze(-1)       # (batch,) or scalar
            if logits.ndim == 0:
                logits = logits[np.newaxis]        # ensure 1-D

            scores = (1.0 / (1.0 + np.exp(-logits))).tolist()

            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

        return all_scores

    def _score_pairs_pytorch(self, pairs: list[list[str]]) -> list[float]:
        """Score pairs using the PyTorch model.

        Args:
            pairs: List of [query, chunk_text] pairs.

        Returns:
            Sigmoid-normalised float scores in the same order as pairs.
        """
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
    """Return a cached CrossEncoderReranker instance, or None if disabled.

    Called once at startup; the result is cached for the process lifetime.
    Returns None when RERANKER_ENABLED=false or the model path is not set,
    allowing the pipeline to skip reranking without raising errors.

    Returns:
        CrossEncoderReranker instance, or None if reranking is disabled or
        the model path cannot be resolved.
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
