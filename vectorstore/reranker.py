"""Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) with ONNX or PyTorch backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from rag.models.rag_response import RetrievedChunk

from config.settings import settings
from utils.logger import get_logger
from vectorstore.embeddings import _ONNX_PROVIDERS

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_reranker_path() -> str:
    """Return the absolute model path, preferring CUDA-native export when available."""
    import torch

    cuda_available = torch.cuda.is_available()

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
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 with sigmoid-normalised scores."""

    def __init__(self, model_path: str, batch_size: int = 32) -> None:
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
        """Initialise the ONNX Runtime inference session for the cross-encoder."""
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
        """Initialise the PyTorch sequence classification model."""
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
        """Rerank retrieval candidates by cross-encoder relevance score."""
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))

        # drop low-ranked candidates before cross-encoding; chunks arrive sorted by hybrid score
        prefilter_n = min(settings.RERANKER_PREFILTER_TOP_N, len(chunks))
        candidates = chunks[:prefilter_n]

        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self._score_pairs(pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

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
        """Run the cross-encoder forward pass on all (query, chunk) pairs."""
        if self._backend == "onnx":
            return self._score_pairs_onnx(pairs)
        return self._score_pairs_pytorch(pairs)

    def _score_pairs_onnx(self, pairs: list[list[str]]) -> list[float]:
        """Score pairs using the ONNX Runtime session."""
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

            logits = outputs[0].squeeze(-1)
            if logits.ndim == 0:
                logits = logits[np.newaxis]

            scores = (1.0 / (1.0 + np.exp(-logits))).tolist()

            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

        return all_scores

    def _score_pairs_pytorch(self, pairs: list[list[str]]) -> list[float]:
        """Score pairs using the PyTorch model."""
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
    """Return a cached CrossEncoderReranker instance, or None if disabled."""
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
