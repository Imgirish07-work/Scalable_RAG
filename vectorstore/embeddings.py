"""Embedding model factory with lru_cache singletons for ONNX/PyTorch backends."""

import os
from pathlib import Path
from functools import lru_cache
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# avoid network calls on air-gapped machines when a local path is configured
if settings.embedding_model_local_path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_providers() -> list:
    """Detect GPU availability and return the ONNX Runtime provider list."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        logger.info("ONNX Runtime: CPUExecutionProvider (CUDAExecutionProvider not compiled in)")
        return ["CPUExecutionProvider"]

    try:
        import torch
        if not torch.cuda.is_available():
            logger.info(
                "ONNX Runtime: CPUExecutionProvider "
                "(no CUDA device — onnxruntime-gpu installed but no GPU passthrough)",
            )
            return ["CPUExecutionProvider"]
    except Exception as exc:
        logger.warning(
            "CUDA availability probe failed (%s); falling back to CPU", exc,
        )
        return ["CPUExecutionProvider"]

    logger.info("ONNX Runtime: CUDAExecutionProvider selected — GPU inference enabled")
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


_ONNX_PROVIDERS: list = _resolve_providers()

# avoids a live embed call at collection creation time
_EMBEDDING_DIM_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5"                 : 384,
    "BAAI/bge-base-en-v1.5"                  : 768,
    "BAAI/bge-large-en-v1.5"                 : 1024,
    "BAAI/bge-m3"                            : 1024,
    "sentence-transformers/all-MiniLM-L6-v2" : 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def _resolve_model_path() -> str:
    """Return the model path to load, preferring a local directory over HuggingFace Hub."""
    local_path = settings.embedding_model_local_path
    if local_path:
        absolute_path = (_PROJECT_ROOT / local_path).resolve()
        if absolute_path.is_dir():
            logger.info(f"Using local embedding model from: {absolute_path}")
            return str(absolute_path)
        else:
            logger.warning(
                f"Local path '{absolute_path}' not found — "
                f"falling back to HuggingFace: {settings.embedding_model}"
            )
    logger.info(f"Using HuggingFace embedding model: {settings.embedding_model}")
    return settings.embedding_model


class ONNXEmbeddings(Embeddings):
    """LangChain-compatible embeddings powered by ONNX Runtime with mean pooling + L2 norm."""

    def __init__(
        self,
        model_path: str,
        onnx_file: str = "onnx/model.onnx",
        batch_size: int = 64,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._batch_size = batch_size
        onnx_path = str(Path(model_path) / onnx_file)

        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at '{onnx_path}'. "
                "Set USE_ONNX_EMBEDDINGS=false to use PyTorch instead."
            )

        logger.info(f"Loading ONNX model from: {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=_ONNX_PROVIDERS,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        self._output_name = self._session.get_outputs()[0].name
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        logger.info(
            f"ONNX model loaded successfully. "
            f"Inputs: {self._input_names} | Output: {self._output_name} | "
            f"Provider: {self._session.get_providers()[0]}"
        )

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Tokenise, run ONNX inference, pool, and L2-normalise a list of texts."""
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            feed = {k: v for k, v in encoded.items() if k in self._input_names}

            outputs = self._session.run([self._output_name], feed)
            token_embeddings = outputs[0]
            attention_mask   = encoded["attention_mask"]

            mask   = attention_mask[..., np.newaxis].astype(np.float32)
            pooled = (token_embeddings * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), a_min=1e-9, a_max=None
            )

            # BGE models require L2 normalisation for cosine similarity
            norms      = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalised = pooled / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.extend(normalised.tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


@lru_cache(maxsize=4)
def _get_pytorch_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    """Return a cached PyTorch-based HuggingFaceEmbeddings instance."""
    logger.info(f"Initialising PyTorch HuggingFaceEmbeddings: {model_path}")
    try:
        instance = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
            },
        )
        logger.info("HuggingFaceEmbeddings initialised successfully")
        return instance
    except OSError as e:
        logger.exception(f"Model not found or download failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error initialising HuggingFaceEmbeddings: {e}")
        raise


@lru_cache(maxsize=4)
def _get_onnx_embeddings(model_path: str) -> ONNXEmbeddings:
    """Return a cached ONNXEmbeddings instance keyed by model path."""
    return ONNXEmbeddings(
        model_path=model_path,
        onnx_file="onnx/model.onnx",
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embeddings instance (ONNX if configured and available, else PyTorch)."""
    model_path = _resolve_model_path()

    if settings.USE_ONNX_EMBEDDINGS:
        onnx_file = Path(model_path) / "onnx" / "model.onnx"
        if onnx_file.exists():
            logger.info("ONNX embeddings selected.")
            return _get_onnx_embeddings(model_path)
        else:
            logger.warning(
                f"USE_ONNX_EMBEDDINGS=true but '{onnx_file}' not found. "
                "Falling back to PyTorch."
            )

    logger.info("PyTorch embeddings selected.")
    return _get_pytorch_embeddings(model_path)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Return the vector dimension for the configured embedding model."""
    try:
        dimension = _EMBEDDING_DIM_MAP.get(settings.embedding_model)
        if dimension is None:
            logger.warning(
                f"Model '{settings.embedding_model}' not in dimension map. "
                "Computing dynamically via test embed."
            )
            test_vector = get_embeddings().embed_query("test")
            dimension = len(test_vector)
            logger.info(f"Computed dimension: {dimension}")
        return dimension
    except Exception as e:
        logger.exception(f"Error getting embedding dimension: {e}")
        raise
