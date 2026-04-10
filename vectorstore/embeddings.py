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

# Project root — resolved relative to this file (vectorstore/embeddings.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force HuggingFace Hub offline mode when a local model path is configured.
if settings.embedding_model_local_path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_providers() -> list:
    """
    Detect GPU availability once at module load and return the ONNX Runtime
    execution provider list in priority order.

    ONNX Runtime iterates providers left-to-right and uses the first one that
    is both installed and functional — CPUExecutionProvider is always the
    safe fallback.

    GPU options:
        gpu_mem_limit          — caps VRAM at 2 GB; leaves headroom for SPLADE.
        arena_extend_strategy  — pre-allocates in power-of-two chunks (fewer
                                 small allocations → lower latency variance).
        cudnn_conv_algo_search — EXHAUSTIVE finds the fastest cuDNN kernel on
                                 first run and caches it for subsequent calls.
        do_copy_in_default_stream — keeps host↔device transfers on the default
                                    CUDA stream to avoid sync overhead.
    """
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        logger.info("ONNX Runtime: CPUExecutionProvider (no CUDA GPU detected)")
        return ["CPUExecutionProvider"]

    logger.info("ONNX Runtime: CUDAExecutionProvider selected — GPU inference enabled")
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": 2 * 1024 ** 3,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


# Resolved once per process — shared by every ONNX InferenceSession in this pipeline.
_ONNX_PROVIDERS: list = _resolve_providers()

# Dimension map — avoids recomputing on every collection creation
_EMBEDDING_DIM_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5"                 : 384,
    "BAAI/bge-base-en-v1.5"                  : 768,
    "BAAI/bge-large-en-v1.5"                 : 1024,
    "BAAI/bge-m3"                            : 1024,
    "sentence-transformers/all-MiniLM-L6-v2" : 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def _resolve_model_path() -> str:
    """
    Resolve which model path to use.

    Priority:
        1. Local path (EMBEDDING_MODEL_LOCAL_PATH in .env) — if set AND folder exists.
           Path is resolved relative to project root, not CWD.
        2. HuggingFace model name — fallback (needs internet).
    """
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


# ONNX Embeddings

class ONNXEmbeddings(Embeddings):
    """
    ONNX Runtime-based embeddings for faster CPU inference.
    Implements the LangChain Embeddings interface so it is a drop-in
    replacement for HuggingFaceEmbeddings anywhere in the pipeline.

    Applies mean pooling + L2 normalisation — required for BGE models
    used with cosine similarity in Qdrant.
    """

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

        # Capture model I/O names dynamically — works for any BERT-based export
        self._output_name = self._session.get_outputs()[0].name
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        logger.info(
            f"ONNX model loaded successfully. "
            f"Inputs: {self._input_names} | Output: {self._output_name} | "
            f"Provider: {self._session.get_providers()[0]}"
        )


    # Internal helpers

    def _encode(self, texts: List[str]) -> List[List[float]]:
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

            # Only pass inputs the ONNX graph actually declares
            feed = {k: v for k, v in encoded.items() if k in self._input_names}

            outputs = self._session.run([self._output_name], feed)
            token_embeddings = outputs[0]          # (batch, seq_len, hidden_dim)
            attention_mask   = encoded["attention_mask"]  # (batch, seq_len)

            # Mean pooling — weighted by attention mask
            mask   = attention_mask[..., np.newaxis].astype(np.float32)
            pooled = (token_embeddings * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), a_min=1e-9, a_max=None
            )

            # L2 normalisation (mandatory for BGE + cosine similarity)
            norms      = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalised = pooled / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.extend(normalised.tolist())

        return all_embeddings


    # LangChain Embeddings interface

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


# PyTorch fallback (HuggingFaceEmbeddings)

@lru_cache(maxsize=4)
def _get_pytorch_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    """
    Cached PyTorch-based embeddings via sentence-transformers.
    normalize_embeddings=True is mandatory for BGE + cosine similarity.
    batch_size=64 for faster ingestion throughput.
    """
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
    """Cached ONNX embeddings instance keyed by model path."""
    return ONNXEmbeddings(
        model_path=model_path,
        onnx_file="onnx/model.onnx",
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


# Public API

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Return a cached embeddings instance.

    Selection priority:
        1. ONNX  — if USE_ONNX_EMBEDDINGS=true AND local model path exists
                   AND onnx/model.onnx is present inside that folder.
        2. PyTorch — fallback for all other cases (remote HF model or no ONNX file).
    """
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
    """
    Return vector dimensions for the configured model.
    Qdrant needs this at collection-creation time.
    Falls back to a live embed if the model is not in the dimension map.
    lru_cache ensures the dynamic path runs at most once.
    """
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
