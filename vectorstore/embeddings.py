"""
Embedding model factory for the RAG pipeline.

Design:
    Factory pattern with lru_cache singletons. Provides a single public entry
    point — get_embeddings() — that returns the appropriate embedding backend
    (ONNX or PyTorch) based on settings. A second public function,
    get_embedding_dimension(), returns the vector dimension needed by Qdrant
    at collection-creation time.

Chain of Responsibility:
    Called by QdrantStore and ContextRanker via get_embeddings().
    Reads config from config.settings → loads model from local path or
    HuggingFace Hub → returns an Embeddings instance to the caller.

Dependencies:
    onnxruntime, transformers, langchain_huggingface, numpy, config.settings
"""

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

# Project root resolved relative to this file (vectorstore/embeddings.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force HuggingFace Hub offline mode when a local model path is configured,
# preventing accidental network calls on air-gapped or corporate machines.
if settings.embedding_model_local_path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_providers() -> list:
    """Detect GPU availability and return the ONNX Runtime provider list.

    ONNX Runtime iterates providers left-to-right and uses the first one that
    is both installed and functional — CPUExecutionProvider is always the
    safe fallback.

    GPU provider options explained:
        arena_extend_strategy  — grows the BFC Arena in power-of-two chunks,
                                 reducing allocation fragmentation and latency
                                 variance under sustained inference load.
        cudnn_conv_algo_search — HEURISTIC selects the cuDNN convolution kernel
                                 via heuristics without benchmarking overhead.
                                 EXHAUSTIVE was avoided because NLP batches have
                                 variable sequence lengths (different padding per
                                 batch) — each unique (batch, seq_len) shape
                                 triggers a fresh 40-50s benchmark, making
                                 ingestion slower than CPU.
        do_copy_in_default_stream — pins host-device transfers to the default
                                    CUDA stream, avoiding cross-stream sync
                                    overhead.

    Note: gpu_mem_limit is intentionally omitted (ORT default = 0, no cap).
    A fixed cap causes OOM when the BERT attention matrix
    (batch × heads × seq² × 4 bytes) plus cuDNN workspace exceeds it.
    ORT's BFC Arena manages VRAM on demand up to physical capacity.

    Returns:
        List of ONNX Runtime execution providers in priority order.
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
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


# Resolved once per process — shared by every ONNX InferenceSession in this pipeline.
_ONNX_PROVIDERS: list = _resolve_providers()

# Known embedding dimensions keyed by HuggingFace model ID.
# Avoids a live embed call on every collection creation.
_EMBEDDING_DIM_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5"                 : 384,
    "BAAI/bge-base-en-v1.5"                  : 768,
    "BAAI/bge-large-en-v1.5"                 : 1024,
    "BAAI/bge-m3"                            : 1024,
    "sentence-transformers/all-MiniLM-L6-v2" : 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def _resolve_model_path() -> str:
    """Return the model path to load, preferring a local directory over HuggingFace Hub.

    Priority:
        1. Local path (EMBEDDING_MODEL_LOCAL_PATH in .env) — if set and the
           directory exists. Path is resolved relative to project root, not CWD.
        2. HuggingFace model name — fallback when local path is absent or missing.

    Returns:
        Absolute path string to the local model directory, or the HuggingFace
        model ID string when falling back to Hub download.
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


class ONNXEmbeddings(Embeddings):
    """LangChain-compatible embeddings powered by ONNX Runtime.

    Drop-in replacement for HuggingFaceEmbeddings wherever the pipeline
    uses the Embeddings interface. Applies mean pooling followed by L2
    normalisation — mandatory for BGE models used with cosine similarity
    in Qdrant.

    Attributes:
        _batch_size: Maximum texts per ONNX forward pass.
        _session: ONNX Runtime InferenceSession for the encoder.
        _tokenizer: HuggingFace tokenizer matching the model.
        _output_name: Name of the ONNX graph output node (token embeddings).
        _input_names: Set of ONNX graph input node names.
    """

    def __init__(
        self,
        model_path: str,
        onnx_file: str = "onnx/model.onnx",
        batch_size: int = 64,
    ) -> None:
        """Load the ONNX model and tokenizer from the given local directory.

        Args:
            model_path: Absolute path to the local model directory.
            onnx_file: Relative path to the .onnx file inside model_path.
            batch_size: Maximum texts per ONNX forward pass.

        Raises:
            FileNotFoundError: If the .onnx file does not exist at the
                               resolved path.
        """
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

        # Capture model I/O names dynamically — works for any BERT-based export.
        self._output_name = self._session.get_outputs()[0].name
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        logger.info(
            f"ONNX model loaded successfully. "
            f"Inputs: {self._input_names} | Output: {self._output_name} | "
            f"Provider: {self._session.get_providers()[0]}"
        )

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Tokenise, run ONNX inference, pool, and L2-normalise a list of texts.

        Processes texts in batches of self._batch_size to respect VRAM limits.

        Args:
            texts: List of strings to embed.

        Returns:
            List of L2-normalised embedding vectors (one per input text).
        """
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

            # Only pass inputs the ONNX graph actually declares.
            feed = {k: v for k, v in encoded.items() if k in self._input_names}

            outputs = self._session.run([self._output_name], feed)
            token_embeddings = outputs[0]                    # (batch, seq_len, hidden_dim)
            attention_mask   = encoded["attention_mask"]     # (batch, seq_len)

            # Mean pooling weighted by the attention mask (ignores padding tokens).
            mask   = attention_mask[..., np.newaxis].astype(np.float32)
            pooled = (token_embeddings * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), a_min=1e-9, a_max=None
            )

            # L2 normalisation — mandatory for BGE models with cosine similarity.
            norms      = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalised = pooled / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.extend(normalised.tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._encode([text])[0]


@lru_cache(maxsize=4)
def _get_pytorch_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    """Return a cached PyTorch-based HuggingFaceEmbeddings instance.

    normalize_embeddings=True is mandatory for BGE models with cosine
    similarity. batch_size is set from settings for consistent throughput.

    Args:
        model_path: Local model directory path or HuggingFace model ID.

    Returns:
        Initialised HuggingFaceEmbeddings instance.

    Raises:
        OSError: If the model path does not exist or the download fails.
        Exception: For any other initialisation failure.
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
    """Return a cached ONNXEmbeddings instance keyed by model path.

    Args:
        model_path: Absolute path to the local model directory.

    Returns:
        Initialised ONNXEmbeddings instance.
    """
    return ONNXEmbeddings(
        model_path=model_path,
        onnx_file="onnx/model.onnx",
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embeddings instance selected by settings.

    Selection priority:
        1. ONNX  — if USE_ONNX_EMBEDDINGS=true AND local model path exists
                   AND onnx/model.onnx is present inside that folder.
        2. PyTorch — fallback for all other cases (remote HF model or no ONNX file).

    Returns:
        An Embeddings instance (either ONNXEmbeddings or HuggingFaceEmbeddings).
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
    """Return the vector dimension for the configured embedding model.

    Looks up the dimension from a static map first. If the model is not in
    the map, falls back to a live test embed to compute it dynamically.
    lru_cache ensures the dynamic path runs at most once per process.

    Returns:
        Integer vector dimension (e.g., 768 for bge-base-en-v1.5).

    Raises:
        Exception: If the live embed call fails.
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
