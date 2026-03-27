import os
from pathlib import Path
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Project root — resolved relative to this file (vectorstore/embeddings.py)
# Stable regardless of CWD where the process is launched from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force HuggingFace Hub offline mode when a local model path is configured.
# HF libraries read os.environ directly — pydantic settings/.env is not enough.
# setdefault preserves any explicit override already set in the shell environment.
if settings.embedding_model_local_path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Dimension Map- avoids recomputing on every collection creation
_EMBEDDING_DIM_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5"                 : 384,
    "BAAI/bge-base-en-v1.5"                  : 768,
    "BAAI/bge-m3"                            : 1024,
    "sentence-transformers/all-MiniLM-L6-v2" : 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}

def _resolve_model_path() -> str:
    """
    Resolve which model path to use.

    Priority:
        1. Local path (EMBEDDING_MODEL_LOCAL_PATH in .env) — if set AND folder exists
           Path is resolved relative to project root, not CWD.
        2. HuggingFace model name — fallback (needs internet)
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

@lru_cache(maxsize=4)
def _get_embeddings_cached(model_path: str) -> HuggingFaceEmbeddings:
    """
    Internal cached factory — keyed by model_path.
    lru_cache(maxsize=4) supports up to 4 distinct models without evicting.
    Called only by get_embeddings() — do not call directly.

    normalize_embeddings=True — mandatory for BGE models.
    BGE outputs raw vectors. Cosine similarity requires unit vectors.
    Without this → similarity scores are mathematically incorrect.
    """
    try:
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {model_path}")
        instance = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("HuggingFaceEmbeddings initialized successfully")
        return instance
    except OSError as e:
        logger.exception(f"Model not found or download failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error initializing HuggingFaceEmbeddings: {e}")
        raise


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Public API — returns a cached HuggingFaceEmbeddings instance.
    Resolves the model path each call (cheap), then delegates to
    _get_embeddings_cached() which is keyed by model_path.
    Switching models auto-invalidates — no manual cache_clear() needed.
    """
    return _get_embeddings_cached(_resolve_model_path())


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """
    Return vector dimensions for the configured model.
    Qdrant needs this at collection creation time.
    If model is not in the map → compute dynamically from a test embed.
    This is slower but safe — prevents silent wrong-dimension errors.
    lru_cache ensures the dynamic embed_query("test") path runs at most once.
    """
    try:
        dimension = _EMBEDDING_DIM_MAP.get(settings.embedding_model)
        if dimension is None:
            logger.warning(f"Model {settings.embedding_model} not in dimension map. Computing dynamically.")
            test_vector = get_embeddings().embed_query("test")
            dimension = len(test_vector)
            logger.info(f"Computed dimension for model {settings.embedding_model}: {dimension}")
        return dimension
    except Exception as e:
        logger.exception(f"Error getting embedding dimension: {e}")
        raise