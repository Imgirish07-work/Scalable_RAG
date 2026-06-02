"""Central configuration loaded from environment variables and .env via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and .env."""

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")

    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    default_provider: str = Field(default="gemini", env="DEFAULT_PROVIDER")

    # groq models — each role has a different rpd budget
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    GROQ_MODEL_STRONG: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACK: str = "qwen/qwen3-32b"

    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")
    GROQ_TIMEOUT: float = Field(default=30.0, env="GROQ_TIMEOUT")

    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        env="EMBEDDING_MODEL",
    )
    embedding_model_local_path: str = Field(
        default="models/bge-base-en-v1.5",
        env="EMBEDDING_MODEL_LOCAL_PATH",
    )
    # onnx embeddings: faster cpu inference
    USE_ONNX_EMBEDDINGS: bool = Field(default=True, env="USE_ONNX_EMBEDDINGS")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, env="EMBEDDING_BATCH_SIZE")

    # outer ingestion batch — 200 for cloud qdrant, 500 for local
    INGESTION_BATCH_SIZE: int = Field(default=200, env="INGESTION_BATCH_SIZE")

    # splade sparse model — when set, fastembed skips hf download (corporate network fix)
    SPLADE_LOCAL_PATH: str = Field(default="", env="SPLADE_LOCAL_PATH")
    # 0 = all logical cores; 6 is optimal for i5-1345U
    SPLADE_INTRA_OP_THREADS: int = Field(default=6, env="SPLADE_INTRA_OP_THREADS")
    # 16 fits 4gb vram; increase to 32 on 8+ gb gpus
    SPLADE_BATCH_SIZE: int = Field(default=16, env="SPLADE_BATCH_SIZE")

    min_chars_per_page: int = Field(default=50, env="MIN_CHARS_PER_PAGE")
    prefer_pdfplumber: bool = Field(default=False, env="PREFER_PDFPLUMBER")

    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    code_chunk_overlap: int = Field(default=150, env="CODE_CHUNK_OVERLAP")
    min_chunk_tokens: int = Field(default=20, env="MIN_CHUNK_TOKENS")

    qdrant_collection_name: str = Field(
        default="rag_collection",
        env="QDRANT_COLLECTION_NAME",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL",
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    # grpc is faster but port 6334 may be blocked on corporate networks; falls back to http
    QDRANT_PREFER_GRPC: bool = Field(default=True, env="QDRANT_PREFER_GRPC")

    max_tokens_per_chunk: int = Field(default=500, env="MAX_TOKENS_PER_CHUNK")
    max_recursion_depth: int = Field(default=5, env="MAX_RECURSION_DEPTH")
    effective_context_limit: int = Field(default=8000, env="EFFECTIVE_CONTEXT_LIMIT")

    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")

    RERANKER_ENABLED: bool = Field(default=False, env="RERANKER_ENABLED")
    RERANKER_MODEL_PATH: str = Field(default="", env="RERANKER_MODEL_PATH")
    # cuda-native onnx export — auto-selected when cuda available, falls back to cpu path
    RERANKER_MODEL_PATH_CUDA: str = Field(default="", env="RERANKER_MODEL_PATH_CUDA")
    RERANKER_BATCH_SIZE: int = Field(default=32, env="RERANKER_BATCH_SIZE")
    # 3x top_k so the cross-encoder has a meaningful candidate pool
    RERANKER_COARSE_TOP_K: int = Field(default=15, env="RERANKER_COARSE_TOP_K")
    # calibrated for bge reranker base (0.4-0.95 relevant, 0.02-0.20 irrelevant)
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.12, env="RERANKER_SCORE_THRESHOLD")
    # relative filter: kept if score >= top_score * ratio AND score >= min_abs_floor
    RERANKER_SCORE_RATIO: float = Field(default=0.4, env="RERANKER_SCORE_RATIO")
    # absolute floor — must be lower than RERANKER_SCORE_THRESHOLD
    RERANKER_MIN_ABS_FLOOR: float = Field(default=0.08, env="RERANKER_MIN_ABS_FLOOR")
    # pre-filter: keep top-N by rrf rank before cross-encoding; equal to coarse_top_k disables
    RERANKER_PREFILTER_TOP_N: int = Field(default=12, env="RERANKER_PREFILTER_TOP_N")
    # 0 = let ort decide; 4-6 optimal for i5/i7
    RERANKER_INTRA_OP_THREADS: int = Field(default=4, env="RERANKER_INTRA_OP_THREADS")

    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_directory: str = Field(default="./data/cache", env="CACHE_DIRECTORY")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    CACHE_L1_MAX_SIZE: int = 1000

    REDIS_ENV: str = "local"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CLOUD_URL: str = ""
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_SOCKET_TIMEOUT: float = 2.0
    REDIS_RETRY_ON_TIMEOUT: bool = True

    CACHE_STRATEGY: str = "exact"

    CACHE_SEMANTIC_THRESHOLD_LOW: float = 0.95
    CACHE_SEMANTIC_THRESHOLD_DIRECT: float = 0.98
    CACHE_SEMANTIC_COLLECTION: str = "cache_semantic"

    CACHE_CIRCUIT_BREAKER_THRESHOLD: int = 5
    CACHE_CIRCUIT_BREAKER_RESET_SECONDS: float = 60.0

    # cache quality gate — short or fast responses are likely errors
    CACHE_MIN_RESPONSE_TOKENS: int = 20
    CACHE_MIN_RESPONSE_LATENCY_MS: float = 100.0

    RAG_DEFAULT_VARIANT: str = "simple"
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_TOKENS: int = 3072
    RAG_RERANK_STRATEGY: str = "mmr"
    RAG_RETRIEVAL_MODE: str = "hybrid"
    RAG_CONFIDENCE_METHOD: str = "retrieval"
    # backfills from coarse pool when ratio filter leaves fewer chunks; prevents incomplete answers
    RAG_MIN_CONTEXT_CHUNKS: int = 2


    # cost per token — used to estimate cache savings
    COST_PER_TOKEN_OPENAI: float = 0.000002
    COST_PER_TOKEN_GEMINI: float = 0.0000001
    COST_PER_TOKEN_GROQ: float = 0.0

    # rate limiter — rpm/rpd looked up per-model from llm/rate_limiter/model_limits.py
    LLM_RATE_LIMITER_ENABLED: bool = True
    LLM_MAX_CONCURRENT: int = 5
    LLM_BURST_MULTIPLIER: float = 1.0

    use_cheap_model_threshold: int = Field(default=500, env="USE_CHEAP_MODEL_THRESHOLD")
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./data/logs/app.log", env="LOG_FILE")

    app_name: str = Field(default="Scalable RAG RLM", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensure default_provider is one of the supported LLM providers."""
        allowed = ["openai", "gemini", "groq"]
        if v.lower() not in allowed:
            raise ValueError(f"default_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Ensure chunk_overlap < chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("CACHE_STRATEGY")
    @classmethod
    def validate_cache_strategy(cls, v: str) -> str:
        """Ensure CACHE_STRATEGY is 'exact' or 'semantic'."""
        allowed = {"exact", "semantic"}
        if v not in allowed:
            raise ValueError(
                f"CACHE_STRATEGY must be one of {allowed}, got '{v}'"
            )
        return v

    @field_validator("CACHE_SEMANTIC_THRESHOLD_LOW", "CACHE_SEMANTIC_THRESHOLD_DIRECT")
    @classmethod
    def validate_semantic_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"semantic threshold must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("REDIS_ENV")
    @classmethod
    def validate_redis_env(cls, v: str) -> str:
        """Ensure REDIS_ENV is a recognised environment profile."""
        allowed = {"local", "cloud", "test", "disabled", ""}
        if v.strip().lower() not in allowed:
            raise ValueError(
                f"REDIS_ENV must be one of {allowed}, got '{v}'"
            )
        return v.strip().lower()


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (constructed once per process)."""
    return Settings()


settings = get_settings()
