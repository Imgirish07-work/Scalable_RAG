from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):

    # LLM Providers
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")

    # LLM Model Names
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    default_provider: str = Field(default="gemini", env="DEFAULT_PROVIDER")

    # Groq Models — role-specific
    # Fast model: classify, decompose, filter, answer simple queries (RPD: 14,400)
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    # Strong model: final answer generation for complex queries (RPD: 1,000)
    GROQ_MODEL_STRONG: str = "llama-3.3-70b-versatile"
    # Fallback #1: if strong model hits 429 (RPD: 1,000, RPM: 60)
    GROQ_MODEL_FALLBACK: str = "qwen/qwen3-32b"

    # LLM Parameters
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")

    # Embedding Model
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        env="EMBEDDING_MODEL",
    )
    embedding_model_local_path: str = Field(
        default="models/bge-base-en-v1.5",
        env="EMBEDDING_MODEL_LOCAL_PATH",
    )
    # ONNX Embeddings — faster CPU inference (uses onnx/model.onnx inside the model folder)
    USE_ONNX_EMBEDDINGS: bool = Field(default=True, env="USE_ONNX_EMBEDDINGS")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, env="EMBEDDING_BATCH_SIZE")

    # Ingestion batching — how many chunks to embed + upsert per Qdrant call.
    # Dense embedding (ONNX) has its own inner batch of EMBEDDING_BATCH_SIZE.
    # This outer batch controls memory pressure and enables progress logging.
    # Rule of thumb: 100 for cloud Qdrant, 200-500 for local Qdrant.
    INGESTION_BATCH_SIZE: int = Field(default=100, env="INGESTION_BATCH_SIZE")

    # SPLADE sparse embedding model local path
    # When set, fastembed skips HuggingFace download entirely (corporate network fix).
    # Download model.onnx + tokenizer files from:
    #   https://huggingface.co/Qdrant/Splade_PP_en_v1/tree/main
    # Place in models/splade-en-v1/ and set this to that directory path.
    SPLADE_LOCAL_PATH: str = Field(default="", env="SPLADE_LOCAL_PATH")
    # ONNX Runtime intra-op threads for SPLADE inference.
    # Controls CPU cores used per SPLADE batch — default ORT behaviour is 1 thread.
    # i5-1345U (2P+8E): 6 is optimal. Set 0 to use all logical cores.
    SPLADE_INTRA_OP_THREADS: int = Field(default=6, env="SPLADE_INTRA_OP_THREADS")

    # Document cleaner settings
    min_chars_per_page : int = Field(default=50, env="MIN_CHARS_PER_PAGE")
    prefer_pdfplumber : bool = Field(default=False, env="PREFER_PDFPLUMBER")

    # Chunker settings
    chunk_size : int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap : int = Field(default=100, env="CHUNK_OVERLAP")
    code_chunk_overlap: int = Field(default=150, env="CODE_CHUNK_OVERLAP")
    min_chunk_tokens : int = Field(default=20, env="MIN_CHUNK_TOKENS")

    # Qdrant Settings
    qdrant_collection_name : str = Field(
        default="rag_collection",
        env="QDRANT_COLLECTION_NAME",
    )
    qdrant_url : str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL",
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # RLM Settings
    max_tokens_per_chunk: int = Field(default=500, env="MAX_TOKENS_PER_CHUNK")
    max_recursion_depth: int = Field(default=5, env="MAX_RECURSION_DEPTH")
    effective_context_limit: int = Field(default=8000, env="EFFECTIVE_CONTEXT_LIMIT")

    # RAG Settings
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")

    # Reranker Settings
    RERANKER_ENABLED: bool = Field(default=False, env="RERANKER_ENABLED")
    RERANKER_MODEL_PATH: str = Field(default="", env="RERANKER_MODEL_PATH")
    RERANKER_BATCH_SIZE: int = Field(default=32, env="RERANKER_BATCH_SIZE")
    # How many coarse candidates to fetch from Qdrant for the cross-encoder to score.
    # Should be 2-3x top_k so the reranker has enough candidates to work with.
    RERANKER_COARSE_TOP_K: int = Field(default=10, env="RERANKER_COARSE_TOP_K")
    # Minimum cross-encoder score to consider a retrieval useful.
    # If the top chunk scores below this after reranking, the pipeline returns
    # a transparent "no relevant context" message instead of hallucinating.
    # Range: 0.0-1.0 (sigmoid). 0.1 rejects near-zero-confidence retrievals.
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.1, env="RERANKER_SCORE_THRESHOLD")
    # Per-chunk cross-encoder score threshold.
    # Chunks scoring below this are dropped even if they are within top_k.
    # 0.0 = disabled (keep all chunks — backward-compatible default).
    # 0.3 = recommended: drops noise while keeping most useful context.
    # Range: 0.0-1.0 (sigmoid). Tune up if irrelevant chunks are present in answers.
    RERANKER_MIN_CHUNK_SCORE: float = Field(default=0.0, env="RERANKER_MIN_CHUNK_SCORE")

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_directory: str = Field(default="./data/cache", env="CACHE_DIRECTORY")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    # Cache L1 (in-memory)
    CACHE_L1_MAX_SIZE: int = 1000
    """Maximum entries in the L1 in-memory LRU cache.
    Per application instance. Higher = more RAM, more hits."""

    # Cache L2 (Redis)
    REDIS_ENV: str = "local"
    """Redis environment: 'local', 'cloud', 'test', or 'disabled'.
        local    — redis://localhost:6379/0, no TLS, dev prefix
        cloud    — reads REDIS_CLOUD_URL, TLS, prod prefix
        test     — redis://localhost:6379/1, test prefix
        disabled — skip Redis entirely, L1 only
    """

    REDIS_URL: str = "redis://localhost:6379/0"
    """Local Redis URL. Used when REDIS_ENV=local."""

    REDIS_CLOUD_URL: str = ""
    """Redis Cloud URL with credentials. Used when REDIS_ENV=cloud.
    Example: rediss://user:password@host:port/0"""

    REDIS_MAX_CONNECTIONS: int = 20
    """Maximum connections in the Redis async connection pool."""

    REDIS_SOCKET_TIMEOUT: float = 2.0
    """Timeout in seconds for individual Redis operations."""

    REDIS_RETRY_ON_TIMEOUT: bool = True
    """Whether to retry Redis operations on timeout."""

    # --- Cache strategy ---
    CACHE_STRATEGY: str = "exact"
    """Active cache strategy: 'exact' or 'semantic'.
    Start with 'exact', switch to 'semantic' after Phase 6."""

    # --- Semantic cache ---
    CACHE_SEMANTIC_THRESHOLD: float = 0.95
    """Minimum cosine similarity for a semantic cache hit.
    Range: 0.0 - 1.0. Start at 0.95, tune based on metrics.
    Only used when CACHE_STRATEGY='semantic'."""

    CACHE_SEMANTIC_THRESHOLD_HIGH: float = 0.93
    """Threshold for 'high confidence' semantic tier.
    Hits between this and CACHE_SEMANTIC_THRESHOLD are flagged
    for monitoring but still served."""

    CACHE_SEMANTIC_THRESHOLD_PARTIAL: float = 0.88
    """Threshold for 'partial' semantic tier.
    Hits between this and CACHE_SEMANTIC_THRESHOLD_HIGH are used
    as LLM seed context (cheaper call, not direct serve)."""

    CACHE_SEMANTIC_COLLECTION: str = "cache_semantic"
    """Qdrant collection name for the semantic cache index.
    Separate from your RAG collection to avoid pollution."""

    # --- Cache circuit breaker ---
    CACHE_CIRCUIT_BREAKER_THRESHOLD: int = 5
    """Consecutive failures before opening the circuit breaker.
    Once open, all requests skip the failed backend for the
    reset period."""

    CACHE_CIRCUIT_BREAKER_RESET_SECONDS: float = 60.0
    """Seconds to wait before retrying a tripped circuit breaker."""

    # --- Cache quality gate ---
    CACHE_MIN_RESPONSE_TOKENS: int = 20
    """Minimum response tokens to cache. Shorter responses are
    likely errors, refusals, or degenerate outputs."""

    CACHE_MIN_RESPONSE_LATENCY_MS: float = 100.0
    """Minimum LLM latency to cache. Sub-100ms responses are
    likely errors returning instantly, not real generations."""

    # --- RAG Configuration ---
    # RAG layer
    RAG_DEFAULT_VARIANT: str = "simple"
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_TOKENS: int = 3072
    RAG_RERANK_STRATEGY: str = "mmr"
    RAG_RETRIEVAL_MODE: str = "hybrid"
    RAG_CONFIDENCE_METHOD: str = "retrieval"

    # CorrectiveRAG specific
    CRAG_RELEVANCE_THRESHOLD_PASS: float = 0.7
    CRAG_RELEVANCE_THRESHOLD_RETRY: float = 0.4
    CRAG_MAX_RETRIES: int = 1

    # CoRAG (Chain-of-RAG) 
    CHAIN_RAG_MAX_HOPS: int = 3
    CHAIN_RAG_DRAFT_MAX_TOKENS: int = 512
    CHAIN_RAG_COMPLETENESS_MAX_TOKENS: int = 512

    # --- Cost per token (for savings calculation) ---
    COST_PER_TOKEN_OPENAI: float = 0.000002
    """Approximate cost per token for OpenAI (gpt-3.5-turbo).
    Used to estimate cost savings from cache hits."""

    COST_PER_TOKEN_GEMINI: float = 0.0000001
    """Approximate cost per token for Gemini (2.5-flash).
    Used to estimate cost savings from cache hits."""

    COST_PER_TOKEN_GROQ: float = 0.0
    """Approximate cost per token for Groq (free tier — $0).
    Used to estimate cost savings from cache hits."""

    # Rate Limiter
    # RPM and RPD are NOT configured here — they are looked up per-model from
    # llm/rate_limiter/model_limits.py so the correct provider limits are
    # always applied regardless of which model is active.
    LLM_RATE_LIMITER_ENABLED: bool = True
    LLM_MAX_CONCURRENT: int = 5       # semaphore — max in-flight requests
    LLM_BURST_MULTIPLIER: float = 1.0  # 1.0 = no burst above sustained RPM

    # Cost Optimization
    use_cheap_model_threshold: int = Field(default=500, env="USE_CHEAP_MODEL_THRESHOLD")
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./data/logs/app.log", env="LOG_FILE")

    # Backend
    app_name: str = Field(default="Scalable RAG RLM", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # Validate default_provider 
    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = ["openai", "gemini", "groq"]
        if v.lower() not in allowed:
            raise ValueError(f"default_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    # Validate chunk_overlap < chunk_size
    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )
        return self

    # Use model_config instead of class Config
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("CACHE_STRATEGY")
    @classmethod
    def validate_cache_strategy(cls, v: str) -> str:
        allowed = {"exact", "semantic"}
        if v not in allowed:
            raise ValueError(
                f"CACHE_STRATEGY must be one of {allowed}, got '{v}'"
            )
        return v

    @field_validator("CACHE_SEMANTIC_THRESHOLD")
    @classmethod
    def validate_semantic_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"CACHE_SEMANTIC_THRESHOLD must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("REDIS_ENV")
    @classmethod
    def validate_redis_env(cls, v: str) -> str:
        allowed = {"local", "cloud", "test", "disabled", ""}
        if v.strip().lower() not in allowed:
            raise ValueError(
                f"REDIS_ENV must be one of {allowed}, got '{v}'"
            )
        return v.strip().lower()


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()