"""
Central configuration for the Scalable RAG RLM application.

Design:
    Pydantic BaseSettings class that reads all configuration from environment
    variables and a .env file. Field validators enforce business rules (e.g.,
    provider names, threshold ranges). A singleton `settings` object is
    exported at module level so every module gets the same parsed config.

Chain of Responsibility:
    Loaded at startup by nearly every module via `from config.settings import settings`.
    Reads from .env → validates → exposes as typed attributes. No downstream calls.

Dependencies:
    pydantic_settings, pydantic
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and .env.

    Attributes:
        openai_api_key: OpenAI API key (optional).
        gemini_api_key: Google Gemini API key (optional).
        groq_api_key: Groq API key (optional).
        openai_model: OpenAI model name to use.
        gemini_model: Gemini model name to use.
        default_provider: Active LLM provider ('openai', 'gemini', or 'groq').
        GROQ_MODEL_FAST: Groq model for fast, lightweight tasks.
        GROQ_MODEL_STRONG: Groq model for complex final-answer generation.
        GROQ_MODEL_FALLBACK: Groq model used when the strong model hits rate limits.
        temperature: Sampling temperature for LLM responses.
        max_tokens: Maximum tokens in a single LLM response.
        request_timeout: Default LLM request timeout in seconds.
        GROQ_TIMEOUT: Groq-specific timeout; kept short to fail fast on blocked networks.
        embedding_model: HuggingFace model ID for dense embeddings.
        embedding_model_local_path: Local directory path for the embedding model.
        USE_ONNX_EMBEDDINGS: If True, uses ONNX Runtime for faster CPU inference.
        EMBEDDING_BATCH_SIZE: Texts per ONNX forward pass during embedding.
        INGESTION_BATCH_SIZE: Chunks per outer Qdrant upsert batch.
        SPLADE_LOCAL_PATH: Local path for the SPLADE sparse model (skips HF download).
        SPLADE_INTRA_OP_THREADS: ONNX Runtime intra-op threads for SPLADE inference.
        SPLADE_BATCH_SIZE: Texts per SPLADE ONNX forward pass (VRAM budget-sensitive).
        min_chars_per_page: Pages with fewer chars than this are treated as empty.
        prefer_pdfplumber: If True, prefers pdfplumber over PyMuPDF for PDF parsing.
        chunk_size: Target token count per chunk.
        chunk_overlap: Token overlap between consecutive chunks.
        code_chunk_overlap: Token overlap used specifically for code chunks.
        min_chunk_tokens: Chunks shorter than this are discarded.
        qdrant_collection_name: Qdrant collection name for the RAG index.
        qdrant_url: Qdrant server URL.
        qdrant_api_key: Qdrant API key for authenticated cloud deployments.
        QDRANT_PREFER_GRPC: If True, uses gRPC transport (faster; may be blocked on VPNs).
        max_tokens_per_chunk: Token limit per chunk in RLM processing.
        max_recursion_depth: Maximum recursion depth for the RLM pipeline.
        effective_context_limit: Token budget for the assembled LLM context.
        top_k_retrieval: Default number of chunks to retrieve per query.
        RERANKER_ENABLED: If True, activates cross-encoder reranking after retrieval.
        RERANKER_MODEL_PATH: Local path to the standard ONNX reranker export.
        RERANKER_MODEL_PATH_CUDA: Local path to the CUDA-native ONNX reranker export.
        RERANKER_BATCH_SIZE: (Query, chunk) pairs per reranker forward pass.
        RERANKER_COARSE_TOP_K: Qdrant candidates fetched before reranking.
        RERANKER_SCORE_THRESHOLD: Minimum reranker score; below this = no useful context.
        RERANKER_SCORE_RATIO: Relative score floor as a fraction of the top chunk's score.
        RERANKER_MIN_ABS_FLOOR: Absolute score floor applied when all chunks score low.
        RERANKER_PREFILTER_TOP_N: Max candidates sent to the cross-encoder after pre-filtering.
        RERANKER_INTRA_OP_THREADS: ONNX Runtime intra-op threads for the reranker session.
        cache_enabled: Master switch for all caching layers.
        cache_directory: Filesystem path for the L2 disk cache.
        cache_ttl_seconds: Default time-to-live for cached entries.
        CACHE_L1_MAX_SIZE: Maximum entries in the in-process LRU cache.
        REDIS_ENV: Redis environment profile ('local', 'cloud', 'test', or 'disabled').
        REDIS_URL: Redis connection URL for the local profile.
        REDIS_CLOUD_URL: Redis Cloud connection URL (TLS) for the cloud profile.
        REDIS_MAX_CONNECTIONS: Max connections in the async Redis connection pool.
        REDIS_SOCKET_TIMEOUT: Timeout in seconds for individual Redis operations.
        REDIS_RETRY_ON_TIMEOUT: Whether to retry Redis ops that time out.
        CACHE_STRATEGY: Active cache strategy ('exact' or 'semantic').
        CACHE_SEMANTIC_THRESHOLD: Cosine similarity floor for a semantic cache hit.
        CACHE_SEMANTIC_THRESHOLD_HIGH: Upper semantic similarity tier threshold.
        CACHE_SEMANTIC_THRESHOLD_PARTIAL: Partial-match semantic similarity tier threshold.
        CACHE_SEMANTIC_COLLECTION: Qdrant collection name for the semantic cache index.
        CACHE_CIRCUIT_BREAKER_THRESHOLD: Consecutive failures before circuit opens.
        CACHE_CIRCUIT_BREAKER_RESET_SECONDS: Seconds before a tripped circuit retries.
        CACHE_MIN_RESPONSE_TOKENS: Minimum tokens in a response before it is cached.
        CACHE_MIN_RESPONSE_LATENCY_MS: Minimum LLM latency before a response is cached.
        RAG_DEFAULT_VARIANT: Default RAG variant used by the factory.
        RAG_TOP_K: Default top-k for the RAG retrieval layer.
        RAG_MAX_CONTEXT_TOKENS: Token budget for assembled RAG context.
        RAG_RERANK_STRATEGY: Post-retrieval reranking strategy ('mmr', 'cross_encoder').
        RAG_RETRIEVAL_MODE: Qdrant search mode ('dense', 'sparse', 'hybrid').
        RAG_CONFIDENCE_METHOD: Method used to compute retrieval confidence score.
        COST_PER_TOKEN_OPENAI: Approximate cost per token for OpenAI (gpt-3.5-turbo).
        COST_PER_TOKEN_GEMINI: Approximate cost per token for Gemini (2.5-flash).
        COST_PER_TOKEN_GROQ: Approximate cost per token for Groq (free tier).
        LLM_RATE_LIMITER_ENABLED: Enables the per-model rate limiter.
        LLM_MAX_CONCURRENT: Semaphore cap for simultaneous in-flight LLM requests.
        LLM_BURST_MULTIPLIER: Burst multiplier above sustained RPM (1.0 = no burst).
        use_cheap_model_threshold: Token count below which the cheaper model is used.
        llm_max_retries: Maximum LLM call retry attempts.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: File path for the rotating log file.
        app_name: Application display name.
        app_version: Application version string.
        debug: Enables debug mode (verbose output, relaxed validation).
        host: Server bind host address.
        port: Server bind port.
    """

    # LLM Providers
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")

    # LLM Model Names
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    default_provider: str = Field(default="gemini", env="DEFAULT_PROVIDER")

    # Groq models — each role has a different RPD budget.
    # Fast model: classify, decompose, filter, answer simple queries (RPD: 14,400)
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    # Strong model: final answer generation for complex queries (RPD: 1,000)
    GROQ_MODEL_STRONG: str = "llama-3.3-70b-versatile"
    # Fallback: used when the strong model hits a 429 rate-limit (RPD: 1,000, RPM: 60)
    GROQ_MODEL_FALLBACK: str = "qwen/qwen3-32b"

    # LLM Parameters
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")
    GROQ_TIMEOUT: float = Field(default=30.0, env="GROQ_TIMEOUT")

    # Embedding Model
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        env="EMBEDDING_MODEL",
    )
    embedding_model_local_path: str = Field(
        default="models/bge-base-en-v1.5",
        env="EMBEDDING_MODEL_LOCAL_PATH",
    )
    # ONNX embeddings: faster CPU inference via onnx/model.onnx inside the model folder.
    USE_ONNX_EMBEDDINGS: bool = Field(default=True, env="USE_ONNX_EMBEDDINGS")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, env="EMBEDDING_BATCH_SIZE")

    # Outer ingestion batch: how many chunks to embed + upsert per Qdrant call.
    # Dense embedding (ONNX) has its own inner batch of EMBEDDING_BATCH_SIZE.
    # This outer batch controls memory pressure and enables progress logging.
    # Rule of thumb: 200 for cloud Qdrant, 500 for local Qdrant.
    INGESTION_BATCH_SIZE: int = Field(default=200, env="INGESTION_BATCH_SIZE")

    # SPLADE sparse embedding model local path.
    # When set, fastembed skips HuggingFace download entirely (corporate network fix).
    # Download model.onnx + tokenizer files from:
    #   https://huggingface.co/Qdrant/Splade_PP_en_v1/tree/main
    # Place in models/splade-en-v1/ and point this setting to that directory.
    SPLADE_LOCAL_PATH: str = Field(default="", env="SPLADE_LOCAL_PATH")
    # ONNX Runtime intra-op threads for SPLADE inference.
    # Controls CPU cores used per batch — ORT default is 1 thread.
    # i5-1345U (2P+8E): 6 is optimal. Set 0 to use all logical cores.
    SPLADE_INTRA_OP_THREADS: int = Field(default=6, env="SPLADE_INTRA_OP_THREADS")
    # SPLADE ONNX batch size: controls texts per single ONNX forward pass.
    # MLM head output shape: (batch, seq_len, vocab_size) = (batch, 512, 30522).
    #   batch=100 → 6.27 GB — exceeds 4 GB VRAM → silent CPU fallback (~53s/batch)
    #   batch=16  → 1.00 GB — fits in 4 GB VRAM → full GPU execution (~2-4s/batch)
    # Increase to 32 on GPUs with 8+ GB VRAM.
    SPLADE_BATCH_SIZE: int = Field(default=16, env="SPLADE_BATCH_SIZE")

    # Document cleaner settings
    min_chars_per_page: int = Field(default=50, env="MIN_CHARS_PER_PAGE")
    prefer_pdfplumber: bool = Field(default=False, env="PREFER_PDFPLUMBER")

    # Chunker settings
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    code_chunk_overlap: int = Field(default=150, env="CODE_CHUNK_OVERLAP")
    min_chunk_tokens: int = Field(default=20, env="MIN_CHUNK_TOKENS")

    # Qdrant Settings
    qdrant_collection_name: str = Field(
        default="rag_collection",
        env="QDRANT_COLLECTION_NAME",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL",
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    # gRPC transport is ~24% faster than HTTP (235ms vs 310ms per call).
    # Set False on networks where port 6334 is blocked (e.g., corporate Zscaler).
    # Falls back to HTTP automatically if the gRPC connection fails at startup.
    QDRANT_PREFER_GRPC: bool = Field(default=True, env="QDRANT_PREFER_GRPC")

    # RLM Settings
    max_tokens_per_chunk: int = Field(default=500, env="MAX_TOKENS_PER_CHUNK")
    max_recursion_depth: int = Field(default=5, env="MAX_RECURSION_DEPTH")
    effective_context_limit: int = Field(default=8000, env="EFFECTIVE_CONTEXT_LIMIT")

    # RAG Settings
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")

    # Reranker Settings
    RERANKER_ENABLED: bool = Field(default=False, env="RERANKER_ENABLED")
    RERANKER_MODEL_PATH: str = Field(default="", env="RERANKER_MODEL_PATH")
    # CUDA-native ONNX export — no Memcpy nodes, ~3x faster than standard export.
    # Generate with: optimum-cli export onnx --device cuda --optimize O3
    # Auto-selected when CUDA is available and this path exists.
    # Falls back to RERANKER_MODEL_PATH on CPU-only machines.
    RERANKER_MODEL_PATH_CUDA: str = Field(default="", env="RERANKER_MODEL_PATH_CUDA")
    RERANKER_BATCH_SIZE: int = Field(default=32, env="RERANKER_BATCH_SIZE")
    # Should be 2-3x top_k so the reranker has enough candidates to work with.
    RERANKER_COARSE_TOP_K: int = Field(default=10, env="RERANKER_COARSE_TOP_K")
    # Minimum cross-encoder score to consider a retrieval useful.
    # If the top chunk scores below this after reranking, the pipeline returns
    # a transparent "no relevant context" message instead of hallucinating.
    # Range: 0.0-1.0 (sigmoid). 0.1 rejects near-zero-confidence retrievals.
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.1, env="RERANKER_SCORE_THRESHOLD")
    # Relative score filtering — adapts to each query's score distribution.
    # A chunk is kept only if: score >= top_score * RATIO  AND  score >= MIN_ABS_FLOOR
    # RATIO=0.4 means a chunk must score at least 40% of the best chunk's score.
    # Example: top=0.984 → threshold=0.394. top=0.200 → threshold=0.100 (floor kicks in).
    RERANKER_SCORE_RATIO: float = Field(default=0.7, env="RERANKER_SCORE_RATIO")
    # Absolute floor — safety net when all chunks score low (poor retrieval).
    # Prevents keeping chunks that score above ratio but are still near-zero quality.
    RERANKER_MIN_ABS_FLOOR: float = Field(default=0.1, env="RERANKER_MIN_ABS_FLOOR")
    # Pre-filter: keep only top-N candidates by RRF rank before cross-encoding.
    # Drops bottom (RERANKER_COARSE_TOP_K - RERANKER_PREFILTER_TOP_N) chunks that
    # ranked last in both dense + sparse retrieval — cross-encoder has never rescued
    # these in practice. Set equal to RERANKER_COARSE_TOP_K to disable pre-filtering.
    RERANKER_PREFILTER_TOP_N: int = Field(default=8, env="RERANKER_PREFILTER_TOP_N")
    # 0 = let ORT decide (usually 1). 4-6 is optimal for i5/i7 CPUs.
    RERANKER_INTRA_OP_THREADS: int = Field(default=4, env="RERANKER_INTRA_OP_THREADS")

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_directory: str = Field(default="./data/cache", env="CACHE_DIRECTORY")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # Cache L1 (in-memory LRU)
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

    # Cache strategy
    CACHE_STRATEGY: str = "exact"
    """Active cache strategy: 'exact' or 'semantic'.
    Start with 'exact', switch to 'semantic' after Phase 6."""

    # Semantic cache
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

    # Cache circuit breaker
    CACHE_CIRCUIT_BREAKER_THRESHOLD: int = 5
    """Consecutive failures before opening the circuit breaker.
    Once open, all requests skip the failed backend for the
    reset period."""

    CACHE_CIRCUIT_BREAKER_RESET_SECONDS: float = 60.0
    """Seconds to wait before retrying a tripped circuit breaker."""

    # Cache quality gate
    CACHE_MIN_RESPONSE_TOKENS: int = 20
    """Minimum response tokens to cache. Shorter responses are
    likely errors, refusals, or degenerate outputs."""

    CACHE_MIN_RESPONSE_LATENCY_MS: float = 100.0
    """Minimum LLM latency to cache. Sub-100ms responses are
    likely errors returning instantly, not real generations."""

    # RAG layer configuration
    RAG_DEFAULT_VARIANT: str = "simple"
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_TOKENS: int = 3072
    RAG_RERANK_STRATEGY: str = "mmr"
    RAG_RETRIEVAL_MODE: str = "hybrid"
    RAG_CONFIDENCE_METHOD: str = "retrieval"


    # Cost per token — used to estimate savings from cache hits
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
    # RPM and RPD limits are looked up per-model from llm/rate_limiter/model_limits.py
    # so the correct limits are always applied regardless of which model is active.
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
        """Ensure chunk_overlap is strictly less than chunk_size."""
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

    @field_validator("CACHE_SEMANTIC_THRESHOLD")
    @classmethod
    def validate_semantic_threshold(cls, v: float) -> float:
        """Ensure CACHE_SEMANTIC_THRESHOLD is within [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"CACHE_SEMANTIC_THRESHOLD must be between 0.0 and 1.0, got {v}"
            )
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
