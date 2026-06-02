"""Backend HTTP-layer settings, separate from pipeline settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class BackendSettings(BaseSettings):
    """Env-driven settings for the FastAPI layer."""

    # no default — must be set via BACKEND_DATABASE_URL so credentials never land in the repo
    database_url: str = Field(default="")

    cors_origins: str = Field(default="*")
    max_upload_size_mb: int = Field(default=50)
    allow_insecure_tls: bool = Field(default=False)
    ingest_temp_dir: str = Field(default="./data/uploads")
    max_concurrent_subqueries: int = Field(default=3)
    log_requests: bool = Field(default=True)

    model_config = {
        "env_prefix": "BACKEND_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_backend_settings() -> BackendSettings:
    return BackendSettings()


backend_settings = get_backend_settings()


class StorageSettings(BaseSettings):
    """Object-store config for MinIO / S3-compatible backends; uses standard S3 env names so boto3 auto-discovers AWS creds."""

    s3_endpoint: str = Field(default="http://minio:9000")
    s3_public_endpoint: str = Field(default="")
    s3_bucket: str = Field(default="scalable-rag-documents")
    s3_region: str = Field(default="us-east-1")
    s3_use_ssl: bool = Field(default=False)

    # comma-separated; applied to the bucket so browsers can PUT directly
    s3_cors_origins: str = Field(default="*")

    # presigned URL lifetime matches the orphan-cleanup window so expired urls guarantee a sweep
    presigned_url_ttl_seconds: int = Field(default=900)
    orphan_sweep_after_seconds: int = Field(default=900)

    sweeper_interval_seconds: int = Field(default=300)

    failed_dlq_ttl_seconds: int = Field(default=7 * 24 * 60 * 60)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def s3_cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.s3_cors_origins.split(",") if o.strip()]

    @property
    def effective_public_endpoint(self) -> str:
        """Endpoint to embed in presigned URLs; falls back to internal endpoint."""
        return self.s3_public_endpoint or self.s3_endpoint


@lru_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()


storage_settings = get_storage_settings()


class WorkerSettings(BaseSettings):
    """Arq worker + Redis pub/sub config for async ingestion."""

    redis_url: str = Field(default="redis://redis:6379/0")
    queue_name: str = Field(default="scalable_rag:ingest")
    max_jobs: int = Field(default=2)
    job_timeout_seconds: int = Field(default=900)

    # arq in-worker retries for transient blips; exhaustion → DLQ
    arq_max_tries: int = Field(default=3)

    # past this, the sweeper treats the row as a dead worker and moves it to DLQ
    processing_lease_ttl_seconds: int = Field(default=600)

    # per-chunk progress throttle — whichever fires first
    progress_publish_every_n_chunks: int = Field(default=5)
    progress_publish_min_interval_ms: int = Field(default=200)

    events_channel_prefix: str = Field(default="events:doc")

    model_config = {
        "env_prefix": "WORKER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def events_channel(self, doc_id: str) -> str:
        return f"{self.events_channel_prefix}:{doc_id}"


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()


worker_settings = get_worker_settings()
