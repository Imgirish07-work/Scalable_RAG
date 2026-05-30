"""Backend HTTP-layer settings, separate from pipeline settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class BackendSettings(BaseSettings):
    """Env-driven settings for the FastAPI layer."""

    # No default — must be set explicitly via BACKEND_DATABASE_URL so the
    # repository never contains a connection string with embedded credentials.
    database_url: str = Field(default="")

    cors_origins: str = Field(default="*")
    max_upload_size_mb: int = Field(default=50)
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
    """Object-store config for MinIO / S3-compatible backends.

    Uses standard S3 env var names (no `BACKEND_` prefix) so the underlying
    boto3 client auto-discovers `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
    from the same environment without explicit plumbing.
    """

    s3_endpoint: str = Field(default="http://minio:9000")
    s3_bucket: str = Field(default="scalable-rag-documents")
    s3_region: str = Field(default="us-east-1")
    s3_use_ssl: bool = Field(default=False)

    # Comma-separated; applied to the bucket so browsers can PUT directly.
    s3_cors_origins: str = Field(default="*")

    # Presigned URL lifetime; matches the orphan-cleanup window so an expired
    # URL guarantees the row will be swept rather than left stuck at pending.
    presigned_url_ttl_seconds: int = Field(default=900)
    orphan_sweep_after_seconds: int = Field(default=900)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def s3_cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.s3_cors_origins.split(",") if o.strip()]


@lru_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()


storage_settings = get_storage_settings()
