"""Backend HTTP-layer settings, separate from pipeline settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class BackendSettings(BaseSettings):
    """Env-driven settings for the FastAPI layer."""

    # No default — must be set explicitly via BACKEND_DATABASE_URL so the
    # repository never contains a connection string with embedded credentials.
    database_url: str = Field(default="", env="BACKEND_DATABASE_URL")

    cors_origins: str = Field(default="*", env="BACKEND_CORS_ORIGINS")
    max_upload_size_mb: int = Field(default=50, env="BACKEND_MAX_UPLOAD_SIZE_MB")
    ingest_temp_dir: str = Field(default="./data/uploads", env="BACKEND_INGEST_TEMP_DIR")
    max_concurrent_subqueries: int = Field(default=3, env="BACKEND_MAX_CONCURRENT_SUBQUERIES")
    log_requests: bool = Field(default=True, env="BACKEND_LOG_REQUESTS")

    model_config = {
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
