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
