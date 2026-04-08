"""
CacheEntry — the value stored in every cache backend.

Wraps LLMResponse with cache-specific metadata:
    - When it was created and when it expires
    - Which query produced it (for deduplication on write)
    - Hit counter for frequency-based eviction decisions
    - Provider + model for cost-aware cache routing

Stored as JSON via json_serializer.py.
Deserialized back into this model on cache read.

Sync — pure Pydantic data class, no I/O."""


from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from llm import LLMResponse


class CacheEntry(BaseModel):
    """Single cached LLM response with metadata."""

    model_config = {"strict": False, "frozen": False}

    response: LLMResponse

    sources: List[dict] = Field(
        default_factory=list,
        description="Serialized RetrievedChunk dicts from the RAG pipeline",
    )

    cache_key: str = Field(
        ...,
        min_length=1,
        description="The key this entry is stored under (hash or vector ID)",
    )

    query_hash: str = Field(
        ...,
        min_length=1,
        description="SHA-256 of normalized query for dedup matching",
    )

    query_text: Optional[str] = Field(
        default=None,
        description="Normalized query text — used to reseed semantic Qdrant on restart",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this entry was first cached",
    )
    expires_at: datetime = Field(
        ...,
        description="UTC timestamp when this entry should be evicted",
    )
    ttl_seconds: int = Field(
        ...,
        gt=0,
        description="TTL assigned by the TTL classifier at write time",
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        description="Number of cache hits served from this entry",
    )
    provider: str = Field(
        ...,
        description="LLM provider that generated this response (openai/gemini)",
    )
    model_name: str = Field(
        ...,
        description="Model name that generated this response",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature used for generation",
    )
    token_cost_estimate: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD saved per cache hit",
    )
    confidence_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence score at time of generation (0.0-1.0). "
                    "Stored so cache hits return the original confidence instead of 1.0.",
    )

    @model_validator(mode="after")
    def validate_expiry_after_creation(self) -> "CacheEntry":
        if self.expires_at <= self.created_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must be after "
                f"created_at ({self.created_at})"
            )
        return self

    @property
    def is_expired(self) -> bool:
        """Check if this entry has passed its TTL."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was created."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds()

    def record_hit(self) -> None:
        """Increment hit counter. Called on every cache read."""
        self.hit_count += 1
