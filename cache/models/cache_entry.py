"""
Pydantic model for a single cached LLM response with full metadata.

Design:
    Wraps LLMResponse with cache-specific fields needed for lifecycle
    management: expiry timestamps, hit counting, provider/model info
    for cost attribution, and the original query text for semantic
    re-seeding on restart. Stored as JSON via JSONSerializer and
    deserialized back into this model on every cache read.

    model_validator enforces that expires_at is strictly after created_at
    to prevent zero-or-negative TTL entries from entering the cache.

Chain of Responsibility:
    Created by CacheManager.set() on the write path. Deserialized by
    JSONSerializer and returned to CacheManager on the read path.
    TTLClassifier determines ttl_seconds. CacheEntry.is_expired guards
    stale reads in both MemoryCacheBackend and CacheManager.

Dependencies:
    pydantic, llm.LLMResponse
"""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from llm import LLMResponse


class CacheEntry(BaseModel):
    """Single cached LLM response with metadata.

    Attributes:
        response: The LLM-generated response being cached.
        sources: Serialized RetrievedChunk dicts from the RAG pipeline.
        cache_key: The key this entry is stored under in the backend.
        query_hash: SHA-256 of the normalized query for dedup matching.
        query_text: Normalized query text for semantic Qdrant re-seeding.
        created_at: UTC timestamp when this entry was first written.
        expires_at: UTC timestamp when this entry should be evicted.
        ttl_seconds: TTL assigned by TTLClassifier at write time.
        hit_count: Number of cache hits served from this entry.
        provider: LLM provider that generated this response.
        model_name: Model name used for generation.
        temperature: Temperature used for generation.
        token_cost_estimate: Estimated USD cost saved per cache hit.
        confidence_value: Retrieval confidence score at generation time.
    """

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
        """Reject entries where expires_at is not strictly after created_at."""
        if self.expires_at <= self.created_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must be after "
                f"created_at ({self.created_at})"
            )
        return self

    @property
    def is_expired(self) -> bool:
        """True if the current UTC time has passed expires_at."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since this entry was created."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds()

    def record_hit(self) -> None:
        """Increment hit counter. Called on every successful cache read."""
        self.hit_count += 1
