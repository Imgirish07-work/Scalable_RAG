"""
CacheResult — returned by cache_manager.get() to the caller.

Contains:
    - The LLMResponse (or None on miss)
    - Whether it was a hit or miss
    - Which layer served the hit (L1 memory / L2 Redis / miss)
    - Which strategy matched (exact / semantic / none)
    - Similarity score (for semantic hits, useful for observability)
    - Lookup latency (how long the cache check took)

The caller uses `result.hit` to decide whether to call the LLM.
The rest of the fields feed into metrics and logging.

Sync — pure Pydantic data class, no I/O.
"""


from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from llm import LLMResponse

class CacheLayer(str, Enum):
    """Which cache layer served the response."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    MISS = "miss"

class CacheStrategy(str, Enum):
    """Which strategy produced the cache key match."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    NONE = "none"

class SemanticTier(str, Enum):
    """Confidence tier for semantic matches.

    DIRECT  — cosine >= 0.98, serve as-is
    HIGH    — cosine 0.93-0.98, serve with internal flag
    PARTIAL — cosine 0.88-0.93, use as LLM seed (cheaper call)
    MISS    — cosine < 0.88, no semantic match
    """

    DIRECT = "direct"
    HIGH = "high"
    PARTIAL = "partial"
    MISS = "miss"

class CacheResult(BaseModel):
    """Outcome of a cache lookup — returned to the caller."""
    model_config = {"strict": False, "frozen": True}

    hit: bool = Field(
        ...,
        description="True if cache returned a usable response",
    )
    response: Optional[LLMResponse] = Field(
        default=None,
        description="Cached LLM response, None on miss",
    )
    layer: CacheLayer = Field(
        default=CacheLayer.MISS,
        description="Which backend layer served this result",
    )
    strategy: CacheStrategy = Field(
        default=CacheStrategy.NONE,
        description="Which key strategy matched",
    )
    semantic_tier: SemanticTier = Field(
        default=SemanticTier.MISS,
        description="Confidence tier for semantic matches",
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0 for exact or miss)",
    )
    lookup_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time spent on cache lookup in milliseconds",
    )
    cache_age_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Age of the cached entry at read time (0.0 on miss)",
    )
    cache_key: str = Field(
        default="",
        description="The key that matched (empty on miss)",
    )
    sources: List[dict] = Field(
        default_factory=list,
        description="Serialized RetrievedChunk dicts, populated on cache hit",
    )
    confidence_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence stored at write time. Replaces the "
                    "hardcoded 1.0 that was returned on every cache hit.",
    )

    @staticmethod
    def miss(latency_ms: float =0.0) -> "CacheResult":
        """Factory method for cache miss results."""
        return CacheResult(hit=False, lookup_latency_ms=latency_ms)

    @staticmethod
    def from_hit(
        response: LLMResponse,
        layer: CacheLayer,
        strategy: CacheStrategy,
        latency_ms: float,
        cache_key: str,
        sources: List[dict] = [],
        cache_age_seconds: float = 0.0,
        similarity_score: float = 0.0,
        semantic_tier: SemanticTier = SemanticTier.MISS,
        confidence_value: float = 0.0,
    ) -> "CacheResult":
        """Factory for a cache hit with full metadata."""
        return CacheResult(
            hit=True,
            response=response,
            layer=layer,
            strategy=strategy,
            lookup_latency_ms=latency_ms,
            cache_key=cache_key,
            sources=sources,
            cache_age_seconds=cache_age_seconds,
            similarity_score=similarity_score,
            semantic_tier=semantic_tier,
            confidence_value=confidence_value,
        )