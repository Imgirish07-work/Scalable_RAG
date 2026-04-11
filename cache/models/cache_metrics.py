"""
Dataclass tracking cumulative cache performance counters.

Design:
    Pydantic model (mutable, not frozen) that accumulates hit/miss/write
    counts, token savings, cost savings, latency sums, and error counts.
    Held in memory by CacheManager for the lifetime of the process.
    Thread-safe for the GIL: individual int/float increments are atomic
    under CPython, so no locks are needed for these counters.

    Latency fields accumulate totals so that averages can be computed
    on demand in avg_lookup_latency_ms and avg_write_latency_ms, avoiding
    division at record time.

Chain of Responsibility:
    Owned by CacheManager. Updated via record_hit(), record_miss(),
    record_write(), record_quality_rejection(), and record_error().
    Exposed through CacheManager.get_metrics() → summary() for logging,
    dashboards, or the /cache/stats API endpoint.

Dependencies:
    pydantic
"""

from pydantic import BaseModel, Field


class CacheMetrics(BaseModel):
    """Cumulative cache performance counters.

    Attributes:
        total_lookups: Total get() calls (hits + misses).
        total_hits: Cache hits across all layers and strategies.
        total_misses: Cache misses (LLM call required).
        l1_hits: Hits served from L1 in-memory backend.
        l2_hits: Hits served from L2 Redis backend.
        exact_hits: Hits from SHA-256 exact-match strategy.
        semantic_hits: Hits from BGE semantic similarity strategy.
        total_writes: Successful set() calls.
        quality_gate_rejections: Writes blocked by QualityGate.
        dedup_replacements: Writes that overwrote an existing entry.
        total_tokens_saved: Cumulative tokens not sent to the LLM.
        total_cost_saved_usd: Cumulative estimated USD saved.
        total_lookup_latency_ms: Sum of all lookup latencies (for avg).
        total_write_latency_ms: Sum of all write latencies (for avg).
        l1_errors: Errors in L1 backend operations.
        l2_errors: Errors in L2 backend operations.
        serialization_errors: Serialization/deserialization failures.
    """

    model_config = {"strict": False, "frozen": False}

    # Hit / miss counters
    total_lookups: int = Field(default=0, ge=0)
    total_hits: int = Field(default=0, ge=0)
    total_misses: int = Field(default=0, ge=0)

    l1_hits: int = Field(default=0, ge=0)
    l2_hits: int = Field(default=0, ge=0)

    exact_hits: int = Field(default=0, ge=0)
    semantic_hits: int = Field(default=0, ge=0)

    # Write path counters
    total_writes: int = Field(default=0, ge=0)
    quality_gate_rejections: int = Field(default=0, ge=0)
    dedup_replacements: int = Field(default=0, ge=0)

    # Cost savings
    total_tokens_saved: int = Field(default=0, ge=0)
    total_cost_saved_usd: float = Field(default=0.0, ge=0.0)

    # Latency accumulators (for computing averages)
    total_lookup_latency_ms: float = Field(default=0.0, ge=0.0)
    total_write_latency_ms: float = Field(default=0.0, ge=0.0)

    # Error counters
    l1_errors: int = Field(default=0, ge=0)
    l2_errors: int = Field(default=0, ge=0)
    serialization_errors: int = Field(default=0, ge=0)

    def record_hit(
        self,
        layer: str,
        strategy: str,
        tokens_saved: int,
        cost_saved: float,
        latency_ms: float,
    ) -> None:
        """Record a successful cache hit with savings and latency.

        Args:
            layer: Backend that served the hit ('l1_memory' or 'l2_redis').
            strategy: Strategy that matched ('exact' or 'semantic').
            tokens_saved: Tokens avoided by serving from cache.
            cost_saved: Estimated USD cost avoided.
            latency_ms: Time spent on the cache lookup.
        """
        self.total_lookups += 1
        self.total_hits += 1
        self.total_tokens_saved += tokens_saved
        self.total_cost_saved_usd += cost_saved
        self.total_lookup_latency_ms += latency_ms

        if layer == "l1_memory":
            self.l1_hits += 1
        elif layer == "l2_redis":
            self.l2_hits += 1

        if strategy == "exact":
            self.exact_hits += 1
        elif strategy == "semantic":
            self.semantic_hits += 1

    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss.

        Args:
            latency_ms: Time spent on the failed cache lookup.
        """
        self.total_lookups += 1
        self.total_misses += 1
        self.total_lookup_latency_ms += latency_ms

    def record_write(self, latency_ms: float) -> None:
        """Record a successful cache write.

        Args:
            latency_ms: Time spent on the write operation.
        """
        self.total_writes += 1
        self.total_write_latency_ms += latency_ms

    def record_quality_rejection(self) -> None:
        """Record a write rejected by the quality gate."""
        self.quality_gate_rejections += 1

    def record_error(self, component: str) -> None:
        """Record an error in a specific cache component.

        Args:
            component: One of 'l1', 'l2', or 'serialization'.
        """
        if component == "l1":
            self.l1_errors += 1
        elif component == "l2":
            self.l2_errors += 1
        elif component == "serialization":
            self.serialization_errors += 1

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate as a percentage."""
        if self.total_lookups == 0:
            return 0.0
        return round((self.total_hits / self.total_lookups) * 100, 2)

    @property
    def avg_lookup_latency_ms(self) -> float:
        """Average lookup latency in milliseconds."""
        if self.total_lookups == 0:
            return 0.0
        return round(self.total_lookup_latency_ms / self.total_lookups, 2)

    @property
    def avg_write_latency_ms(self) -> float:
        """Average write latency in milliseconds."""
        if self.total_writes == 0:
            return 0.0
        return round(self.total_write_latency_ms / self.total_writes, 2)

    def summary(self) -> dict:
        """Snapshot of key metrics for logging or API response.

        Returns:
            Dict with hit rate, counts, savings, latencies, and errors.
        """
        return {
            "hit_rate_pct": self.hit_rate,
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "total_tokens_saved": self.total_tokens_saved,
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
            "avg_lookup_latency_ms": self.avg_lookup_latency_ms,
            "avg_write_latency_ms": self.avg_write_latency_ms,
            "quality_gate_rejections": self.quality_gate_rejections,
            "errors": {
                "l1": self.l1_errors,
                "l2": self.l2_errors,
                "serialization": self.serialization_errors,
            },
        }
