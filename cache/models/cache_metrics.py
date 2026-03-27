"""
CacheMetrics — cumulative observability counters for the cache layer.

Tracks:
    - Hit/miss counts per layer and strategy
    - Total tokens and estimated cost saved
    - Latency distributions for cache lookups
    - Write-path quality gate rejections

These metrics are held in memory by cache_manager.py and can be
queried via get_metrics() for logging, dashboards, or the
/cache/stats API endpoint (Layer 10).

Thread-safe: all mutations are simple increments on ints/floats,
and Python's GIL makes individual attribute increments atomic.

Sync — pure Pydantic data class, no I/O.
"""

from pydantic import BaseModel, Field

class CacheMetrics(BaseModel):
    """Cumulative cache performance counters."""

    model_config = {"strict": False, "frozen": False}

    # --- Hit / miss counters ---
    total_lookups: int = Field(default=0, ge=0)
    total_hits: int = Field(default=0, ge=0)
    total_misses: int = Field(default=0, ge=0)

    l1_hits: int = Field(default=0, ge=0)
    l2_hits: int = Field(default=0, ge=0)

    exact_hits: int = Field(default=0, ge=0)
    semantic_hits: int = Field(default=0, ge=0)

    # --- Write path counters ---
    total_writes: int = Field(default=0, ge=0)
    quality_gate_rejections: int = Field(default=0, ge=0)
    dedup_replacements: int = Field(default=0, ge=0)

    # --- Cost savings ---
    total_tokens_saved: int = Field(default=0, ge=0)
    total_cost_saved_usd: float = Field(default=0.0, ge=0.0)

    # --- Latency accumulators (for computing averages) ---
    total_lookup_latency_ms: float = Field(default=0.0, ge=0.0)
    total_write_latency_ms: float = Field(default=0.0, ge=0.0)

    # --- Error counters ---
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
        """Record a successful cache hit."""
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
        """Record a cache miss."""
        self.total_lookups += 1
        self.total_misses += 1
        self.total_lookup_latency_ms += latency_ms

    def record_write(self, latency_ms: float) -> None:
        """Record a successful cache write."""
        self.total_writes += 1
        self.total_write_latency_ms += latency_ms

    def record_quality_rejection(self) -> None:
        """Record a write rejected by the quality gate."""
        self.quality_gate_rejections += 1

    def record_error(self, component: str) -> None:
        """Record an error in a specific component."""
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
        """Snapshot of key metrics for logging or API response."""
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