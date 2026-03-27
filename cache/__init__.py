"""
Cache layer — multi-tier LLM response caching with hybrid strategy.

Public API:
    CacheManager    — orchestrator, the only class callers import
    CacheResult     — return type from cache lookups
    CacheEntry      — stored value model
    CacheMetrics    — observability counters

Usage:
    from cache import CacheManager, CacheResult

    cache = CacheManager(settings)
    await cache.initialize()

    result: CacheResult = await cache.get(query, model, temperature)
    if result.hit:
        return result.response

    llm_response = await llm.generate(query)
    await cache.set(query, model, temperature, llm_response)
"""

from cache.cache_manager import CacheManager
from cache.models.cache_entry import CacheEntry
from cache.models.cache_result import (
    CacheResult,
    CacheLayer,
    CacheStrategy,
    SemanticTier,
)
from cache.models.cache_metrics import CacheMetrics

__all__ = [
    "CacheManager",
    "CacheEntry",
    "CacheResult",
    "CacheLayer",
    "CacheStrategy",
    "SemanticTier",
    "CacheMetrics",
]