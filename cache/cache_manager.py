"""Multi-layer LLM response cache orchestrator composing exact + semantic strategies over L1/L2 backends."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from utils.logger import get_logger
from utils.helpers import hash_text
from config.settings import Settings
from llm.models.llm_response import LLMResponse
from cache.models.cache_entry import CacheEntry
from cache.models.cache_result import (
    CacheResult,
    CacheLayer,
    CacheStrategy,
    SemanticTier,
)
from cache.models.cache_metrics import CacheMetrics
from cache.backend.base_backend import BaseCacheBackend
from cache.backend.memory_backend import MemoryCacheBackend
from cache.backend.redis_backend import RedisCacheBackend
from cache.backend.redis_config import RedisConfigFactory
from cache.strategies.base_strategy import BaseCacheStrategy, SimilarityMatch
from cache.strategies.exact_strategy import ExactCacheStrategy
from cache.strategies.semantic_strategy import SemanticCacheStrategy
from cache.serializers.base_serializer import BaseCacheSerializer
from cache.serializers.json_serializer import JSONSerializer
from cache.normalizers.query_normalizer import QueryNormalizerChain
from cache.quality.ttl_classifier import TTLClassifier
from cache.quality.quality_gate import QualityGate
from cache.exceptions.cache_exceptions import (
    CacheError,
    CacheConnectionError,
    CacheSerializationError,
)

logger = get_logger(__name__)


class CacheManager:
    """Multi-layer LLM response cache orchestrator with hybrid strategy."""

    def __init__(self, settings: Settings) -> None:
        """Sync constructor — builds components, no I/O."""
        self._settings = settings
        self._enabled: bool = settings.cache_enabled
        self._initialized: bool = False

        self._normalizer = QueryNormalizerChain()

        self._l1: MemoryCacheBackend = MemoryCacheBackend(
            max_size=settings.CACHE_L1_MAX_SIZE,
        )
        self._l2: Optional[RedisCacheBackend] = None

        self._exact_strategy: ExactCacheStrategy = ExactCacheStrategy(
            normalizer=self._normalizer,
            backends=[self._l1],
        )
        self._semantic_strategy: Optional[SemanticCacheStrategy] = None

        self._serializer: BaseCacheSerializer = JSONSerializer()
        self._metrics: CacheMetrics = CacheMetrics()
        self._in_flight: dict[str, asyncio.Event] = {}
        self._in_flight_lock = asyncio.Lock()

        self._ttl_classifier = TTLClassifier(
            default_ttl=settings.cache_ttl_seconds,
        )
        self._quality_gate = QualityGate(
            min_tokens=settings.CACHE_MIN_RESPONSE_TOKENS,
            min_latency_ms=settings.CACHE_MIN_RESPONSE_LATENCY_MS,
        )

        logger.info(
            "CacheManager created: enabled=%s, strategy=%s, l1_max_size=%d",
            self._enabled,
            settings.CACHE_STRATEGY,
            settings.CACHE_L1_MAX_SIZE,
        )

    async def initialize(self) -> None:
        """Async initialization — set up all backends and strategies."""
        if self._initialized:
            logger.debug("CacheManager already initialized, skipping")
            return

        await self._initialize_l2()

        backends = [self._l1]
        if self._l2 is not None:
            backends.append(self._l2)

        self._exact_strategy = ExactCacheStrategy(
            normalizer=self._normalizer,
            backends=backends,
        )

        await self._initialize_semantic()

        await self._promote_l2_to_l1()
        await self._seed_semantic_from_l2()

        self._initialized = True

        strategy_name = "hybrid (exact + semantic)" if self._semantic_strategy else "exact"
        logger.info(
            "CacheManager initialized: strategy=%s, backends=[%s]",
            strategy_name,
            ", ".join(self._get_active_backend_names()),
        )

    async def _initialize_l2(self) -> None:
        config = RedisConfigFactory.create(self._settings)

        if config is None:
            logger.info("L2 Redis disabled — no config produced")
            return

        try:
            self._l2 = RedisCacheBackend.from_config(config)
            await self._l2.initialize()
            logger.info(
                "L2 Redis initialized: env=%s, url=%s, prefix='%s'",
                config.environment,
                config.redacted_url,
                config.prefix,
            )
        except CacheConnectionError as e:
            logger.warning("L2 Redis unavailable — L1 only: %s", e.message)
            self._l2 = None
        except Exception as e:
            logger.warning("L2 Redis init failed — L1 only: %s", e)
            self._l2 = None

    async def _initialize_semantic(self) -> None:
        """Initialize semantic strategy if configured; non-fatal on failure."""
        cache_strategy = getattr(self._settings, "CACHE_STRATEGY", "exact")
        if cache_strategy != "semantic":
            logger.info("Semantic strategy disabled — CACHE_STRATEGY='%s'", cache_strategy)
            return

        try:
            collection = getattr(
                self._settings, "CACHE_SEMANTIC_COLLECTION", "cache_semantic"
            )
            threshold_direct = getattr(
                self._settings, "CACHE_SEMANTIC_THRESHOLD_DIRECT", 0.98
            )
            threshold_low = getattr(
                self._settings, "CACHE_SEMANTIC_THRESHOLD_LOW", 0.95
            )
            # in-memory Qdrant required: reusing QDRANT_URL for cache would add ~200ms RTT per lookup
            self._semantic_strategy = await asyncio.to_thread(
                SemanticCacheStrategy,
                collection_name=collection,
                threshold_direct=threshold_direct,
                threshold_low=threshold_low,
                use_memory=True,
            )
            await self._semantic_strategy.initialize()
            logger.info("Semantic strategy initialized: collection='%s'", collection)

        except Exception as e:
            logger.warning(
                "Semantic strategy init failed — falling back to exact-only: %s", e
            )
            self._semantic_strategy = None

    async def _promote_l2_to_l1(self, limit: int = 100) -> None:
        """Warm L1 from L2 at startup to eliminate post-restart cold cache."""
        if self._l2 is None:
            return

        try:
            keys = await self._l2.scan_recent_keys(limit=limit)
            if not keys:
                logger.debug("L2→L1 promotion: L2 is empty, nothing to promote")
                return

            promoted = 0
            skipped_expired = 0

            for key in keys:
                try:
                    raw = await self._l2.get(key)
                    if raw is None:
                        continue

                    entry = self._serializer.deserialize(raw)
                    if entry.is_expired:
                        skipped_expired += 1
                        continue

                    remaining_ttl = max(1, int(entry.ttl_seconds - entry.age_seconds))
                    await self._l1.set(key, raw, remaining_ttl)
                    promoted += 1

                except Exception:
                    continue

            logger.info(
                "L2→L1 promotion complete: promoted=%d, expired_skipped=%d, scanned=%d",
                promoted,
                skipped_expired,
                len(keys),
            )

        except Exception:
            logger.warning(
                "L2→L1 promotion failed — L1 starts cold",
                exc_info=True,
            )

    async def _seed_semantic_from_l2(self) -> None:
        """Seed semantic Qdrant from L2 entries after restart."""
        if self._semantic_strategy is None or self._l2 is None:
            return

        try:
            keys = await self._l2.scan_recent_keys(limit=100)
            if not keys:
                return

            seeded = 0
            skipped = 0
            for key in keys:
                try:
                    raw = await self._l2.get(key)
                    if raw is None:
                        continue
                    entry = self._serializer.deserialize(raw)
                    if entry.is_expired or entry.query_text is None:
                        skipped += 1
                        continue
                    await self._semantic_strategy.index_entry(
                        query=entry.query_text,
                        cache_key=entry.cache_key,
                        model_name=entry.model_name,
                        temperature=entry.temperature,
                    )
                    seeded += 1
                except Exception as exc:
                    logger.debug(
                        "Semantic seeding: skipped key=%s | %s", key[:16], exc
                    )
                    continue

            logger.info(
                "Semantic Qdrant seeded from L2: seeded=%d, skipped=%d (no query_text)",
                seeded,
                skipped,
            )

        except Exception:
            logger.warning("Semantic seeding failed — semantic cache starts cold", exc_info=True)

    async def get(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt: str = "",
        user_id: str = "",
        scope_hash: str = "",
    ) -> CacheResult:
        """Look up a cached LLM response; never raises to the caller."""
        start = time.perf_counter()

        if not self._enabled:
            return CacheResult.miss()

        try:
            system_prompt_hash = hash_text(system_prompt) if system_prompt else ""

            key = self._exact_strategy.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )

            result = await self._try_get_from_l1(key, start)
            if result is not None:
                self._record_hit_metrics(result)
                return result

            if self._l2 is not None:
                result = await self._try_get_from_l2(key, start)
                if result is not None:
                    await self._promote_to_l1(key, result)
                    self._record_hit_metrics(result)
                    return result

            if self._semantic_strategy is not None:
                result = await self._try_semantic_lookup(
                    query, model_name, temperature, system_prompt_hash, start, user_id,
                    scope_hash=scope_hash,
                )
                if result is not None:
                    self._record_hit_metrics(result)
                    return result

            latency = self._elapsed_ms(start)
            self._metrics.record_miss(latency)
            logger.debug(
                "Cache miss: query='%s', key=%s, latency=%.2fms",
                query[:60],
                key[:16] + "...",
                latency,
            )
            return CacheResult.miss(latency_ms=latency)

        except Exception:
            latency = self._elapsed_ms(start)
            logger.exception("Cache get failed, returning miss")
            self._metrics.record_miss(latency)
            return CacheResult.miss(latency_ms=latency)

    async def _try_get_from_l1(
        self, key: str, start: float
    ) -> Optional[CacheResult]:
        try:
            raw = await self._l1.get(key)
            if raw is None:
                return None

            entry = self._serializer.deserialize(raw)
            if entry.is_expired:
                await self._l1.delete(key)
                return None

            entry.record_hit()
            latency = self._elapsed_ms(start)

            return CacheResult.from_hit(
                response=entry.response,
                layer=CacheLayer.L1_MEMORY,
                strategy=CacheStrategy.EXACT,
                latency_ms=latency,
                cache_key=key,
                sources=entry.sources,
                cache_age_seconds=entry.age_seconds,
                confidence_value=entry.confidence_value,
            )
        except CacheSerializationError:
            logger.warning("L1 corrupt entry removed: key=%s", key[:16] + "...")
            self._metrics.record_error("serialization")
            await self._l1.delete(key)
            return None
        except Exception:
            logger.exception("L1 get failed: key=%s", key[:16] + "...")
            self._metrics.record_error("l1")
            return None

    async def _try_get_from_l2(
        self, key: str, start: float
    ) -> Optional[CacheResult]:
        if self._l2 is None:
            return None

        try:
            raw = await self._l2.get(key)
            if raw is None:
                return None

            entry = self._serializer.deserialize(raw)
            if entry.is_expired:
                try:
                    await self._l2.delete(key)
                except Exception:
                    pass
                return None

            entry.record_hit()
            latency = self._elapsed_ms(start)

            return CacheResult.from_hit(
                response=entry.response,
                layer=CacheLayer.L2_REDIS,
                strategy=CacheStrategy.EXACT,
                latency_ms=latency,
                cache_key=key,
                sources=entry.sources,
                cache_age_seconds=entry.age_seconds,
                confidence_value=entry.confidence_value,
            )
        except CacheSerializationError:
            logger.warning("L2 corrupt entry removed: key=%s", key[:16] + "...")
            self._metrics.record_error("serialization")
            try:
                await self._l2.delete(key)
            except Exception:
                pass
            return None
        except Exception:
            logger.exception("L2 get failed: key=%s", key[:16] + "...")
            self._metrics.record_error("l2")
            return None

    async def _try_semantic_lookup(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str,
        start: float,
        user_id: str = "",
        scope_hash: str = "",
    ) -> Optional[CacheResult]:
        """Attempt semantic similarity match via Qdrant; never raises."""
        try:
            match = await self._semantic_strategy.find_similar(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )

            if match is None:
                return None

            semantic_key = match.cache_key

            raw = await self._l1.get(semantic_key)
            layer = CacheLayer.L1_MEMORY

            if raw is None and self._l2 is not None:
                raw = await self._l2.get(semantic_key)
                layer = CacheLayer.L2_REDIS

            if raw is None:
                logger.debug(
                    "Semantic match found but entry missing from L1/L2: key=%s",
                    semantic_key[:16] + "...",
                )
                return None

            entry = self._serializer.deserialize(raw)

            if entry.is_expired:
                logger.debug(
                    "Semantic match found but entry expired: key=%s",
                    semantic_key[:16] + "...",
                )
                return None

            entry.record_hit()
            latency = self._elapsed_ms(start)

            tier_map = {
                "direct": SemanticTier.DIRECT,
                "high": SemanticTier.HIGH,
            }
            semantic_tier = tier_map.get(match.tier, SemanticTier.HIGH)

            if layer == CacheLayer.L2_REDIS:
                await self._promote_to_l1(semantic_key, CacheResult.from_hit(
                    response=entry.response,
                    layer=layer,
                    strategy=CacheStrategy.SEMANTIC,
                    latency_ms=latency,
                    cache_key=semantic_key,
                ))

            return CacheResult.from_hit(
                response=entry.response,
                layer=layer,
                strategy=CacheStrategy.SEMANTIC,
                latency_ms=latency,
                cache_key=semantic_key,
                sources=entry.sources,
                cache_age_seconds=entry.age_seconds,
                similarity_score=match.similarity_score,
                semantic_tier=semantic_tier,
                confidence_value=entry.confidence_value,
            )

        except CacheSerializationError:
            self._metrics.record_error("serialization")
            return None
        except Exception:
            logger.exception("Semantic lookup failed: query='%s'", query[:60])
            return None

    async def _promote_to_l1(self, key: str, result: CacheResult) -> None:
        if result.response is None:
            return
        try:
            entry = self._build_entry_from_result(key, result)
            raw = self._serializer.serialize(entry)
            remaining_ttl = max(1, int(entry.ttl_seconds - entry.age_seconds))
            await self._l1.set(key, raw, remaining_ttl)
            logger.debug(
                "L2→L1 promotion: key=%s, remaining_ttl=%ds",
                key[:16] + "...",
                remaining_ttl,
            )
        except Exception:
            logger.exception("L1 promotion failed: key=%s", key[:16] + "...")

    async def set(
        self,
        query: str,
        model_name: str,
        temperature: float,
        response: LLMResponse,
        system_prompt: str = "",
        ttl_seconds: Optional[int] = None,
        sources: list = [],
        confidence_value: float = 0.0,
        user_id: str = "",
        scope_hash: str = "",
    ) -> bool:
        """Cache an LLM response; returns True if written to at least L1."""
        start = time.perf_counter()

        if not self._enabled:
            return False

        try:
            passed, rejection_reason = self._quality_gate.check(response)
            if not passed:
                self._metrics.record_quality_rejection()
                logger.debug(
                    "Quality gate rejected: query='%s', reason=%s",
                    query[:60],
                    rejection_reason,
                )
                return False

            system_prompt_hash = hash_text(system_prompt) if system_prompt else ""

            key = self._exact_strategy.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )

            if ttl_seconds is not None:
                ttl = ttl_seconds
            else:
                ttl, query_type = self._ttl_classifier.get_ttl_with_type(query)
                logger.debug(
                    "TTL classified: query='%s', type=%s, ttl=%ds",
                    query[:60],
                    query_type.value,
                    ttl,
                )

            now = datetime.now(timezone.utc)

            entry = CacheEntry(
                response=response,
                cache_key=key,
                query_hash=self._exact_strategy.get_query_hash(query),
                query_text=self._normalizer.normalize(query),
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                ttl_seconds=ttl,
                provider=response.provider,
                model_name=response.model,
                temperature=temperature,
                token_cost_estimate=self._estimate_cost(response),
                sources=sources,
                confidence_value=confidence_value,
            )

            raw = self._serializer.serialize(entry)

            await self._l1.set(key, raw, ttl)

            if self._l2 is not None:
                try:
                    await self._l2.set(key, raw, ttl)
                except Exception:
                    logger.exception("L2 write failed: key=%s", key[:16] + "...")
                    self._metrics.record_error("l2")

            if self._semantic_strategy is not None:
                try:
                    await self._semantic_strategy.index_entry(
                        query=query,
                        cache_key=key,
                        model_name=model_name,
                        temperature=temperature,
                        system_prompt_hash=system_prompt_hash,
                        user_id=user_id,
                        scope_hash=scope_hash,
                    )
                except Exception:
                    logger.exception("Semantic index failed: key=%s", key[:16] + "...")

            latency = self._elapsed_ms(start)
            self._metrics.record_write(latency)
            logger.debug(
                "Cache write: query='%s', key=%s, ttl=%ds, "
                "backends=[%s], semantic=%s, latency=%.2fms",
                query[:60],
                key[:16] + "...",
                ttl,
                ", ".join(self._get_active_backend_names()),
                self._semantic_strategy is not None,
                latency,
            )
            return True

        except Exception:
            logger.exception("Cache set failed: query='%s'", query[:60])
            return False

    async def get_or_wait(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt: str = "",
        timeout: float = 10.0,
        user_id: str = "",
        scope_hash: str = "",
    ) -> CacheResult:
        """Cache lookup with request coalescing; waits on in-flight duplicate LLM calls."""
        if not self._enabled:
            return CacheResult.miss()

        result = await self.get(
            query, model_name, temperature, system_prompt,
            user_id=user_id, scope_hash=scope_hash,
        )
        if result.hit:
            return result

        system_prompt_hash = hash_text(system_prompt) if system_prompt else ""

        try:
            key = self._exact_strategy.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )
        except Exception:
            return result

        async with self._in_flight_lock:
            if key in self._in_flight:
                event = self._in_flight[key]
            else:
                event = asyncio.Event()
                self._in_flight[key] = event
                return result

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Coalescing timeout after %.1fs: key=%s",
                timeout,
                key[:16] + "...",
            )
            return CacheResult.miss()

        return await self.get(
            query, model_name, temperature, system_prompt,
            user_id=user_id, scope_hash=scope_hash,
        )

    async def resolve_in_flight(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt: str = "",
        user_id: str = "",
        scope_hash: str = "",
    ) -> None:
        """Signal that an in-flight LLM call has completed, unblocking get_or_wait() waiters."""
        system_prompt_hash = hash_text(system_prompt) if system_prompt else ""

        try:
            key = self._exact_strategy.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )
        except Exception:
            return

        async with self._in_flight_lock:
            event = self._in_flight.pop(key, None)

        if event is not None:
            event.set()
            logger.debug("In-flight resolved: key=%s", key[:16] + "...")

    def _passes_quality_gate(self, response: LLMResponse) -> bool:
        """Delegate to extracted QualityGate. Kept for backward compatibility."""
        return self._quality_gate.passes(response)

    def get_metrics(self) -> dict:
        """Return a snapshot of cache performance metrics."""
        return self._metrics.summary()

    async def get_full_stats(self) -> dict:
        """Return comprehensive stats including backend, semantic, TTL, and quality gate details."""
        stats = {
            "enabled": self._enabled,
            "initialized": self._initialized,
            "strategy": "hybrid" if self._semantic_strategy else "exact",
            "metrics": self._metrics.summary(),
            "backends": {},
        }

        try:
            stats["backends"]["l1"] = await self._l1.stats()
        except Exception:
            stats["backends"]["l1"] = {"error": "stats unavailable"}

        if self._l2 is not None:
            try:
                stats["backends"]["l2"] = await self._l2.stats()
            except Exception:
                stats["backends"]["l2"] = {"error": "stats unavailable"}

        if self._semantic_strategy is not None:
            try:
                stats["semantic"] = await self._semantic_strategy.get_collection_stats()
            except Exception:
                stats["semantic"] = {"error": "stats unavailable"}

        stats["ttl_classifier"] = self._ttl_classifier.ttl_map
        stats["quality_gate"] = self._quality_gate.thresholds
        stats["in_flight_count"] = len(self._in_flight)
        return stats

    async def invalidate(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt: str = "",
        user_id: str = "",
        scope_hash: str = "",
    ) -> bool:
        """Delete a specific cached entry from all backends; returns True if found in at least one."""
        system_prompt_hash = hash_text(system_prompt) if system_prompt else ""

        try:
            key = self._exact_strategy.make_key(
                query=query,
                model_name=model_name,
                temperature=temperature,
                system_prompt_hash=system_prompt_hash,
                user_id=user_id,
                scope_hash=scope_hash,
            )
        except Exception:
            logger.exception("Invalidation failed — key generation error")
            return False

        deleted = False

        try:
            if await self._l1.delete(key):
                deleted = True
        except Exception:
            logger.exception("L1 delete failed: key=%s", key[:16] + "...")

        if self._l2 is not None:
            try:
                if await self._l2.delete(key):
                    deleted = True
            except Exception:
                logger.exception("L2 delete failed: key=%s", key[:16] + "...")

        if deleted:
            logger.info("Cache entry invalidated: key=%s", key[:16] + "...")

        return deleted

    async def invalidate_user(self, user_id: str) -> bool:
        """Delete all semantic cache entries for the given user; returns True on success."""
        if not self._semantic_strategy or not user_id:
            return False
        try:
            await asyncio.to_thread(self._semantic_strategy.delete_user_entries, user_id)
            logger.info("User cache invalidated: user_id=%s", user_id)
            return True
        except Exception:
            logger.exception("invalidate_user failed: user_id=%s", user_id)
            return False

    async def clear_all(self) -> dict:
        """Clear all entries from all backends; returns per-backend removal counts (-1 on error)."""
        result = {}

        try:
            result["l1"] = await self._l1.clear()
        except Exception:
            logger.exception("L1 clear failed")
            result["l1"] = -1

        if self._l2 is not None:
            try:
                result["l2"] = await self._l2.clear()
            except Exception:
                logger.exception("L2 clear failed")
                result["l2"] = -1

        self._metrics = CacheMetrics()
        logger.info("Cache cleared: %s", result)
        return result

    async def close(self) -> None:
        """Graceful shutdown — close all backends and strategies."""
        try:
            await self._l1.close()
        except Exception:
            logger.exception("L1 close failed")

        if self._l2 is not None:
            try:
                await self._l2.close()
            except Exception:
                logger.exception("L2 close failed")

        if self._semantic_strategy is not None:
            try:
                await self._semantic_strategy.close()
            except Exception:
                logger.exception("Semantic strategy close failed")

        async with self._in_flight_lock:
            for event in self._in_flight.values():
                event.set()
            self._in_flight.clear()

        self._initialized = False
        logger.info(
            "CacheManager closed — final metrics: %s",
            self._metrics.summary(),
        )

    def _estimate_cost(self, response: LLMResponse) -> float:
        """Estimate token cost in USD; returns 0.0 for unknown providers."""
        tokens = response.tokens_used
        if response.provider == "openai":
            return tokens * self._settings.COST_PER_TOKEN_OPENAI
        elif response.provider == "gemini":
            return tokens * self._settings.COST_PER_TOKEN_GEMINI
        elif response.provider == "groq":
            return tokens * self._settings.COST_PER_TOKEN_GROQ
        return 0.0

    def _record_hit_metrics(self, result: CacheResult) -> None:
        """Compute token/cost savings from a cache hit and update metrics counters."""
        tokens_saved = 0
        cost_saved = 0.0

        if result.response is not None:
            tokens_saved = result.response.tokens_used
            if result.response.provider == "openai":
                cost_saved = tokens_saved * self._settings.COST_PER_TOKEN_OPENAI
            elif result.response.provider == "gemini":
                cost_saved = tokens_saved * self._settings.COST_PER_TOKEN_GEMINI
            elif result.response.provider == "groq":
                cost_saved = tokens_saved * self._settings.COST_PER_TOKEN_GROQ

        self._metrics.record_hit(
            layer=result.layer.value,
            strategy=result.strategy.value,
            tokens_saved=tokens_saved,
            cost_saved=cost_saved,
            latency_ms=result.lookup_latency_ms,
        )

    def _build_entry_from_result(self, key: str, result: CacheResult) -> CacheEntry:
        """Construct a CacheEntry from a CacheResult for L2 → L1 promotion."""
        ttl = self._settings.cache_ttl_seconds
        now = datetime.now(timezone.utc)

        return CacheEntry(
            response=result.response,
            cache_key=key,
            query_hash=key,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            ttl_seconds=ttl,
            provider=result.response.provider,
            model_name=result.response.model,
            temperature=0.0,
            token_cost_estimate=self._estimate_cost(result.response),
            sources=result.sources,
            confidence_value=result.confidence_value,
        )

    def _get_active_backend_names(self) -> list[str]:
        """Return names of all currently active backends."""
        names = [self._l1.name]
        if self._l2 is not None:
            names.append(self._l2.name)
        return names

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Compute elapsed milliseconds since a perf_counter start time."""
        return (time.perf_counter() - start) * 1000
