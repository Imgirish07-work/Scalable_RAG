"""
L2 Redis cache backend — shared, persistent, TTL-native production store.

Design:
    Implements BaseCacheBackend over redis.asyncio (async Redis client).
    Uses connection pooling for efficiency, key prefixing for namespace
    isolation, SETEX for atomic TTL-aware writes, and SCAN (not KEYS)
    for non-blocking key enumeration. A CircuitBreaker wraps every call
    to prevent cascading failures when Redis becomes unhealthy.

    Performance:
        get():    ~0.5-2ms (network round-trip)
        set():    ~0.5-2ms (network round-trip)
        exists(): ~0.5-1ms (lightweight check)

Chain of Responsibility:
    Instantiated by CacheManager._initialize_l2() via from_config().
    RedisConnectionConfig is provided by RedisConfigFactory. All calls
    pass through the internal CircuitBreaker before reaching the network.

Dependencies:
    redis>=5.0.0 (redis.asyncio)
"""

import redis.asyncio as aioredis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    RedisError,
)
from typing import Optional

from utils.logger import get_logger
from cache.backend.base_backend import BaseCacheBackend
from cache.backend.redis_config import RedisConnectionConfig
from cache.backend.circuit_breaker import CircuitBreaker
from cache.exceptions.cache_exceptions import (
    CacheConnectionError,
    CacheBackendError,
)

logger = get_logger(__name__)


class RedisCacheBackend(BaseCacheBackend):
    """L2 Redis cache backend with circuit breaker and connection pooling.

    Attributes:
        _url: Redis connection URL.
        _prefix: Key prefix for namespace isolation.
        _socket_timeout: Timeout per Redis operation in seconds.
        _max_connections: Connection pool size limit.
        _retry_on_timeout: Whether to retry on timeout.
        _environment: Environment name for logging (local/cloud/test).
        _pool: redis.asyncio connection pool (created in initialize()).
        _client: redis.asyncio.Redis client (created in initialize()).
        _breaker: Circuit breaker instance.
        _initialized: Whether initialize() has been called.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "llmcache:",
        socket_timeout: float = 2.0,
        max_connections: int = 20,
        retry_on_timeout: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_seconds: float = 60.0,
        environment: str = "local",
    ) -> None:
        """Sync constructor — stores config, no connections made.

        Args:
            url: Redis connection URL (redis:// or rediss:// for TLS).
            prefix: Key prefix for namespace isolation in shared Redis.
            socket_timeout: Per-operation timeout in seconds.
            max_connections: Max connections in the async pool.
            retry_on_timeout: Whether redis-py retries on timeout.
            circuit_breaker_threshold: Failures before tripping breaker.
            circuit_breaker_reset_seconds: Seconds before retry after trip.
            environment: Environment name for logging.
        """
        self._url = url
        self._prefix = prefix
        self._socket_timeout = socket_timeout
        self._max_connections = max_connections
        self._retry_on_timeout = retry_on_timeout
        self._environment = environment

        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._initialized: bool = False

        self._breaker = CircuitBreaker(
            name="redis_l2",
            failure_threshold=circuit_breaker_threshold,
            reset_seconds=circuit_breaker_reset_seconds,
        )

        logger.info(
            "RedisCacheBackend created: env=%s, url=%s, prefix='%s', "
            "timeout=%.1fs, max_conn=%d",
            self._environment,
            self._redacted_url,
            self._prefix,
            self._socket_timeout,
            self._max_connections,
        )

    @classmethod
    def from_config(cls, config: RedisConnectionConfig) -> "RedisCacheBackend":
        """Factory method — create backend from RedisConnectionConfig.

        This is the preferred way to construct a RedisCacheBackend.
        The config object comes from RedisConfigFactory.create(settings).

        Args:
            config: RedisConnectionConfig dataclass instance.

        Returns:
            Configured RedisCacheBackend (not yet initialized — call
            await backend.initialize() after construction).
        """
        return cls(
            url=config.url,
            prefix=config.prefix,
            socket_timeout=config.socket_timeout,
            max_connections=config.max_connections,
            retry_on_timeout=config.retry_on_timeout,
            circuit_breaker_threshold=config.circuit_breaker_threshold,
            circuit_breaker_reset_seconds=config.circuit_breaker_reset_seconds,
            environment=config.environment,
        )

    @property
    def name(self) -> str:
        """Backend identifier used in logs and metrics."""
        return "l2_redis"

    @property
    def _redacted_url(self) -> str:
        """URL with password redacted for safe logging."""
        if "@" in self._url:
            protocol_end = self._url.index("://") + 3
            at_pos = self._url.rindex("@")
            return self._url[:protocol_end] + "***@" + self._url[at_pos + 1:]
        return self._url

    def _make_key(self, key: str) -> str:
        """Prepend namespace prefix to a key."""
        return f"{self._prefix}{key}"

    async def initialize(self) -> None:
        """Create connection pool and verify connectivity.

        Must be called once before any get/set operations.
        Safe to call multiple times — subsequent calls are no-ops.

        Raises:
            CacheConnectionError: If Redis is unreachable on first connect.
        """
        if self._initialized:
            return

        try:
            # Create the async connection pool
            self._pool = aioredis.ConnectionPool.from_url(
                self._url,
                max_connections=self._max_connections,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_timeout,
                retry_on_timeout=self._retry_on_timeout,
                decode_responses=True,
            )

            # Attach a client to the pool
            self._client = aioredis.Redis(connection_pool=self._pool)

            # Verify the server is reachable before marking initialized
            await self._client.ping()
            self._initialized = True

            info = await self._client.info()
            version = info.get("redis_version", "unknown")

            logger.info(
                "RedisCacheBackend initialized: env=%s, version=%s, url=%s",
                self._environment,
                version,
                self._redacted_url,
            )

        except (RedisConnectionError, RedisTimeoutError, OSError) as e:
            logger.error(
                "Redis connection failed: env=%s, url=%s, error=%s",
                self._environment,
                self._redacted_url,
                e,
            )
            self._initialized = False
            raise CacheConnectionError(
                backend="l2_redis",
                message=f"Cannot connect to Redis at {self._redacted_url}: {e}",
            ) from e

        except (RedisError, Exception) as e:
            logger.exception(
                "Unexpected Redis error during initialization: env=%s, url=%s",
                self._environment,
                self._redacted_url,
            )
            self._initialized = False
            raise CacheConnectionError(
                backend="l2_redis",
                message=f"Redis error during initialization at {self._redacted_url}: {e}",
            ) from e

    def _ensure_initialized(self) -> None:
        """Guard all public methods — raises if initialize() was not called."""
        if not self._initialized or self._client is None:
            raise CacheConnectionError(
                backend="l2_redis",
                message="RedisCacheBackend not initialized. Call initialize() first.",
            )

    async def get(self, key: str) -> Optional[str]:
        """Retrieve a cached value from Redis.

        TTL is managed by Redis natively — expired keys return None
        automatically. No client-side TTL checking needed.

        Args:
            key: Cache key (SHA-256 hex digest).

        Returns:
            Serialized cache entry (JSON string), or None on miss.

        Raises:
            CacheBackendError: On Redis communication failure.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return None

        try:
            result = await self._client.get(self._make_key(key))
            self._breaker.record_success()

            if result is not None:
                logger.debug("L2 Redis hit: key=%s", key[:16] + "...")
            return result

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._breaker.record_failure()
            logger.warning(
                "L2 Redis get failed (connection): key=%s, error=%s",
                key[:16] + "...",
                e,
            )
            raise CacheBackendError(
                backend="l2_redis", message=f"GET failed: {e}"
            ) from e

        except RedisError as e:
            self._breaker.record_failure()
            logger.exception("L2 Redis get failed: key=%s", key[:16] + "...")
            raise CacheBackendError(
                backend="l2_redis", message=f"GET failed: {e}"
            ) from e

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store a value in Redis with native TTL.

        Uses SETEX which atomically sets the value and TTL in one
        round-trip. Redis handles expiry automatically.

        Args:
            key: Cache key.
            value: Serialized cache entry (JSON string).
            ttl_seconds: Time-to-live in seconds.

        Raises:
            CacheBackendError: On Redis communication failure.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return

        # SETEX requires ttl >= 1; skip rather than raise so callers can
        # disable caching by passing ttl_seconds=0 without error handling.
        if ttl_seconds <= 0:
            logger.debug("Skipping set: ttl_seconds=%d <= 0 for key=%s", ttl_seconds, key[:16] + "...")
            return

        try:
            # SETEX atomically writes value + TTL in a single command
            await self._client.setex(
                name=self._make_key(key),
                time=ttl_seconds,
                value=value,
            )

            self._breaker.record_success()

            logger.debug(
                "L2 Redis set: key=%s, ttl=%ds, size=%d bytes",
                key[:16] + "...",
                ttl_seconds,
                len(value),
            )

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._breaker.record_failure()
            logger.warning(
                "L2 Redis set failed (connection): key=%s, error=%s",
                key[:16] + "...",
                e,
            )
            raise CacheBackendError(
                backend="l2_redis", message=f"SETEX failed: {e}"
            ) from e

        except RedisError as e:
            self._breaker.record_failure()
            logger.exception("L2 Redis set failed: key=%s", key[:16] + "...")
            raise CacheBackendError(
                backend="l2_redis", message=f"SETEX failed: {e}"
            ) from e

    async def delete(self, key: str) -> bool:
        """Delete a single entry from Redis.

        Args:
            key: Cache key to delete.

        Returns:
            True if the key existed and was deleted.

        Raises:
            CacheBackendError: On Redis communication failure.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return False

        try:
            removed = await self._client.delete(self._make_key(key))
            self._breaker.record_success()

            deleted = removed > 0
            if deleted:
                logger.debug("L2 Redis deleted: key=%s", key[:16] + "...")
            return deleted

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"DELETE failed: {e}"
            ) from e

        except RedisError as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"DELETE failed: {e}"
            ) from e

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis (not expired).

        Args:
            key: Cache key to check.

        Returns:
            True if key exists.

        Raises:
            CacheBackendError: On Redis communication failure.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return False

        try:
            result = await self._client.exists(self._make_key(key))
            self._breaker.record_success()
            return result > 0

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"EXISTS failed: {e}"
            ) from e

        except RedisError as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"EXISTS failed: {e}"
            ) from e

    async def clear(self) -> int:
        """Remove all entries with the configured prefix.

        Uses SCAN + DELETE to avoid blocking Redis with KEYS *.
        KEYS * blocks until all keys are returned, which can cause
        timeouts in production. SCAN iterates in batches without
        blocking the server.

        Returns:
            Number of entries removed.

        Raises:
            CacheBackendError: On Redis communication failure.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return 0

        try:
            pattern = f"{self._prefix}*"
            total_deleted = 0
            cursor = 0

            while True:
                cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=100)

                if keys:
                    removed = await self._client.delete(*keys)
                    total_deleted += removed

                if cursor == 0:
                    break

            self._breaker.record_success()
            logger.info("L2 Redis cleared: removed=%d entries", total_deleted)
            return total_deleted

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"CLEAR failed: {e}"
            ) from e

        except RedisError as e:
            self._breaker.record_failure()
            raise CacheBackendError(
                backend="l2_redis", message=f"CLEAR failed: {e}"
            ) from e

    async def size(self) -> int:
        """Count entries with the configured prefix.

        Returns:
            Approximate number of entries.
        """
        self._ensure_initialized()

        # Short-circuit if the backend is known to be unhealthy
        if not self._breaker.allow_request():
            return 0

        try:
            pattern = f"{self._prefix}*"
            count = 0
            cursor = 0

            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor, match=pattern, count=100
                )
                count += len(keys)
                if cursor == 0:
                    break

            self._breaker.record_success()
            return count
        except RedisError as e:
            self._breaker.record_failure()
            logger.warning("L2 Redis size failed: %s", e)
            return -1  # Return -1 to indicate size is unavailable due to error

    async def stats(self) -> dict:
        """Return Redis backend statistics.

        Returns:
            Dict with connection details, memory usage, and circuit breaker state.
        """
        result = {
            "name": self.name,
            "backend_type": "redis",
            "environment": self._environment,
            "url": self._redacted_url,
            "prefix": self._prefix,
            "initialized": self._initialized,
            "circuit_breaker": self._breaker.stats(),
        }

        if not self._initialized or self._client is None:
            return result

        if not self._breaker.allow_request():
            result["status"] = "circuit_open"
            return result

        try:
            info = await self._client.info(section="memory")
            result["used_memory_human"] = info.get("used_memory_human", "unknown")
            result["used_memory_bytes"] = info.get("used_memory", 0)

            pool_info = await self._client.info(section="clients")
            result["connected_clients"] = pool_info.get("connected_clients", 0)

            result["cached_key_count"] = await self.size()
            result["status"] = "healthy"

            self._breaker.record_success()

        except RedisError as e:
            self._breaker.record_failure()
            result["status"] = f"error: {e}"

        return result

    async def ping(self) -> bool:
        """Health check — verify Redis is responsive.

        Returns:
            True if Redis responds to PING, False otherwise.
        """
        if not self._initialized or self._client is None:
            return False

        try:
            result = await self._client.ping()
            self._breaker.record_success()
            return result

        except RedisError as e:
            self._breaker.record_failure()
            return False

    async def scan_recent_keys(self, limit: int = 100) -> list[str]:
        """Return up to `limit` bare keys (without prefix) from Redis.

        Used by CacheManager at startup to promote recent L2 entries
        into L1 so the first queries after a restart get L1 speed.

        Uses SCAN (non-blocking) instead of KEYS (blocks Redis).
        Returns bare keys so the caller can pass them directly to get().

        Args:
            limit: Maximum number of keys to return.

        Returns:
            List of bare cache keys (prefix stripped). Empty on error.
        """
        if not self._initialized or self._client is None:
            return []
        if not self._breaker.allow_request():
            return []

        try:
            keys: list[str] = []
            async for full_key in self._client.scan_iter(
                match=f"{self._prefix}*",
                count=limit,
            ):
                bare = full_key[len(self._prefix):]
                keys.append(bare)
                if len(keys) >= limit:
                    break
            self._breaker.record_success()
            logger.debug("scan_recent_keys: found %d keys (limit=%d)", len(keys), limit)
            return keys
        except Exception as e:
            self._breaker.record_failure()
            logger.warning("scan_recent_keys failed: %s", e)
            return []

    async def close(self) -> None:
        """Graceful shutdown — close connection pool."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.exception("Error closing Redis client")

        if self._pool is not None:
            try:
                await self._pool.disconnect()
            except Exception as e:
                logger.exception("Error closing Redis connection pool")

        self._client = None
        self._pool = None
        self._initialized = False

        logger.info(
            "RedisCacheBackend closed: env=%s, breaker_stats=%s",
            self._environment,
            self._breaker.stats(),
        )
