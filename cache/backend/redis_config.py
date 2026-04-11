"""
Redis connection configuration and environment-based factory.

Design:
    Separates WHAT to connect to (config) from HOW to use it (backend).
    RedisCacheBackend stays a single class — this module feeds it the
    correct connection parameters based on the REDIS_ENV setting.
    RedisConnectionConfig is a frozen dataclass — immutable after creation.
    RedisConfigFactory uses a builder pattern keyed by environment name.

    Environments:
        local  — redis://localhost:6379/0, no TLS, dev prefix
        cloud  — rediss://user:pass@host:port/0, TLS, prod prefix
        test   — redis://localhost:6379/1, separate DB, test prefix

Chain of Responsibility:
    Called by CacheManager._initialize_l2(). The resulting config is
    passed to RedisCacheBackend.from_config() to create the L2 backend.
    All connection details come from Settings (environment variables).

Dependencies:
    dataclasses (stdlib only — no third-party dependencies at this level)

Usage:
    from cache.backend.redis_config import RedisConfigFactory

    config = RedisConfigFactory.create(settings)
    backend = RedisCacheBackend.from_config(config)
    await backend.initialize()
"""

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RedisConnectionConfig:
    """Immutable Redis connection configuration.

    Passed to RedisCacheBackend.from_config() to construct
    a fully configured backend instance.

    Attributes:
        url: Full Redis connection URL (redis:// or rediss://).
        prefix: Key namespace prefix for isolation.
        max_connections: Connection pool size.
        socket_timeout: Per-operation timeout in seconds.
        retry_on_timeout: Whether redis-py retries on timeout.
        circuit_breaker_threshold: Failures before tripping breaker.
        circuit_breaker_reset_seconds: Seconds before retry after trip.
        environment: Environment name for logging (local/cloud/test).
    """

    url: str
    prefix: str = "llmcache:"
    max_connections: int = 20
    socket_timeout: float = 2.0
    retry_on_timeout: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: float = 60.0
    environment: str = "local"

    @property
    def is_tls(self) -> bool:
        """Whether this connection uses TLS (rediss:// scheme)."""
        return self.url.startswith("rediss://")

    @property
    def redacted_url(self) -> str:
        """URL with password masked for safe logging."""
        if "@" in self.url:
            protocol_end = self.url.index("://") + 3
            at_pos = self.url.rindex("@")
            return self.url[:protocol_end] + "***@" + self.url[at_pos + 1:]
        return self.url


class RedisConfigFactory:
    """Creates RedisConnectionConfig from application settings.

    Reads REDIS_ENV to determine which environment profile to use,
    then builds the appropriate config. All connection details come
    from .env — zero hardcoded URLs.

    Attributes:
        VALID_ENVIRONMENTS: Accepted values for REDIS_ENV.

    Supported environments:
        local — Development against local Redis
        cloud — Production against Redis Cloud / ElastiCache / Upstash
        test  — Isolated test database on local Redis
    """

    VALID_ENVIRONMENTS = {"local", "cloud", "test"}

    @classmethod
    def create(cls, settings) -> Optional[RedisConnectionConfig]:
        """Build Redis config from settings.

        Args:
            settings: Application settings with Redis fields.

        Returns:
            RedisConnectionConfig if Redis is configured, None if disabled.
        """
        redis_env = getattr(settings, "REDIS_ENV", "").strip().lower()

        if not redis_env or redis_env == "disabled":
            logger.info("Redis disabled — REDIS_ENV is empty or 'disabled'")
            return None

        if redis_env not in cls.VALID_ENVIRONMENTS:
            logger.warning(
                "Unknown REDIS_ENV='%s', falling back to 'local'. "
                "Valid options: %s",
                redis_env,
                cls.VALID_ENVIRONMENTS,
            )
            redis_env = "local"

        builder = {
            "local": cls._build_local,
            "cloud": cls._build_cloud,
            "test": cls._build_test,
        }

        config = builder[redis_env](settings)

        logger.info(
            "Redis config created: env=%s, url=%s, prefix='%s', "
            "tls=%s, pool=%d, timeout=%.1fs",
            config.environment,
            config.redacted_url,
            config.prefix,
            config.is_tls,
            config.max_connections,
            config.socket_timeout,
        )

        return config

    @classmethod
    def _build_local(cls, settings) -> RedisConnectionConfig:
        """Local development — localhost, no TLS, relaxed settings."""
        url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")

        if not url:
            url = "redis://localhost:6379/0"

        return RedisConnectionConfig(
            url=url,
            prefix="llmcache_dev:",
            max_connections=getattr(settings, "REDIS_MAX_CONNECTIONS", 10),
            socket_timeout=getattr(settings, "REDIS_SOCKET_TIMEOUT", 2.0),
            retry_on_timeout=getattr(settings, "REDIS_RETRY_ON_TIMEOUT", True),
            circuit_breaker_threshold=getattr(
                settings, "CACHE_CIRCUIT_BREAKER_THRESHOLD", 5
            ),
            circuit_breaker_reset_seconds=getattr(
                settings, "CACHE_CIRCUIT_BREAKER_RESET_SECONDS", 60.0
            ),
            environment="local",
        )

    @classmethod
    def _build_cloud(cls, settings) -> RedisConnectionConfig:
        """Cloud production — remote URL, TLS expected, strict settings."""
        url = getattr(settings, "REDIS_CLOUD_URL", "")
        if not url:
            logger.error(
                "REDIS_ENV=cloud but REDIS_CLOUD_URL is empty. "
                "Falling back to local."
            )
            return cls._build_local(settings)

        return RedisConnectionConfig(
            url=url,
            prefix="llmcache_prod:",
            max_connections=getattr(settings, "REDIS_MAX_CONNECTIONS", 20),
            socket_timeout=getattr(settings, "REDIS_SOCKET_TIMEOUT", 3.0),
            retry_on_timeout=True,
            circuit_breaker_threshold=getattr(
                settings, "CACHE_CIRCUIT_BREAKER_THRESHOLD", 5
            ),
            circuit_breaker_reset_seconds=getattr(
                settings, "CACHE_CIRCUIT_BREAKER_RESET_SECONDS", 60.0
            ),
            environment="cloud",
        )

    @classmethod
    def _build_test(cls, settings) -> RedisConnectionConfig:
        """Test environment — local Redis, separate DB, fast timeouts."""
        return RedisConnectionConfig(
            url="redis://localhost:6379/1",
            prefix="llmcache_test:",
            max_connections=5,
            socket_timeout=1.0,
            retry_on_timeout=False,
            circuit_breaker_threshold=3,
            circuit_breaker_reset_seconds=10.0,
            environment="test",
        )
