"""Async SQLAlchemy engine, session factory, and transactional helpers.

The engine is created lazily on first use so the process can boot without a
database when the auth/persistence layer is not yet wired up.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.settings import backend_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Lazily build and return the module-level async engine."""
    global _engine
    if _engine is None:
        logger.info("Creating async DB engine")
        _engine = create_async_engine(
            backend_settings.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the cached session factory; creating it on first call."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session for scripts and services that need DB access."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close all pooled connections; safe to call even if the engine never started."""
    global _engine, _session_factory
    if _engine is not None:
        logger.info("Disposing async DB engine")
        await _engine.dispose()
        _engine = None
        _session_factory = None
