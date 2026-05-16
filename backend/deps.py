"""FastAPI dependencies: pipeline singleton, DB session, Principal."""

from typing import AsyncIterator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.api_key import hash_key
from backend.auth.principal import Principal
from backend.repos.api_keys import lookup_active_key, touch_last_used
from backend.repos.base import get_session_factory
from backend.repos.users import get_user_by_id
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Request-scoped session; commit on success, rollback on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_pipeline(request: Request) -> RAGPipeline:
    """Return the lifespan-created RAGPipeline; 503 if not ready."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return pipeline


async def get_principal(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> Principal:
    """Resolve `Authorization: Bearer <api_key>` into a Principal."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
        )

    api_key = await lookup_active_key(db, hash_key(token))
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    user = await get_user_by_id(db, api_key.user_id)
    if user is None:
        logger.error(
            "Orphan API key | key_id=%s references missing user_id=%s",
            api_key.key_id, api_key.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Key references missing user",
        )

    try:
        await touch_last_used(db, api_key.key_id)
    except Exception:
        logger.warning("touch_last_used failed for key_id=%s", api_key.key_id)

    return Principal(
        user_id=user.user_id,
        email=user.email,
        key_id=api_key.key_id,
        plan=user.plan,
    )
