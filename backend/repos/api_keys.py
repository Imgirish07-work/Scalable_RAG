"""API keys table and CRUD."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from backend.repos.base import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.user_id"), index=True, nullable=False
    )
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


async def create_api_key(
    session: AsyncSession,
    user_id: str,
    key_hash: str,
    key_prefix: str,
    name: Optional[str] = None,
) -> ApiKey:
    api_key = ApiKey(
        key_id=str(uuid.uuid4()),
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
    )
    session.add(api_key)
    await session.flush()
    return api_key


async def lookup_active_key(session: AsyncSession, key_hash: str) -> Optional[ApiKey]:
    """Return the active (non-revoked) key matching the hash, or None."""
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.key_hash == key_hash,
            ApiKey.revoked_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def touch_last_used(session: AsyncSession, key_id: str) -> None:
    """Best-effort update of `last_used_at` to now."""
    key = await session.get(ApiKey, key_id)
    if key is not None:
        key.last_used_at = datetime.now(timezone.utc)
