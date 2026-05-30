"""Data-access layer for the `documents` table.

The repository is the only place that writes SQLAlchemy queries against this
table. Controllers and services depend on it; nobody else issues SQL.

The session is injected per request (via `get_db_session`), so one instance of
this class lives for exactly one HTTP request and shares its transaction.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.document import Document
from backend.repositories.database import get_db_session
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """Encapsulates every read/write against the `documents` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, document: Document) -> Document:
        """Insert a new document row; the dependency commits on handler return."""
        self._session.add(document)
        await self._session.flush()
        logger.info(
            "Document created | doc_id=%s | user_id=%s | collection=%s | size_bytes=%d",
            document.id, document.user_id, document.collection, document.size_bytes,
        )
        return document

    async def find_by_id(self, doc_id: str) -> Optional[Document]:
        """Look up by primary key. Returns soft-deleted rows too."""
        stmt = select(Document).where(Document.id == doc_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_active_by_content_hash(
        self,
        user_id: str,
        content_hash: str,
    ) -> Optional[Document]:
        """Dedup probe: the user's active document for this content hash, if any."""
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .where(Document.content_hash == content_hash)
            .where(Document.deleted_at.is_(None))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        doc_id: str,
        status: str,
        chunks_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Atomic status transition; updates `updated_at` in the same statement."""
        values: dict = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if chunks_count is not None:
            values["chunks_count"] = chunks_count
        if error_message is not None:
            values["error_message"] = error_message
        stmt = update(Document).where(Document.id == doc_id).values(**values)
        await self._session.execute(stmt)
        logger.info(
            "Document status updated | doc_id=%s | status=%s | chunks_count=%s",
            doc_id, status, chunks_count if chunks_count is not None else "<unchanged>",
        )

    async def soft_delete(self, doc_id: str) -> None:
        """Mark the document deleted without removing the row."""
        now = datetime.now(timezone.utc)
        stmt = (
            update(Document)
            .where(Document.id == doc_id)
            .where(Document.deleted_at.is_(None))
            .values(deleted_at=now, updated_at=now)
        )
        await self._session.execute(stmt)
        logger.info("Document soft-deleted | doc_id=%s", doc_id)

    async def hard_delete(self, doc_id: str) -> None:
        """Permanently remove the row. Used on terminal ingestion failure
        and on the rollback path of a duplicate-detected upload."""
        stmt = delete(Document).where(Document.id == doc_id)
        await self._session.execute(stmt)
        logger.info("Document hard-deleted | doc_id=%s", doc_id)

    async def set_content_hash(self, doc_id: str, content_hash: str) -> None:
        """Persist the SHA-256 computed during finalize's stream-download."""
        stmt = (
            update(Document)
            .where(Document.id == doc_id)
            .values(
                content_hash=content_hash,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

    async def list_by_user(
        self,
        user_id: str,
        collection: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Document]:
        """Tenant-scoped list with optional filters, newest first."""
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .where(Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if collection is not None:
            stmt = stmt.where(Document.collection == collection)
        if status is not None:
            stmt = stmt.where(Document.status == status)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_collections_for_user(
        self, user_id: str,
    ) -> list[tuple[str, int, datetime]]:
        """Distinct logical collections for a user with doc counts + last update."""
        stmt = (
            select(
                Document.collection,
                func.count().label("doc_count"),
                func.max(Document.updated_at).label("last_updated"),
            )
            .where(Document.user_id == user_id)
            .where(Document.deleted_at.is_(None))
            .group_by(Document.collection)
            .order_by(func.max(Document.updated_at).desc())
        )
        result = await self._session.execute(stmt)
        return [(row[0], row[1], row[2]) for row in result.all()]


def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    """FastAPI dependency: per-request repository sharing the request's session."""
    return DocumentRepository(session)
