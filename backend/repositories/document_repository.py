"""Data-access layer for the `documents` table — sole writer of SQL on this table."""

from datetime import datetime, timedelta, timezone
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
        """Atomic status transition; manages `processing_started_at` lease."""
        now = datetime.now(timezone.utc)
        values: dict = {"status": status, "updated_at": now}
        if chunks_count is not None:
            values["chunks_count"] = chunks_count
        if error_message is not None:
            values["error_message"] = error_message
        elif status == "ready":
            values["error_message"] = None
        if status == "processing":
            values["processing_started_at"] = now
        elif status in ("ready", "failed"):
            values["processing_started_at"] = None
        stmt = update(Document).where(Document.id == doc_id).values(**values)
        await self._session.execute(stmt)
        logger.info(
            "Document status updated | doc_id=%s | status=%s | chunks_count=%s",
            doc_id, status, chunks_count if chunks_count is not None else "<unchanged>",
        )

    async def mark_ready_if_processing(
        self, doc_id: str, chunks_count: int,
    ) -> bool:
        """Conditional processing→ready. False if row was already moved by sweeper."""
        now = datetime.now(timezone.utc)
        stmt = (
            update(Document)
            .where(Document.id == doc_id)
            .where(Document.status == "processing")
            .values(
                status="ready",
                chunks_count=chunks_count,
                error_message=None,
                processing_started_at=None,
                updated_at=now,
            )
        )
        result = await self._session.execute(stmt)
        committed = (result.rowcount or 0) > 0
        if committed:
            logger.info(
                "Document marked ready | doc_id=%s | chunks=%d",
                doc_id, chunks_count,
            )
        else:
            logger.warning(
                "Ready transition rejected (no longer 'processing') | doc_id=%s",
                doc_id,
            )
        return committed

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

    async def list_stale_by_status(
        self, status: str, older_than_seconds: int, limit: int = 500,
    ) -> list[tuple[str, str]]:
        """Sweeper-only: rows of `status` older than the cutoff (by updated_at)."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        stmt = (
            select(Document.id, Document.s3_key)
            .where(Document.status == status)
            .where(Document.updated_at < cutoff)
            .order_by(Document.updated_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def mark_failed(
        self, doc_id: str, error_message: str,
    ) -> None:
        """Set `status='failed'` + `error_message` for DLQ inspection/retry."""
        now = datetime.now(timezone.utc)
        stmt = (
            update(Document)
            .where(Document.id == doc_id)
            .values(
                status="failed",
                error_message=error_message,
                processing_started_at=None,
                updated_at=now,
            )
        )
        await self._session.execute(stmt)
        logger.info("Document marked failed (DLQ) | doc_id=%s", doc_id)

    async def list_stuck_processing(
        self, lease_ttl_seconds: int, limit: int = 500,
    ) -> list[str]:
        """Sweeper-only: doc IDs stuck in 'processing' past the lease window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_ttl_seconds)
        stmt = (
            select(Document.id)
            .where(Document.status == "processing")
            .where(Document.processing_started_at < cutoff)
            .order_by(Document.processing_started_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [row[0] for row in result.all()]

    async def mark_stuck_failed(
        self, doc_id: str, lease_ttl_seconds: int, error_message: str,
    ) -> bool:
        """Atomic processing→failed for stuck rows; re-checks lease in WHERE."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_ttl_seconds)
        now = datetime.now(timezone.utc)
        stmt = (
            update(Document)
            .where(Document.id == doc_id)
            .where(Document.status == "processing")
            .where(Document.processing_started_at < cutoff)
            .values(
                status="failed",
                error_message=error_message,
                processing_started_at=None,
                updated_at=now,
            )
        )
        result = await self._session.execute(stmt)
        marked = (result.rowcount or 0) > 0
        if marked:
            logger.warning(
                "Stuck-processing row moved to DLQ | doc_id=%s", doc_id,
            )
        return marked

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
