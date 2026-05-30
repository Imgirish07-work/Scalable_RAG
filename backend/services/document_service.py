"""Document orchestration service — the brain of the upload + ingest flow.

Owns the full lifecycle from upload-session creation through ingestion to
soft-delete. Touches three external systems (Postgres, MinIO, Qdrant) plus
the in-process RAG pipeline. Cross-system writes use compensating actions
on failure (saga); intra-system writes are atomic via SQLAlchemy sessions.
"""

import asyncio
import hashlib
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import magic
from fastapi import Depends, HTTPException, Request, status

from backend.models.document import Document
from backend.repositories.database import session_scope
from backend.repositories.document_repository import (
    DocumentRepository,
    get_document_repository,
)
from backend.schemas.document import (
    DocumentDetailView,
    DocumentListView,
    FinalizeAck,
    UploadSessionRequest,
    UploadSessionView,
)
from backend.services.event_bus import EventBus, get_event_bus
from backend.settings import (
    BackendSettings,
    StorageSettings,
    backend_settings,
    storage_settings,
)
from backend.storage.object_store import ObjectStore, get_object_store
from pipeline.rag_pipeline import RAGPipeline
from utils.helpers import generate_unique_id
from utils.logger import get_logger

logger = get_logger(__name__)


# Allowed file types — extension → set of acceptable MIME values.
# Enforced at upload-session creation AND re-verified via python-magic
# during finalize, in case the client lied about Content-Type.
_ALLOWED_TYPES: dict[str, set[str]] = {
    ".pdf":  {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    ".doc":  {"application/msword"},
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    },
    ".ppt":  {"application/vnd.ms-powerpoint"},
    ".txt":  {"text/plain"},
}

# Retry policy for transient failures during ingestion (MinIO, Qdrant).
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 8.0
_RETRY_JITTER_RATIO = 0.25


class DocumentService:
    """Coordinates upload sessions, ingestion, listing, and deletion."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: ObjectStore,
        pipeline: RAGPipeline,
        event_bus: EventBus,
        storage: StorageSettings,
        backend: BackendSettings,
    ) -> None:
        self._repo = repo
        self._store = object_store
        self._pipeline = pipeline
        self._bus = event_bus
        self._storage = storage
        self._backend = backend

    async def create_upload_session(
        self, user_id: str, request: UploadSessionRequest,
    ) -> UploadSessionView:
        """Validate metadata, create a pending row, return a presigned PUT URL."""
        self._validate_file_metadata(
            file_name=request.file_name,
            mime_type=request.mime_type,
            size_bytes=request.size_bytes,
        )

        doc_id = generate_unique_id()
        s3_key = self._build_s3_key(user_id, doc_id, request.file_name)
        now = datetime.now(timezone.utc)

        document = Document(
            id=doc_id,
            user_id=user_id,
            content_hash=None,
            file_name=request.file_name,
            mime_type=request.mime_type,
            size_bytes=request.size_bytes,
            s3_bucket=self._storage.s3_bucket,
            s3_key=s3_key,
            collection=request.collection,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        await self._repo.create(document)

        presigned_url = await self._store.generate_presigned_put_url(
            key=s3_key, content_type=request.mime_type,
        )
        expires_at = now + timedelta(
            seconds=self._storage.presigned_url_ttl_seconds,
        )

        logger.info(
            "Upload session created | doc_id=%s | user_id=%s | key=%s | size=%d",
            doc_id, user_id, s3_key, request.size_bytes,
        )
        return UploadSessionView(
            doc_id=doc_id,
            s3_key=s3_key,
            presigned_url=presigned_url,
            expires_at=expires_at,
        )

    async def finalize(
        self, doc_id: str, user_id: str, request_id: Optional[str] = None,
    ) -> FinalizeAck:
        """Kick off background ingestion; return 202-style ack immediately."""
        document = await self._fetch_owned(doc_id, user_id)

        if document.status != "pending":
            logger.info(
                "Finalize ignored (non-pending) | doc_id=%s | status=%s",
                doc_id, document.status,
            )
            return FinalizeAck(doc_id=doc_id, status=document.status)

        asyncio.create_task(
            self._ingest_background(doc_id, user_id, request_id),
            name=f"ingest-{doc_id}",
        )
        return FinalizeAck(doc_id=doc_id, status="processing")

    async def list(
        self,
        user_id: str,
        collection: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> DocumentListView:
        documents = await self._repo.list_by_user(
            user_id=user_id,
            collection=collection,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        return DocumentListView(
            documents=[self._to_view(d) for d in documents],
            count=len(documents),
        )

    async def get(self, doc_id: str, user_id: str) -> DocumentDetailView:
        document = await self._fetch_owned(doc_id, user_id)
        return self._to_view(document)

    async def soft_delete(self, doc_id: str, user_id: str) -> None:
        """Soft-delete row first (user-facing ack), then cascade best-effort."""
        document = await self._fetch_owned(doc_id, user_id)
        await self._repo.soft_delete(doc_id)
        # Best-effort cleanup — outcome does not affect the API response.
        await self._cascade_delete(doc_id, document.s3_key)

    async def _ingest_background(
        self, doc_id: str, user_id: str, request_id: Optional[str],
    ) -> None:
        """Fire-and-forget ingestion task spawned by finalize."""
        publish = self._make_publisher(doc_id, request_id)
        start = time.perf_counter()
        temp_path: Optional[Path] = None

        try:
            # Reload row in a fresh session (the request session is dead here).
            # Capture every field we need INSIDE the block — detached attribute
            # access depends on session-factory config we should not silently rely on.
            async with self._with_repo() as repo:
                document = await repo.find_by_id(doc_id)
                if document is None:
                    raise RuntimeError(f"Document vanished: {doc_id}")
                s3_key = document.s3_key
                expected_mime = document.mime_type
                file_ext = Path(document.file_name).suffix
                logical_collection = document.collection
                await repo.update_status(doc_id, "processing")

            await publish("processing")

            # Verify MinIO has the bytes (defense against client never PUTting).
            metadata = await self._retry(self._store.head_object, s3_key)
            await publish("downloading", size_bytes=int(metadata.get("ContentLength", 0)))

            # Stream MinIO → tempfile + SHA-256.
            temp_path, content_hash = await self._download_and_hash(
                s3_key=s3_key, file_ext=file_ext, doc_id=doc_id,
            )
            await publish("hashed", content_hash=content_hash)

            self._verify_mime_from_disk(temp_path, expected=expected_mime)

            # Dedup probe in its own short session — release the connection
            # before any I/O-heavy follow-up (duplicate cascade or chunking).
            async with self._with_repo() as repo:
                existing = await repo.find_active_by_content_hash(user_id, content_hash)
            duplicate_id = (
                existing.id if existing is not None and existing.id != doc_id else None
            )

            if duplicate_id is not None:
                await self._handle_duplicate(
                    doc_id=doc_id,
                    s3_key=s3_key,
                    existing_id=duplicate_id,
                    publish=publish,
                )
                return

            # Persist hash in its own session so the connection is free during
            # the slow chunking/embedding work that follows.
            async with self._with_repo() as repo:
                await repo.set_content_hash(doc_id, content_hash)

            await publish("chunking")
            # pipeline.ingest positional args: file_path, collection, user_id, doc_id.
            # Pass `logical_collection` so Qdrant payload.metadata.collection
            # matches the documents row — collection-scoped queries depend on it.
            ingestion = await self._retry(
                self._pipeline.ingest,
                str(temp_path), logical_collection, user_id, doc_id,
            )

            async with self._with_repo() as repo:
                await repo.update_status(
                    doc_id, "ready", chunks_count=ingestion.chunks_stored,
                )

            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            await publish(
                "ready",
                chunks_count=ingestion.chunks_stored,
                elapsed_ms=elapsed_ms,
            )
            logger.info(
                "Ingestion complete | doc_id=%s | chunks=%d | elapsed_ms=%.1f",
                doc_id, ingestion.chunks_stored, elapsed_ms,
            )

        except Exception as exc:
            logger.exception("Ingestion failed | doc_id=%s", doc_id)
            await self._rollback(doc_id, reason=str(exc))
            await publish("failed", reason=type(exc).__name__, message=str(exc))

        finally:
            self._unlink_quietly(temp_path)

    def _validate_file_metadata(
        self, file_name: str, mime_type: str, size_bytes: int,
    ) -> None:
        max_bytes = self._backend.max_upload_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {self._backend.max_upload_size_mb} MB cap",
            )
        ext = Path(file_name).suffix.lower()
        allowed_mimes = _ALLOWED_TYPES.get(ext)
        if allowed_mimes is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {ext or '<none>'}",
            )
        if mime_type not in allowed_mimes:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"MIME '{mime_type}' does not match extension '{ext}'",
            )

    def _build_s3_key(self, user_id: str, doc_id: str, file_name: str) -> str:
        safe_name = Path(file_name).name  # strip any path components
        return f"{user_id}/{doc_id}/{safe_name}"

    async def _fetch_owned(self, doc_id: str, user_id: str) -> Document:
        document = await self._repo.find_by_id(doc_id)
        if (
            document is None
            or document.user_id != user_id
            or document.deleted_at is not None
        ):
            # 404 (not 403) so we do not leak existence of other users' docs.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        return document

    @asynccontextmanager
    async def _with_repo(self):
        """Yield a repo bound to a fresh, self-committing session.

        Used by background ingestion paths, where the request-scoped session
        from FastAPI's `get_db_session` is already closed.
        """
        async with session_scope() as session:
            yield DocumentRepository(session)

    async def _download_and_hash(
        self, s3_key: str, file_ext: str, doc_id: str,
    ) -> tuple[Path, str]:
        """Stream MinIO → on-disk temp file; return path + SHA-256.

        The temp file is the caller's to delete; we use `mkstemp` (not
        `NamedTemporaryFile`) because the file must outlive this function
        so the pipeline can read it from disk.
        """
        hasher = hashlib.sha256()
        fd, temp_path_str = tempfile.mkstemp(
            suffix=file_ext, prefix=f"{doc_id}_",
        )
        os.close(fd)
        temp_path = Path(temp_path_str)

        try:
            with temp_path.open("wb") as out:
                async for chunk in self._store.get_object_stream(s3_key):
                    hasher.update(chunk)
                    out.write(chunk)
            return temp_path, hasher.hexdigest()
        except Exception:
            self._unlink_quietly(temp_path)
            raise

    def _verify_mime_from_disk(self, path: Path, expected: str) -> None:
        sniffed = magic.from_file(str(path), mime=True)
        if sniffed != expected:
            raise RuntimeError(
                f"MIME mismatch — declared='{expected}' sniffed='{sniffed}'"
            )

    async def _handle_duplicate(
        self,
        doc_id: str,
        s3_key: str,
        existing_id: str,
        publish: Callable[..., Awaitable[None]],
    ) -> None:
        await self._cascade_delete(doc_id, s3_key)
        async with self._with_repo() as repo:
            await repo.hard_delete(doc_id)
        await publish("duplicate", duplicate_of=existing_id)
        logger.info(
            "Duplicate suppressed | doc_id=%s | existing=%s", doc_id, existing_id,
        )

    async def _rollback(self, doc_id: str, reason: str) -> None:
        """Terminal failure path — clear MinIO + Qdrant + DB."""
        async with self._with_repo() as repo:
            document = await repo.find_by_id(doc_id)
        if document is None:
            return
        await self._cascade_delete(doc_id, document.s3_key)
        async with self._with_repo() as repo:
            await repo.hard_delete(doc_id)
        logger.info("Rollback complete | doc_id=%s | reason=%s", doc_id, reason)

    async def _cascade_delete(self, doc_id: str, s3_key: str) -> None:
        """Best-effort Qdrant + MinIO cleanup; failures are logged, not raised."""
        try:
            await self._delete_qdrant_chunks(doc_id)
        except Exception:
            logger.exception("Qdrant cleanup failed | doc_id=%s", doc_id)
        try:
            await self._retry(
                self._store.delete_object, s3_key, attempts=2,
            )
        except Exception:
            logger.exception(
                "MinIO cleanup failed | doc_id=%s | key=%s", doc_id, s3_key,
            )

    async def _delete_qdrant_chunks(self, doc_id: str) -> None:
        store = getattr(self._pipeline, "_store", None)
        if store is None:
            return
        await store.delete_by_doc_id(doc_id)

    def _make_publisher(
        self, doc_id: str, request_id: Optional[str],
    ) -> Callable[..., Awaitable[None]]:
        """Return a closure that publishes events tagged with doc_id + request_id."""
        async def publish(phase: str, **payload) -> None:
            event = {
                "doc_id": doc_id,
                "phase": phase,
                "ts": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                **payload,
            }
            await self._bus.publish(doc_id, event)
        return publish

    async def _retry(self, fn, *args, attempts: int = _RETRY_ATTEMPTS):
        """Exponential backoff with ±25% jitter. Retries on any Exception.

        Reasonable for our use case (Qdrant/MinIO transient errors); finer-
        grained per-call filtering can come later if needed.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return await fn(*args)
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    raise
                base = min(
                    _RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
                    _RETRY_MAX_DELAY_S,
                )
                jitter = base * _RETRY_JITTER_RATIO * (2 * random.random() - 1)
                wait = max(0.0, base + jitter)
                logger.warning(
                    "Retrying | fn=%s | attempt=%d/%d | wait=%.2fs | error=%s",
                    getattr(fn, "__name__", repr(fn)),
                    attempt, attempts, wait, exc,
                )
                await asyncio.sleep(wait)
        raise last_exc  # unreachable; satisfies type checker

    def _unlink_quietly(self, path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            if path.exists():
                path.unlink()
        except Exception:
            logger.warning("Failed to delete temp file | path=%s", path)

    def _to_view(self, document: Document) -> DocumentDetailView:
        return DocumentDetailView(
            doc_id=document.id,
            user_id=document.user_id,
            file_name=document.file_name,
            mime_type=document.mime_type,
            size_bytes=document.size_bytes,
            collection=document.collection,
            status=document.status,
            chunks_count=document.chunks_count,
            error_message=document.error_message,
            content_hash=document.content_hash,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


def get_document_service(
    request: Request,
    repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentService:
    """FastAPI dependency. Pulls the pipeline from app.state (set by lifespan)."""
    pipeline: Optional[RAGPipeline] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return DocumentService(
        repo=repo,
        object_store=get_object_store(),
        pipeline=pipeline,
        event_bus=get_event_bus(),
        storage=storage_settings,
        backend=backend_settings,
    )
