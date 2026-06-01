"""HTTP-layer orchestration for /v1/documents — upload, list, delete, SSE."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from backend.models.document import Document
from backend.repositories.document_repository import (
    DocumentRepository,
    get_document_repository,
)
from backend.schemas.document import (
    DocumentDetailView,
    DocumentListView,
    DownloadView,
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
from backend.workers.queue_client import QueueClient, QueueUnavailableError
from pipeline.rag_pipeline import RAGPipeline
from utils.helpers import generate_unique_id
from utils.logger import get_logger

logger = get_logger(__name__)


# IngestionService re-verifies sniffed bytes via python-magic in case the client lied
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


class DocumentService:
    """Coordinates upload sessions, listing, deletion, and SSE subscriptions."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: ObjectStore,
        pipeline: RAGPipeline,
        event_bus: EventBus,
        queue_client: QueueClient,
        storage: StorageSettings,
        backend: BackendSettings,
    ) -> None:
        self._repo = repo
        self._store = object_store
        self._pipeline = pipeline
        self._bus = event_bus
        self._queue = queue_client
        self._storage = storage
        self._backend = backend

    async def create_upload_session(
        self, user_id: str, request: UploadSessionRequest,
    ) -> UploadSessionView:
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
        document = await self._fetch_owned(doc_id, user_id)
        if document.status != "pending":
            logger.info(
                "Finalize ignored (non-pending) | doc_id=%s | status=%s",
                doc_id, document.status,
            )
            return FinalizeAck(doc_id=doc_id, status=document.status)

        await self._enqueue_ingest(doc_id, user_id, request_id)
        return FinalizeAck(doc_id=doc_id, status="processing")

    async def retry(
        self, doc_id: str, user_id: str, request_id: Optional[str] = None,
    ) -> FinalizeAck:
        document = await self._fetch_owned(doc_id, user_id)
        if document.status != "failed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot retry a document in status '{document.status}'",
            )
        await self._enqueue_ingest(doc_id, user_id, request_id)
        logger.info("Retry kicked off | doc_id=%s | user_id=%s", doc_id, user_id)
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

    async def get_download_url(self, doc_id: str, user_id: str) -> DownloadView:
        document = await self._fetch_owned(doc_id, user_id)
        url = await self._store.generate_presigned_get_url(document.s3_key)
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=storage_settings.presigned_url_ttl_seconds,
        )
        return DownloadView(
            doc_id=doc_id,
            file_name=document.file_name,
            mime_type=document.mime_type,
            presigned_url=url,
            expires_at=expires_at,
        )

    async def soft_delete(self, doc_id: str, user_id: str) -> None:
        document = await self._fetch_owned(doc_id, user_id)
        await self._repo.soft_delete(doc_id)
        await self._cleanup_blob_and_vectors(doc_id, document.s3_key)

    async def subscribe_to_events(self, doc_id: str, user_id: str):
        """Yield a DB snapshot first, then tail the event bus until terminal."""
        document = await self._fetch_owned(doc_id, user_id)
        yield {
            "doc_id": doc_id,
            "phase": self._phase_from_status(document.status),
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": document.status,
            "chunks_count": document.chunks_count,
            "error_message": document.error_message,
            "snapshot": True,
        }

        if document.status in ("ready", "failed"):
            return

        async for event in self._bus.subscribe(doc_id):
            yield event

    async def _enqueue_ingest(
        self, doc_id: str, user_id: str, request_id: Optional[str],
    ) -> None:
        try:
            await self._queue.enqueue_ingest(
                doc_id=doc_id, user_id=user_id, request_id=request_id,
            )
        except QueueUnavailableError as exc:
            logger.error("Queue unavailable | doc_id=%s | error=%s", doc_id, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion queue unavailable",
            ) from exc

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
        safe_name = Path(file_name).name
        return f"{user_id}/{doc_id}/{safe_name}"

    async def _fetch_owned(self, doc_id: str, user_id: str) -> Document:
        document = await self._repo.find_by_id(doc_id)
        if (
            document is None
            or document.user_id != user_id
            or document.deleted_at is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        return document

    async def _cleanup_blob_and_vectors(
        self, doc_id: str, s3_key: str,
    ) -> None:
        """Best-effort vector + blob delete; failures logged, never raised."""
        try:
            store = getattr(self._pipeline, "_store", None)
            if store is not None:
                await store.delete_by_doc_id(doc_id)
        except Exception:
            logger.exception("Qdrant cleanup failed | doc_id=%s", doc_id)
        try:
            await self._store.delete_object(s3_key)
        except Exception:
            logger.exception(
                "MinIO cleanup failed | doc_id=%s | key=%s", doc_id, s3_key,
            )

    @staticmethod
    def _phase_from_status(status_value: str) -> str:
        if status_value == "ready":
            return "ready"
        if status_value == "failed":
            return "failed"
        if status_value == "processing":
            return "processing"
        return "pending"

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
    """FastAPI dependency. Pulls pipeline + queue client from app.state."""
    pipeline: Optional[RAGPipeline] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    queue_client: Optional[QueueClient] = getattr(
        request.app.state, "queue_client", None,
    )
    if queue_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue client not initialized",
        )
    return DocumentService(
        repo=repo,
        object_store=get_object_store(),
        pipeline=pipeline,
        event_bus=get_event_bus(),
        queue_client=queue_client,
        storage=storage_settings,
        backend=backend_settings,
    )
