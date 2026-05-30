"""HTTP endpoints for the documents resource.

New Architecture-A flow (steps 2+):
  POST   /v1/documents                       create upload session → presigned URL
  POST   /v1/documents/{doc_id}/finalize     trigger background ingestion
  GET    /v1/documents                       list with filters
  GET    /v1/documents/{doc_id}              single doc detail (poll for status)
  DELETE /v1/documents/{doc_id}              soft-delete + cascade

Legacy synchronous flow (kept until step 3 swaps it out):
  POST   /v1/ingest                          multipart upload + sync ingest
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from backend.dependencies import get_current_user_id, get_pipeline
from backend.metrics import ingest_chunks_total, ingest_total
from backend.schemas.document import (
    DocumentCreatedView,
    DocumentDetailView,
    DocumentListView,
    FinalizeAck,
    UploadSessionRequest,
    UploadSessionView,
)
from backend.services.document_service import DocumentService, get_document_service
from backend.settings import backend_settings
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["documents"])

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post(
    "/documents",
    response_model=UploadSessionView,
    status_code=status.HTTP_201_CREATED,
)
async def create_upload_session(
    payload: UploadSessionRequest,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> UploadSessionView:
    """Reserve a doc_id, create a pending row, and return a presigned PUT URL."""
    return await service.create_upload_session(user_id=user_id, request=payload)


@router.post(
    "/documents/{doc_id}/finalize",
    response_model=FinalizeAck,
    status_code=status.HTTP_202_ACCEPTED,
)
async def finalize_upload(
    doc_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> FinalizeAck:
    """Kick off background ingestion. Returns immediately with processing ack."""
    request_id = getattr(request.state, "request_id", None)
    return await service.finalize(
        doc_id=doc_id, user_id=user_id, request_id=request_id,
    )


@router.get("/documents", response_model=DocumentListView)
async def list_documents(
    collection: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListView:
    return await service.list(
        user_id=user_id,
        collection=collection,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetailView)
async def get_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> DocumentDetailView:
    return await service.get(doc_id=doc_id, user_id=user_id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> Response:
    await service.soft_delete(doc_id=doc_id, user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Legacy /v1/ingest — removed in step 3 once the new flow is verified in the UI.
@router.post(
    "/ingest",
    response_model=DocumentCreatedView,
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    request: Request,
    file: UploadFile = File(..., description="PDF, DOCX, TXT, MD, or HTML"),
    collection: str = Form(default="default"),
    user_id: str = Depends(get_current_user_id),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentCreatedView:
    """Stream the upload to a temp file, run pipeline.ingest, then delete the temp."""
    request_id = getattr(request.state, "request_id", None)
    max_bytes = backend_settings.max_upload_size_mb * 1024 * 1024
    doc_id = str(uuid.uuid4())

    temp_dir = Path(backend_settings.ingest_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename or "upload.bin").name
    temp_path = temp_dir / f"{doc_id}_{original_name}"

    try:
        try:
            total_bytes = 0
            with open(temp_path, "wb") as out:
                while True:
                    chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Upload exceeds {backend_settings.max_upload_size_mb} MB",
                        )
                    out.write(chunk)

            logger.info(
                "Ingest received | request_id=%s | doc_id=%s | bytes=%d",
                request_id, doc_id, total_bytes,
            )
            result = await pipeline.ingest(
                file_path=str(temp_path),
                collection=collection,
                user_id=user_id,
                doc_id=doc_id,
            )
        except HTTPException:
            _record_metric("rejected")
            raise
        except Exception as exc:
            _record_metric("error")
            logger.exception(
                "Ingest failed | request_id=%s | doc_id=%s | error=%s",
                request_id, doc_id, exc,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ingest failed: {type(exc).__name__}",
            )
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            logger.warning("Failed to clean up temp file: %s", temp_path)

    _record_metric("ok", chunks=result.chunks_stored)
    return DocumentCreatedView(doc_id=doc_id, result=result)


def _record_metric(outcome: str, chunks: int = 0) -> None:
    try:
        ingest_total.labels(outcome=outcome).inc()
        if chunks:
            ingest_chunks_total.inc(chunks)
    except Exception:
        logger.warning("Failed to record ingest metric", exc_info=True)
