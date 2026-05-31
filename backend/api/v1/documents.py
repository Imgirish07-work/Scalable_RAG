"""HTTP endpoints for the documents resource.

Architecture-A flow:
  POST   /v1/ingest                          start ingestion → presigned URL
  POST   /v1/documents/{doc_id}/finalize     trigger background ingestion
  GET    /v1/documents/{doc_id}/events       SSE stream of ingestion progress
  POST   /v1/documents/{doc_id}/retry        re-run a failed ingestion (DLQ)
  GET    /v1/documents                       list with filters
  GET    /v1/documents/{doc_id}              single doc detail
  DELETE /v1/documents/{doc_id}              soft-delete + cascade
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from backend.dependencies import get_current_user_id
from backend.schemas.document import (
    DocumentDetailView,
    DocumentListView,
    DownloadView,
    FinalizeAck,
    UploadSessionRequest,
    UploadSessionView,
)
from backend.services.document_service import DocumentService, get_document_service
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["documents"])


# SSE protocol bits.
_SSE_KEEPALIVE_INTERVAL_S = 15.0


@router.post(
    "/ingest",
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
    request_id = getattr(request.state, "request_id", None)
    return await service.finalize(
        doc_id=doc_id, user_id=user_id, request_id=request_id,
    )


@router.post(
    "/documents/{doc_id}/retry",
    response_model=FinalizeAck,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ingestion(
    doc_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> FinalizeAck:
    """Re-run ingestion for a DLQ row using its preserved MinIO blob."""
    request_id = getattr(request.state, "request_id", None)
    return await service.retry(
        doc_id=doc_id, user_id=user_id, request_id=request_id,
    )


@router.get("/documents/{doc_id}/events")
async def stream_events(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> StreamingResponse:
    """Server-Sent Events stream of ingestion progress for a single doc.

    Eagerly fetches the first event (snapshot) so tenancy errors surface as
    proper HTTP status codes BEFORE the StreamingResponse is committed.
    """
    event_iter = service.subscribe_to_events(doc_id=doc_id, user_id=user_id)
    try:
        first_event = await event_iter.__anext__()
    except StopAsyncIteration:
        first_event = None
    return StreamingResponse(
        _sse_stream(first_event, event_iter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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


@router.get("/documents/{doc_id}/download", response_model=DownloadView)
async def get_document_download_url(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> DownloadView:
    """Return a short-lived presigned GET URL the browser can fetch from MinIO."""
    return await service.get_download_url(doc_id=doc_id, user_id=user_id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),
) -> Response:
    await service.soft_delete(doc_id=doc_id, user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _sse_stream(first_event, event_iter):
    """SSE wire format wrapper with a pre-fetched first event.

    Uses `asyncio.wait` (not `wait_for`) for the keepalive timeout so the
    pending `__anext__` task is NOT cancelled on each idle interval — that
    would close the underlying generator and end the stream prematurely.
    """
    next_task = None
    try:
        if first_event is not None:
            yield f"event: phase\ndata: {json.dumps(first_event)}\n\n"

        while True:
            if next_task is None:
                next_task = asyncio.create_task(event_iter.__anext__())

            done, _ = await asyncio.wait(
                {next_task}, timeout=_SSE_KEEPALIVE_INTERVAL_S,
            )

            if next_task in done:
                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    return
                next_task = None
                yield f"event: phase\ndata: {json.dumps(event)}\n\n"
            else:
                yield ": keepalive\n\n"
    except Exception:
        logger.exception("SSE stream aborted")
        return
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
        try:
            await event_iter.aclose()
        except Exception:
            pass
