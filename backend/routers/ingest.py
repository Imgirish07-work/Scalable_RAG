"""POST /v1/ingest — synchronous multipart file upload."""

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from backend.auth.principal import Principal
from backend.config import backend_settings
from backend.deps import get_pipeline, get_principal
from backend.models.ingest import ApiIngestResponse
from backend.observability.metrics import ingest_chunks_total, ingest_total
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["ingest"])

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post("/ingest", response_model=ApiIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    request: Request,
    file: UploadFile = File(..., description="PDF, DOCX, TXT, MD, or HTML"),
    collection: str = Form(..., description="Target Qdrant collection"),
    principal: Principal = Depends(get_principal),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ApiIngestResponse:
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
                "Ingest received | request_id=%s | user_id=%s | doc_id=%s | bytes=%d",
                request_id, principal.user_id, doc_id, total_bytes,
            )

            result = await pipeline.ingest(
                file_path=str(temp_path),
                collection=collection,
                user_id=principal.user_id,
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
    return ApiIngestResponse(doc_id=doc_id, result=result)


def _record_metric(outcome: str, chunks: int = 0) -> None:
    """Best-effort metric write; never fails the caller."""
    try:
        ingest_total.labels(outcome=outcome).inc()
        if chunks:
            ingest_chunks_total.inc(chunks)
    except Exception:
        logger.warning("Failed to record ingest metric", exc_info=True)
