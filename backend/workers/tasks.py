"""Arq task entrypoints — registered in `arq_settings.WorkerSettings.functions`."""

from typing import Optional

from backend.services.ingestion_service import IngestionService
from utils.logger import get_logger

logger = get_logger(__name__)


async def ingest_document(
    ctx: dict,
    doc_id: str,
    user_id: str,
    request_id: Optional[str] = None,
) -> None:
    ingestion: IngestionService = ctx["ingestion"]
    logger.info(
        "Worker received ingest job | doc_id=%s | user_id=%s | job_id=%s",
        doc_id, user_id, ctx.get("job_id"),
    )
    await ingestion.run(
        doc_id=doc_id, user_id=user_id, request_id=request_id,
    )
