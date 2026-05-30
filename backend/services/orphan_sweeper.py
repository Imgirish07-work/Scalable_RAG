"""Periodic background sweeper for orphan and DLQ document rows.

Runs as a single asyncio task spawned in the FastAPI lifespan. Each tick:

  - reaps `status='pending'` rows older than `orphan_max_age_seconds`
    (user never finished the upload)
  - reaps `status='failed'` rows older than `dlq_max_age_seconds`
    (DLQ retention window expired)

Each reap cascades to MinIO (blob delete) and Qdrant (chunk delete) before
hard-deleting the DB row. Per-row failures are logged and skipped — one bad
row never aborts the pass.
"""

import asyncio
from typing import Callable, Optional

from backend.repositories.database import session_scope
from backend.repositories.document_repository import DocumentRepository
from backend.storage.object_store import ObjectStore, get_object_store
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


_SWEEP_BATCH_LIMIT = 500


class OrphanSweeper:
    """Periodic cleanup loop. Owns its own asyncio task and stop event."""

    def __init__(
        self,
        get_pipeline: Callable[[], Optional[RAGPipeline]],
        interval_seconds: int,
        orphan_max_age_seconds: int,
        dlq_max_age_seconds: int,
        object_store: Optional[ObjectStore] = None,
    ) -> None:
        # Pipeline is supplied via a getter so the sweeper can start before
        # pipeline init completes — first sweeps simply skip Qdrant cleanup.
        self._get_pipeline = get_pipeline
        self._interval = interval_seconds
        self._orphan_age = orphan_max_age_seconds
        self._dlq_age = dlq_max_age_seconds
        self._store = object_store or get_object_store()
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        logger.info(
            "Orphan sweeper started | interval=%ds | orphan_age=%ds | dlq_age=%ds",
            self._interval, self._orphan_age, self._dlq_age,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    await self._sweep_once()
                except Exception:
                    logger.exception("Sweeper tick failed; continuing")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._interval,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("Orphan sweeper stopped")

    def request_stop(self) -> None:
        self._stop_event.set()

    async def _sweep_once(self) -> None:
        orphans = await self._reap_status("pending", self._orphan_age)
        dlq = await self._reap_status("failed", self._dlq_age)
        if orphans or dlq:
            logger.info(
                "Sweep complete | orphans_reaped=%d | dlq_reaped=%d", orphans, dlq,
            )

    async def _reap_status(self, target_status: str, max_age_seconds: int) -> int:
        async with session_scope() as session:
            stale = await DocumentRepository(session).list_stale_by_status(
                status=target_status,
                older_than_seconds=max_age_seconds,
                limit=_SWEEP_BATCH_LIMIT,
            )

        reaped = 0
        for doc_id, s3_key in stale:
            try:
                await self._cascade_delete(doc_id, s3_key)
                async with session_scope() as session:
                    await DocumentRepository(session).hard_delete(doc_id)
                reaped += 1
                logger.info(
                    "Row reaped | status=%s | doc_id=%s | key=%s",
                    target_status, doc_id, s3_key,
                )
            except Exception:
                logger.exception("Reap failed | doc_id=%s | key=%s", doc_id, s3_key)
        return reaped

    async def _cascade_delete(self, doc_id: str, s3_key: str) -> None:
        try:
            await self._delete_qdrant_chunks(doc_id)
        except Exception:
            logger.exception("Qdrant cleanup failed | doc_id=%s", doc_id)
        try:
            await self._store.delete_object(s3_key)
        except Exception:
            logger.exception("MinIO cleanup failed | doc_id=%s | key=%s", doc_id, s3_key)

    async def _delete_qdrant_chunks(self, doc_id: str) -> None:
        pipeline: Optional[RAGPipeline] = self._get_pipeline()
        if pipeline is None:
            return
        store = getattr(pipeline, "_store", None)
        if store is None:
            return
        await store.delete_by_doc_id(doc_id)
