"""Periodic sweep: reaps stale pending/failed rows and stuck 'processing' leases."""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from backend.metrics import ingest_jobs_failed_total
from backend.repositories.database import session_scope
from backend.repositories.document_repository import DocumentRepository
from backend.services.event_bus import EventBus, get_event_bus
from backend.storage.object_store import ObjectStore, get_object_store
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


_SWEEP_BATCH_LIMIT = 500
_STUCK_ERROR_MESSAGE = "Stuck-processing lease expired — worker died mid-job"


class OrphanSweeper:
    """Periodic cleanup loop. Owns its own asyncio task and stop event."""

    def __init__(
        self,
        get_pipeline: Callable[[], Optional[RAGPipeline]],
        interval_seconds: int,
        orphan_max_age_seconds: int,
        dlq_max_age_seconds: int,
        processing_lease_ttl_seconds: int,
        object_store: Optional[ObjectStore] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._get_pipeline = get_pipeline
        self._interval = interval_seconds
        self._orphan_age = orphan_max_age_seconds
        self._dlq_age = dlq_max_age_seconds
        self._processing_lease = processing_lease_ttl_seconds
        self._store = object_store or get_object_store()
        self._bus = event_bus or get_event_bus()
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        logger.info(
            "Orphan sweeper started | interval=%ds | orphan_age=%ds | "
            "dlq_age=%ds | processing_lease=%ds",
            self._interval, self._orphan_age, self._dlq_age,
            self._processing_lease,
        )
        try:
            # defer first tick so pipeline init finishes before we contend for db connections
            if await self._sleep_or_stop():
                return
            while not self._stop_event.is_set():
                try:
                    await self._sweep_once()
                except Exception:
                    logger.exception("Sweeper tick failed; continuing")
                if await self._sleep_or_stop():
                    return
        finally:
            logger.info("Orphan sweeper stopped")

    async def _sleep_or_stop(self) -> bool:
        """Wait one interval. Returns True if stop was requested."""
        try:
            await asyncio.wait_for(
                self._stop_event.wait(), timeout=self._interval,
            )
        except asyncio.TimeoutError:
            return False
        return True

    def request_stop(self) -> None:
        self._stop_event.set()

    async def _sweep_once(self) -> None:
        orphans = await self._reap_status("pending", self._orphan_age)
        dlq = await self._reap_status("failed", self._dlq_age)
        stuck = await self._reap_stuck_processing()
        if orphans or dlq or stuck:
            logger.info(
                "Sweep complete | orphans_reaped=%d | dlq_reaped=%d | stuck_dlq=%d",
                orphans, dlq, stuck,
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

    async def _reap_stuck_processing(self) -> int:
        async with session_scope() as session:
            stuck_ids = await DocumentRepository(session).list_stuck_processing(
                lease_ttl_seconds=self._processing_lease,
                limit=_SWEEP_BATCH_LIMIT,
            )

        marked = 0
        for doc_id in stuck_ids:
            try:
                async with session_scope() as session:
                    moved = await DocumentRepository(session).mark_stuck_failed(
                        doc_id=doc_id,
                        lease_ttl_seconds=self._processing_lease,
                        error_message=_STUCK_ERROR_MESSAGE,
                    )
                if not moved:
                    continue
                # minio blob is preserved so the user can retry via POST /retry
                try:
                    await self._delete_qdrant_chunks(doc_id)
                except Exception:
                    logger.exception(
                        "Qdrant cleanup failed for stuck row | doc_id=%s", doc_id,
                    )
                await self._publish_failed(doc_id)
                ingest_jobs_failed_total.labels(reason="stuck").inc()
                marked += 1
            except Exception:
                logger.exception("Stuck-processing reap failed | doc_id=%s", doc_id)
        return marked

    async def _publish_failed(self, doc_id: str) -> None:
        try:
            await self._bus.publish(
                doc_id,
                {
                    "doc_id": doc_id,
                    "phase": "failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reason": "StuckLeaseExpired",
                    "message": _STUCK_ERROR_MESSAGE,
                },
            )
        except Exception:
            logger.exception(
                "Failed-event publish failed for stuck row | doc_id=%s", doc_id,
            )

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
