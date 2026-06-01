"""API-side producer for the Arq ingest queue."""

from contextlib import suppress
from typing import Optional

from arq import create_pool
from arq.connections import ArqRedis

from backend.metrics import ingest_jobs_queued_total
from backend.settings import worker_settings
from backend.workers.arq_settings import _redis_settings_from_url
from utils.logger import get_logger

logger = get_logger(__name__)


class QueueUnavailableError(RuntimeError):
    """Raised when the queue is not started or Redis is unreachable."""


class QueueClient:
    """Thin wrapper over arq's redis pool for enqueueing ingest jobs."""

    def __init__(self) -> None:
        self._pool: Optional[ArqRedis] = None

    async def start(self) -> None:
        if self._pool is not None:
            return
        self._pool = await create_pool(
            _redis_settings_from_url(worker_settings.redis_url),
            default_queue_name=worker_settings.queue_name,
        )
        logger.info(
            "Queue client started | queue=%s | redis=%s",
            worker_settings.queue_name, worker_settings.redis_url,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        with suppress(Exception):
            await self._pool.aclose()
        self._pool = None
        logger.info("Queue client closed")

    async def enqueue_ingest(
        self,
        doc_id: str,
        user_id: str,
        request_id: Optional[str] = None,
    ) -> bool:
        if self._pool is None:
            raise QueueUnavailableError("Queue client not started")
        try:
            job = await self._pool.enqueue_job(
                "ingest_document",
                doc_id, user_id, request_id,
                _job_id=doc_id,
                _queue_name=worker_settings.queue_name,
            )
        except Exception as exc:
            logger.exception("Enqueue failed | doc_id=%s", doc_id)
            raise QueueUnavailableError(str(exc)) from exc

        accepted = job is not None
        if accepted:
            ingest_jobs_queued_total.labels(origin="api").inc()
        logger.info(
            "Ingest job enqueued=%s | doc_id=%s | user_id=%s | job_id=%s",
            accepted, doc_id, user_id, doc_id,
        )
        return accepted
