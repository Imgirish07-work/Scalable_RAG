"""Arq worker entrypoint — `arq backend.workers.arq_settings.WorkerSettings`."""

from urllib.parse import urlparse

from arq.connections import RedisSettings

from backend.repositories.database import dispose_engine
from backend.services.event_bus import get_event_bus
from backend.services.ingestion_service import IngestionService
from backend.services.pipeline_factory import build_pipeline
from backend.settings import worker_settings
from backend.storage.object_store import get_object_store
from backend.workers.tasks import ingest_document
from utils.logger import get_logger

logger = get_logger(__name__)


def _redis_settings_from_url(url: str) -> RedisSettings:
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or 0),
        password=parsed.password,
    )


async def on_startup(ctx: dict) -> None:
    logger.info(
        "Worker booting | queue=%s | max_jobs=%d",
        worker_settings.queue_name, worker_settings.worker_max_jobs,
    )
    await get_object_store().ensure_bucket()

    pipeline = await build_pipeline()
    ctx["pipeline"] = pipeline
    ctx["ingestion"] = IngestionService(
        object_store=get_object_store(),
        pipeline=pipeline,
        event_bus=get_event_bus(),
    )
    logger.info("Worker ready | queue=%s", worker_settings.queue_name)


async def on_shutdown(ctx: dict) -> None:
    logger.info("Worker shutting down")
    pipeline = ctx.get("pipeline")
    if pipeline is not None:
        try:
            await pipeline.shutdown()
        except Exception:
            logger.exception("Pipeline shutdown failed")
    try:
        await get_event_bus().close()
    except Exception:
        logger.exception("Event bus close failed")
    try:
        await dispose_engine()
    except Exception:
        logger.exception("DB engine dispose failed")
    logger.info("Worker shutdown complete")


class WorkerSettings:
    """Arq reads these attributes by name; keep them canonical."""

    functions = [ingest_document]
    redis_settings = _redis_settings_from_url(worker_settings.redis_url)
    queue_name = worker_settings.queue_name
    max_jobs = worker_settings.worker_max_jobs
    job_timeout = worker_settings.worker_job_timeout_seconds
    max_tries = worker_settings.arq_max_tries
    # 0 = purge result on completion so retries can reuse _job_id=doc_id.
    keep_result = 0
    on_startup = on_startup
    on_shutdown = on_shutdown
