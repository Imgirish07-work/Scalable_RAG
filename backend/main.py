"""FastAPI app with lifespan-managed RAG pipeline and Arq queue client."""

import asyncio
import os

os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1 import (
    collections as collections_router,
    documents as documents_router,
    health as health_router,
    query as query_router,
)
from backend.middleware import RequestIDMiddleware
from backend.repositories.database import dispose_engine
from backend.services.event_bus import get_event_bus
from backend.services.orphan_sweeper import OrphanSweeper
from backend.services.pipeline_factory import build_query_pipeline
from backend.settings import backend_settings, storage_settings, worker_settings
from backend.storage.object_store import get_object_store
from backend.workers.queue_client import QueueClient
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


async def _initialize_pipeline(app: FastAPI) -> None:
    """Boot the RAG pipeline; flip `app.state.ready` on success."""
    start = time.perf_counter()
    try:
        await get_object_store().ensure_bucket()
        pipeline = await build_query_pipeline()
        app.state.pipeline = pipeline
        app.state.ready = True
        logger.info(
            "Backend ready in %.0f ms", (time.perf_counter() - start) * 1000,
        )
    except asyncio.CancelledError:
        logger.info("Pipeline initialization cancelled mid-boot")
        raise
    except Exception:
        # stay alive in not-ready state so /readyz keeps returning a clean 503
        logger.exception(
            "Pipeline initialization failed — backend will remain not-ready"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Backend starting | app=%s v%s", settings.app_name, settings.app_version,
    )
    app.state.ready = False
    app.state.pipeline = None
    app.state.queue_client = None

    queue_client = QueueClient()
    try:
        await queue_client.start()
        app.state.queue_client = queue_client
    except Exception:
        logger.exception("Queue client failed to start — ingest endpoints will 503")

    init_task = asyncio.create_task(
        _initialize_pipeline(app), name="pipeline-init",
    )

    sweeper = OrphanSweeper(
        get_pipeline=lambda: getattr(app.state, "pipeline", None),
        interval_seconds=storage_settings.sweeper_interval_seconds,
        orphan_max_age_seconds=storage_settings.orphan_sweep_after_seconds,
        dlq_max_age_seconds=storage_settings.failed_dlq_ttl_seconds,
        processing_lease_ttl_seconds=worker_settings.processing_lease_ttl_seconds,
    )
    sweeper_task = asyncio.create_task(sweeper.run(), name="orphan-sweeper")

    yield

    logger.info("Backend shutting down")
    app.state.ready = False

    sweeper.request_stop()

    # cancel in-flight init before tearing down its dependencies
    if not init_task.done():
        init_task.cancel()
    for task in (init_task, sweeper_task):
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            await pipeline.shutdown()
        except Exception:
            logger.exception("Pipeline shutdown failed")
    try:
        await queue_client.close()
    except Exception:
        logger.exception("Queue client close failed")
    try:
        await dispose_engine()
    except Exception:
        logger.exception("DB engine dispose failed")
    try:
        await get_event_bus().close()
    except Exception:
        logger.exception("Event bus close failed")
    logger.info("Backend shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Scalable RAG API",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=backend_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health_router.router)
    app.include_router(query_router.router)
    app.include_router(documents_router.router)
    app.include_router(collections_router.router)
    return app


app = create_app()
