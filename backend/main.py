"""FastAPI app with lifespan-managed RAGPipeline singleton."""

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
from backend.settings import backend_settings
from backend.storage.object_store import get_object_store
from cache.cache_manager import CacheManager
from config.settings import settings
from llm.llm_factory import LLMFactory
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


async def _initialize_pipeline(app: FastAPI) -> None:
    """Build and warm up the RAG pipeline; flip `app.state.ready` on success.

    Runs in the background after the lifespan yields, so uvicorn can serve
    health checks and reject non-ready /v1/* calls with a clean 503 while
    the heavy ONNX/SPLADE/Qdrant boot continues. Every blocking sync call
    downstream is already routed through `asyncio.to_thread`, so this
    coroutine never holds the event loop.
    """
    start = time.perf_counter()
    try:
        # Object store first — cheap, but downstream document upload paths depend
        # on the bucket existing with CORS applied, so we fail fast if MinIO is
        # mis-configured rather than discovering it on the first upload.
        await get_object_store().ensure_bucket()

        llm = LLMFactory.create_from_settings()
        logger.info("LLM ready | %s/%s", llm.provider_name, llm.model_name)

        store = QdrantStore(in_memory=False, search_mode=settings.RAG_RETRIEVAL_MODE)
        cache = CacheManager(settings) if settings.cache_enabled else None

        pipeline = RAGPipeline(llm=llm, store=store, cache=cache)
        await pipeline.initialize()

        # Single physical Qdrant collection; tenancy enforced via user_id filter.
        agent_collections = {
            settings.qdrant_collection_name: "All user documents",
        }
        pipeline.configure_agents(
            collections=agent_collections,
            max_concurrent=backend_settings.max_concurrent_subqueries,
        )
        logger.info(
            "Agent layer configured | physical_collection=%s",
            settings.qdrant_collection_name,
        )

        app.state.pipeline = pipeline
        app.state.ready = True
        logger.info("Backend ready in %.0f ms", (time.perf_counter() - start) * 1000)

    except asyncio.CancelledError:
        # Triggered when shutdown arrives mid-boot; cleanup runs in lifespan.
        logger.info("Pipeline initialization cancelled mid-boot")
        raise
    except Exception:
        # Stay alive in not-ready state instead of crash-looping — operators
        # can inspect logs and `/readyz` will keep reporting 503 cleanly.
        logger.exception(
            "Pipeline initialization failed — backend will remain not-ready"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot quickly; defer heavy init to a background task.

    Yielding right after spawning the init task lets uvicorn begin handling
    requests immediately. Docker's healthcheck on `/healthz` therefore passes
    from process start, eliminating the restart loop that occurs when an
    aggressive healthcheck window is shorter than the model-load time.
    """
    logger.info(
        "Backend starting | app=%s v%s", settings.app_name, settings.app_version
    )
    app.state.ready = False
    app.state.pipeline = None

    init_task = asyncio.create_task(
        _initialize_pipeline(app), name="pipeline-init",
    )

    yield

    logger.info("Backend shutting down")
    app.state.ready = False

    # Cancel any in-flight init before tearing down dependencies it might own.
    if not init_task.done():
        init_task.cancel()
    try:
        await init_task
    except (asyncio.CancelledError, Exception):
        pass

    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            await pipeline.shutdown()
        except Exception:
            logger.exception("Pipeline shutdown failed")
    try:
        await dispose_engine()
    except Exception:
        logger.exception("DB engine dispose failed")
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
