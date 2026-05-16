"""FastAPI app with lifespan-managed RAGPipeline singleton."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import backend_settings
from backend.middleware.request_id import RequestIDMiddleware
from backend.repos.base import dispose_engine
from backend.routers import (
    auth as auth_router,
    collections as collections_router,
    health as health_router,
    ingest as ingest_router,
    query as query_router,
)
from cache.cache_manager import CacheManager
from config.settings import settings
from llm.llm_factory import LLMFactory
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot subsystems on startup; tear them down on shutdown."""
    logger.info("Backend starting | app=%s v%s", settings.app_name, settings.app_version)
    app.state.ready = False
    app.state.pipeline = None

    start = time.perf_counter()

    llm = LLMFactory.create_from_settings()
    logger.info("LLM ready | %s/%s", llm.provider_name, llm.model_name)

    store = QdrantStore(in_memory=False, search_mode=settings.RAG_RETRIEVAL_MODE)
    cache = CacheManager(settings) if settings.cache_enabled else None

    pipeline = RAGPipeline(llm=llm, store=store, cache=cache)
    await pipeline.initialize()

    collections = backend_settings.collections_dict
    if collections:
        pipeline.configure_agents(
            collections=collections,
            max_concurrent=backend_settings.max_concurrent_subqueries,
        )
        logger.info("Agent layer configured | collections=%d", len(collections))
    else:
        logger.info("Agent layer disabled (no collections registered)")

    app.state.pipeline = pipeline
    app.state.ready = True
    logger.info("Backend ready in %.0f ms", (time.perf_counter() - start) * 1000)

    yield

    logger.info("Backend shutting down")
    app.state.ready = False
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
    app.include_router(auth_router.router)
    app.include_router(query_router.router)
    app.include_router(ingest_router.router)
    app.include_router(collections_router.router)
    return app


app = create_app()
