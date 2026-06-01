"""Builds and warms up the RAGPipeline. Shared by API lifespan and worker startup."""

from backend.settings import backend_settings
from cache.cache_manager import CacheManager
from config.settings import settings
from llm.llm_factory import LLMFactory
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


async def build_pipeline() -> RAGPipeline:
    """Construct and warm up the RAG pipeline. Caller owns shutdown."""
    llm = LLMFactory.create_from_settings()
    logger.info("LLM ready | %s/%s", llm.provider_name, llm.model_name)

    store = QdrantStore(in_memory=False, search_mode=settings.RAG_RETRIEVAL_MODE)
    cache = CacheManager(settings) if settings.cache_enabled else None

    pipeline = RAGPipeline(llm=llm, store=store, cache=cache)
    await pipeline.initialize()

    pipeline.configure_agents(
        collections={settings.qdrant_collection_name: "All user documents"},
        max_concurrent=backend_settings.max_concurrent_subqueries,
    )
    logger.info(
        "Pipeline ready | physical_collection=%s",
        settings.qdrant_collection_name,
    )
    return pipeline
