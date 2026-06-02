"""Facade over all subsystems (LLM, vector store, cache, RAG variants, agent layer); owns lifecycle, query routing, ingestion, health, and fallback."""

import asyncio
import json
import time
from typing import Awaitable, Callable, Literal, Optional

PipelineMode = Literal["query", "ingest"]

from agents.agent_orchestrator import AgentOrchestrator
from agents.exceptions.agent_exceptions import AgentError
from agents.planner.complexity_detector import should_decompose
from cache.cache_manager import CacheManager
from chunking.chunker import Chunker
from chunking.document_cleaner import DocumentCleaner
from chunking.structure_preserver import StructurePreserver
from config.settings import settings
from llm.contracts.base_llm import BaseLLM
from llm.exceptions.llm_exceptions import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm.llm_factory import LLMFactory
from pipeline.exceptions.pipeline_exceptions import (
    PipelineFallbackExhaustedError,
    PipelineIngestionError,
    PipelineInitError,
    PipelineValidationError,
)
from pipeline.models.pipeline_request import (
    IngestionResult,
    PipelineHealthStatus,
    PipelineQuery,
)
from rag.base_rag import BaseRAG
from rag.exceptions.rag_exceptions import (
    RAGError,
    RAGGenerationError,
    RAGRetrievalError,
)
from rag.models.rag_request import RAGConfig, RAGRequest
from rag.models.rag_response import RAGResponse
from rag.rag_factory import RAGFactory
from vectorstore.embeddings import get_embeddings
from vectorstore.qdrant_store import QdrantStore
from utils.helpers import hash_text
from utils.logger import get_logger

logger = get_logger(__name__)

_FALLBACK_VARIANT = "simple"


class RAGPipeline:
    """Single entry point for the entire RAG system."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        fallback_llm: Optional[BaseLLM] = None,
        store: Optional[QdrantStore] = None,
        cache: Optional[CacheManager] = None,
        mode: PipelineMode = "query",
    ) -> None:
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._store = store
        self._cache = cache
        self._mode: PipelineMode = mode
        self._initialized = False
        self._agent_orchestrator: Optional[AgentOrchestrator] = None
        self._collections: dict[str, str] = {}
        # all share self._store._client so they query the same Qdrant instance
        self._collection_stores: dict[str, QdrantStore] = {}

    async def initialize(self) -> None:
        """Boot all subsystems in dependency order."""
        if self._initialized:
            logger.info("Pipeline already initialized, skipping")
            return

        logger.info("Pipeline initializing subsystems | mode=%s", self._mode)
        init_start = time.perf_counter()

        try:
            # ingest pods skip LLM/cache/agents — they only embed + upsert
            if self._mode == "query":
                if self._llm is None:
                    self._llm = LLMFactory.create_from_settings()
                    logger.info(
                        "Primary LLM created: %s/%s",
                        self._llm.provider_name, self._llm.model_name,
                    )

                if self._fallback_llm is None:
                    self._fallback_llm = self._try_create_fallback_llm()

            if self._store is None:
                self._store = QdrantStore(
                    in_memory=settings.debug,
                    search_mode=settings.RAG_RETRIEVAL_MODE,
                )
            await self._store.initialize()
            logger.info("Vector store initialized")

            if self._mode == "query":
                if self._cache is None and settings.cache_enabled:
                    self._cache = CacheManager(settings)
                if self._cache:
                    await self._cache.initialize()
                    logger.info("Cache initialized")

        except PipelineInitError:
            raise
        except Exception as exc:
            raise PipelineInitError(
                message=f"Pipeline initialization failed: {exc}",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            ) from exc

        await self._run_warmup()

        elapsed = (time.perf_counter() - init_start) * 1000
        self._initialized = True
        logger.info("Pipeline initialized in %.1fms", elapsed)

    async def shutdown(self) -> None:
        """Tear down all subsystems in reverse dependency order."""
        logger.info("Pipeline shutting down")

        if self._cache:
            try:
                await self._cache.close()
                logger.info("Cache shut down")
            except Exception:
                logger.exception("Cache shutdown failed")

        if self._store:
            try:
                await self._store.close()
                logger.info("Vector store shut down")
            except Exception:
                logger.exception("Vector store shutdown failed")

        self._initialized = False
        logger.info("Pipeline shutdown complete")

    async def _run_warmup(self) -> None:
        """Force all heavy models to load before the first real query."""
        warmup_start = time.perf_counter()
        logger.info("Pipeline warm-up starting...")

        # 200-token synthetic warmup chunk
        _WARMUP_TEXT = (
            "The quick brown fox jumps over the lazy dog near the riverbank. "
            "Machine learning models require substantial computational resources during "
            "both training and inference phases. Vector embeddings encode semantic meaning "
            "into dense numerical representations that enable similarity search. "
            "Large language models have transformed natural language processing tasks "
            "including summarization, question answering, and document retrieval. "
            "Retrieval-augmented generation combines dense retrieval with generative models "
            "to ground responses in factual source documents. Chunking strategies determine "
            "how documents are split into indexable units for downstream embedding. "
        )
        _WARMUP_BATCH = [_WARMUP_TEXT for _ in range(100)]

        async def _warmup_embeddings() -> None:
            model = await asyncio.to_thread(get_embeddings)
            await asyncio.to_thread(model.embed_documents, _WARMUP_BATCH)
            logger.debug("Warm-up: dense embedder ready (batch=%d)", len(_WARMUP_BATCH))

        async def _warmup_splade() -> None:
            if self._store and hasattr(self._store, "_get_sparse_embeddings"):
                sparse = await asyncio.to_thread(self._store._get_sparse_embeddings)
                await asyncio.to_thread(sparse.embed_documents, _WARMUP_BATCH)
                logger.debug("Warm-up: SPLADE ready (batch=%d)", len(_WARMUP_BATCH))

        async def _warmup_qdrant() -> None:
            if self._store:
                try:
                    await self._store.similarity_search_with_vectors("warmup", k=1)
                    logger.debug("Warm-up: Qdrant HNSW index ready")
                except Exception:
                    pass  # empty collection is fine, hnsw still loads

        async def _warmup_llm() -> None:
            if self._llm:
                try:
                    await self._llm.generate("Reply with: OK", max_tokens=2)
                    logger.debug("Warm-up: LLM connection pool ready")
                except Exception:
                    pass  # non-critical, connection opens on first real query

        coros = [_warmup_embeddings(), _warmup_splade(), _warmup_qdrant()]
        warmup_names = ["embedding", "splade", "qdrant"]
        if self._mode == "query":
            coros.append(_warmup_llm())
            warmup_names.append("llm")

        results = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(warmup_names, results):
            if isinstance(result, Exception):
                logger.warning("Warm-up failed for %s: %s", name, result)

        elapsed = (time.perf_counter() - warmup_start) * 1000
        logger.info("Pipeline warm-up complete in %.1fms", elapsed)

    async def health_check(self) -> PipelineHealthStatus:
        """Check health of all subsystems."""
        llm_status = await self._check_llm_health()
        store_status = await self._check_store_health()
        cache_status = await self._check_cache_health()

        ready = llm_status == "ok" and store_status == "ok"

        return PipelineHealthStatus(
            ready=ready,
            llm=llm_status,
            vector_store=store_status,
            cache=cache_status,
            details={
                "primary_llm": (
                    f"{self._llm.provider_name}/{self._llm.model_name}"
                    if self._llm else "not configured"
                ),
                "fallback_llm": (
                    f"{self._fallback_llm.provider_name}/{self._fallback_llm.model_name}"
                    if self._fallback_llm else "not configured"
                ),
            },
        )

    def configure_agents(
        self,
        collections: dict[str, str],
        max_concurrent: int = 4,
    ) -> None:
        """Configure the agent layer for query decomposition."""
        self._ensure_initialized()
        self._collections = collections

        fast_llm = self._try_create_fast_llm()

        self._agent_orchestrator = AgentOrchestrator(
            strong_llm=self._llm,
            fast_llm=fast_llm or self._llm,
            store_factory=self._get_store_for_collection,
            collections=collections,
            embeddings_fn=get_embeddings,
            max_concurrent=max_concurrent,
            fallback_llm=self._fallback_llm,
        )
        logger.info(
            "Agent layer configured | collections=%d | fast_llm=%s",
            len(collections),
            fast_llm.model_name if fast_llm else "fallback_to_strong",
        )

    async def query(
        self,
        pipeline_query: PipelineQuery,
    ) -> RAGResponse:
        """Execute a query through the full RAG pipeline."""
        self._ensure_initialized()
        request = self._validate_and_convert(pipeline_query)

        logger.info(
            "Pipeline processing query, request_id=%s, collection=%s, variant=%s",
            request.request_id,
            request.collection_name,
            request.config.rag_variant if request.config else "default",
        )

        query_start = time.perf_counter()

        try:
            response = await self._execute_query(request, self._llm)
            self._log_query_metrics(request, response, query_start)
            return response

        except (LLMAuthError, LLMRateLimitError):
            raise

        except (RAGError, LLMTimeoutError) as exc:
            logger.warning(
                "Primary execution failed for request_id=%s: %s",
                request.request_id, exc,
            )
            return await self._handle_fallback(request, exc, query_start)

    async def query_raw(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        """Execute a query using a raw RAGRequest (advanced usage)."""
        self._ensure_initialized()

        logger.info(
            "Pipeline processing raw query, request_id=%s",
            request.request_id,
        )

        query_start = time.perf_counter()

        try:
            rag = await self._build_rag_for_request(request, self._llm)
            response = await rag.query(request)
            self._log_query_metrics(request, response, query_start)
            return response

        except (LLMAuthError, LLMRateLimitError):
            raise

        except (RAGError, LLMTimeoutError) as exc:
            logger.warning(
                "Primary raw execution failed for request_id=%s: %s",
                request.request_id, exc,
            )
            return await self._handle_fallback(request, exc, query_start)

    _DEFAULT_LOGICAL_COLLECTION = "default"

    async def ingest(
        self,
        file_path: str,
        collection: Optional[str] = None,
        user_id: str = "",
        doc_id: str = "",
        on_batch_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> IngestionResult:
        """Ingest a document into the vector store."""
        self._ensure_initialized()

        logical_collection = (collection or self._DEFAULT_LOGICAL_COLLECTION).strip() \
            or self._DEFAULT_LOGICAL_COLLECTION
        physical_collection = settings.qdrant_collection_name

        logger.info(
            "Pipeline ingesting file='%s' | physical='%s' | logical='%s' | "
            "user_id=%s | doc_id=%s",
            file_path, physical_collection, logical_collection,
            user_id or "<none>", doc_id or "<none>",
        )
        ingest_start = time.perf_counter()

        try:
            cleaner = DocumentCleaner()
            raw_docs = await asyncio.to_thread(cleaner.load_and_clean, file_path)
            logger.info("Loaded %d pages from '%s'", len(raw_docs), file_path)

            preserver = StructurePreserver()
            structured_docs = await asyncio.to_thread(preserver.preserve, raw_docs)

            chunker = Chunker()
            chunks = await asyncio.to_thread(chunker.split_documents, structured_docs)
            total_chunks = len(chunks)
            logger.info("Produced %d chunks", total_chunks)

            for chunk in chunks:
                if user_id:
                    chunk.metadata["user_id"] = user_id
                if doc_id:
                    chunk.metadata["doc_id"] = doc_id
                chunk.metadata["collection"] = logical_collection

            # reuse the boot-time QdrantStore — same collection, same client, same SPLADE.
            # avoids reloading 5s of SPLADE and 5 redundant qdrant roundtrips per job.
            point_ids = await self._store.add_documents(
                chunks, on_batch_progress=on_batch_progress,
            )

            if point_ids:
                try:
                    await self._store.similarity_search_with_vectors("warmup", k=1)
                except Exception:
                    pass

            elapsed = (time.perf_counter() - ingest_start) * 1000
            stored = len(point_ids)
            duplicates = total_chunks - stored

            result = IngestionResult(
                file_path=file_path,
                collection=logical_collection,
                chunks_stored=stored,
                total_chunks=total_chunks,
                duplicates_skipped=max(0, duplicates),
                elapsed_ms=round(elapsed, 1),
            )

            logger.info(
                "Ingestion complete: %d chunks stored in %.1fms",
                stored, elapsed,
            )
            return result

        except Exception as exc:
            raise PipelineIngestionError(
                message=f"Ingestion failed for '{file_path}': {exc}",
                details={
                    "file_path": file_path,
                    "physical_collection": physical_collection,
                    "logical_collection": logical_collection,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ) from exc

    async def list_collections(self) -> list[dict]:
        """Return Qdrant collections enriched with description and document counts."""
        self._ensure_initialized()
        names = await self._store.list_all_collections()

        results: list[dict] = []
        for name in names:
            store = await self._get_store_for_collection(name)
            stats = await store.get_collection_stats()
            results.append({
                "name": name,
                "description": self._collections.get(name, ""),
                "document_count": int(stats.get("document_count") or 0),
            })
        return results

    def _should_use_agent(self, request: RAGRequest) -> bool:
        """Decide whether to route this request through the agent layer."""
        force = request.config.force_agent
        if force is True:
            return self._agent_orchestrator is not None
        if force is False:
            return False
        return self._agent_orchestrator is not None and should_decompose(request.query)

    async def _execute_query(
        self,
        request: RAGRequest,
        llm: BaseLLM,
    ) -> RAGResponse:
        """Build the RAG variant and execute the query."""
        if self._should_use_agent(request):
            logger.info(
                "Routing to agent decomposition | request_id=%s",
                request.request_id,
            )
            # cache check before expensive planner + parallel retrieval + synthesis
            if self._cache:
                cached = await self._try_agent_cache_read(request)
                if cached:
                    return cached

            try:
                agent_response = await self._agent_orchestrator.execute(request)
                rag_response = agent_response.to_rag_response()

                if self._cache:
                    await self._try_agent_cache_write(request, rag_response)

                return rag_response

            except AgentError as exc:
                logger.warning(
                    "Agent path failed | request_id=%s | error=%s | reason=%s | "
                    "falling back to direct RAG",
                    request.request_id,
                    type(exc).__name__,
                    exc,
                )

        rag = await self._build_rag_for_request(request, llm)
        return await rag.query(request)

    @staticmethod
    def _build_scope_hash(request: RAGRequest) -> str:
        """Deterministic hash of all request fields that scope the cached answer."""
        parts = [
            request.logical_collection,
            request.config.domain or "",
            request.config.rag_variant or "",
            request.config.retrieval_mode,
            request.config.rerank_strategy,
            str(request.config.top_k),
        ]
        if request.config.metadata_filters:
            parts.append(hash_text(json.dumps(
                [f.model_dump() for f in request.config.metadata_filters],
                sort_keys=True,
            )))
        return hash_text("|".join(parts))

    async def _try_agent_cache_read(self, request: RAGRequest) -> RAGResponse | None:
        """Attempt to read a cached agent response. Returns None on miss or error."""
        try:
            result = await self._cache.get_or_wait(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=0.0,
                system_prompt="__agent__",
                user_id=request.user_id or "",
                scope_hash=self._build_scope_hash(request),
            )
            if result.hit:
                if result.strategy.value == "semantic":
                    logger.info(
                        "Agent cache hit | request_id=%s | layer=%s | strategy=%s | "
                        "similarity=%.3f | latency=%.1f ms",
                        request.request_id, result.layer, result.strategy,
                        result.similarity_score, result.lookup_latency_ms,
                    )
                else:
                    logger.info(
                        "Agent cache hit | request_id=%s | layer=%s | strategy=%s | "
                        "latency=%.1f ms",
                        request.request_id, result.layer, result.strategy,
                        result.lookup_latency_ms,
                    )
                from rag.models.rag_response import RetrievedChunk
                cached_sources = [RetrievedChunk(**s) for s in result.sources]
                return RAGResponse.from_cache(
                    cached_response=result.response,
                    request_id=request.request_id,
                    rag_variant="agent",
                    cache_layer=result.layer,
                    lookup_latency_ms=result.lookup_latency_ms,
                    sources=cached_sources,
                    confidence_value=result.confidence_value,
                )
        except Exception as exc:
            logger.warning(
                "Agent cache read failed | request_id=%s | error=%s",
                request.request_id, exc,
            )
        return None

    async def _try_agent_cache_write(
        self,
        request: RAGRequest,
        response: RAGResponse,
    ) -> None:
        """Write a synthesized agent response to cache. Errors are caught and logged."""
        # never cache a degraded response — both LLMs failed and answer is a stub
        if response.model_name == "unavailable":
            logger.warning(
                "Agent cache write skipped — degraded response | request_id=%s",
                request.request_id,
            )
            return
        try:
            from llm.models.llm_response import LLMResponse
            stub = LLMResponse(
                text=response.answer,
                model=response.model_name,
                provider=self._llm.provider_name,
                finish_reason="stop",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                tokens_used=response.prompt_tokens + response.completion_tokens,
                latency_ms=response.timings.total_ms,
            )
            await self._cache.set(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=0.0,
                response=stub,
                system_prompt="__agent__",
                sources=[chunk.model_dump() for chunk in response.sources],
                confidence_value=response.confidence.value if response.confidence else 0.0,
                user_id=request.user_id or "",
                scope_hash=self._build_scope_hash(request),
            )
            await self._cache.resolve_in_flight(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=0.0,
                system_prompt="__agent__",
                user_id=request.user_id or "",
                scope_hash=self._build_scope_hash(request),
            )
        except Exception as exc:
            logger.warning(
                "Agent cache write failed | request_id=%s | error=%s",
                request.request_id, exc,
            )

    async def _get_store_for_collection(self, collection_name: str) -> QdrantStore:
        """Return a QdrantStore scoped to the given collection."""
        if collection_name == self._store.collection_name:
            return self._store

        if collection_name not in self._collection_stores:
            store = QdrantStore(
                collection_name=collection_name,
                client=self._store._client,
                search_mode=settings.RAG_RETRIEVAL_MODE,
            )
            await store.initialize()
            self._collection_stores[collection_name] = store

        return self._collection_stores[collection_name]

    async def _build_rag_for_request(
        self,
        request: RAGRequest,
        llm: BaseLLM,
    ) -> BaseRAG:
        """Build the appropriate RAG variant for this request."""
        store = await self._get_store_for_collection(request.collection_name)
        return RAGFactory.create_from_request(
            request=request,
            store=store,
            llm=llm,
            cache=self._cache,
            embeddings_fn=get_embeddings,
        )

    async def _handle_fallback(
        self,
        request: RAGRequest,
        original_error: Exception,
        query_start: float,
    ) -> RAGResponse:
        """Attempt recovery after primary execution fails."""
        variant = self._get_request_variant(request)
        if variant != _FALLBACK_VARIANT:
            logger.info(
                "Fallback: retrying request_id=%s with variant='%s'",
                request.request_id, _FALLBACK_VARIANT,
            )
            try:
                fallback_request = self._downgrade_variant(request)
                rag = await self._build_rag_for_request(fallback_request, self._llm)
                response = await rag.query(fallback_request)
                self._log_query_metrics(request, response, query_start, fallback=True)
                return response
            except Exception as exc:
                logger.warning(
                    "Fallback variant failed for request_id=%s: %s",
                    request.request_id, exc,
                )

        if self._fallback_llm:
            logger.info(
                "Fallback: retrying request_id=%s with fallback LLM",
                request.request_id,
            )
            try:
                fallback_request = self._downgrade_variant(request)
                rag = await self._build_rag_for_request(
                    fallback_request, self._fallback_llm,
                )
                response = await rag.query(fallback_request)
                self._log_query_metrics(request, response, query_start, fallback=True)
                return response
            except Exception as exc:
                logger.warning(
                    "Fallback LLM failed for request_id=%s: %s",
                    request.request_id, exc,
                )

        raise PipelineFallbackExhaustedError(
            message="All fallback strategies exhausted",
            details={
                "request_id": request.request_id,
                "original_error": str(original_error),
                "original_error_type": type(original_error).__name__,
                "attempted_fallbacks": self._describe_fallbacks(variant),
            },
        )

    def _downgrade_variant(self, request: RAGRequest) -> RAGRequest:
        """Create a copy of the request with the variant downgraded to simple."""
        original_config = request.config or RAGConfig()

        downgraded_config = RAGConfig(
            rag_variant=_FALLBACK_VARIANT,
            retrieval_mode=original_config.retrieval_mode,
            top_k=original_config.top_k,
            rerank_strategy=original_config.rerank_strategy,
            max_context_tokens=original_config.max_context_tokens,
            temperature=original_config.temperature,
            system_prompt=original_config.system_prompt,
            metadata_filters=original_config.metadata_filters,
            include_sources=original_config.include_sources,
            confidence_method=original_config.confidence_method,
        )

        return RAGRequest(
            query=request.query,
            collection_name=request.collection_name,
            config=downgraded_config,
            conversation_history=request.conversation_history,
            request_id=request.request_id,
            user_id=request.user_id,
            logical_collection=request.logical_collection,
        )

    def _describe_fallbacks(self, original_variant: str) -> list[str]:
        """Describe which fallback strategies were attempted."""
        attempted = []
        if original_variant != _FALLBACK_VARIANT:
            attempted.append(f"variant_downgrade: {original_variant} -> {_FALLBACK_VARIANT}")
        if self._fallback_llm:
            attempted.append(
                f"llm_fallback: {self._fallback_llm.provider_name}/{self._fallback_llm.model_name}"
            )
        return attempted

    def _ensure_initialized(self) -> None:
        """Guard against using the pipeline before initialization."""
        if not self._initialized:
            raise PipelineValidationError(
                message="Pipeline not initialized. Call await pipeline.initialize() first.",
                details={"state": "not_initialized"},
            )

    def _validate_and_convert(
        self,
        pipeline_query: PipelineQuery,
    ) -> RAGRequest:
        """Validate external query and convert to internal RAGRequest."""
        try:
            return pipeline_query.to_rag_request()
        except Exception as exc:
            raise PipelineValidationError(
                message=f"Invalid query: {exc}",
                details={
                    "query_preview": pipeline_query.query[:100],
                    "collection": pipeline_query.collection,
                    "error": str(exc),
                },
            ) from exc

    def _get_request_variant(self, request: RAGRequest) -> str:
        """Extract the variant name from a request, with default."""
        if request.config and request.config.rag_variant:
            return request.config.rag_variant
        return settings.RAG_DEFAULT_VARIANT

    async def _check_llm_health(self) -> str:
        """Check LLM provider availability."""
        if not self._llm:
            return "not configured"
        try:
            available = await self._llm.is_available()
            return "ok" if available else "unavailable"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_store_health(self) -> str:
        """Check vector store availability."""
        if not self._store:
            return "not configured"
        try:
            await self._store.get_collection_stats()
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_cache_health(self) -> str:
        """Check cache subsystem availability."""
        if not settings.cache_enabled:
            return "disabled"
        if not self._cache:
            return "not configured"
        try:
            metrics = self._cache.get_metrics()
            return "ok" if metrics is not None else "degraded"
        except Exception as exc:
            return f"degraded: {exc}"

    def _try_create_fast_llm(self) -> Optional[BaseLLM]:
        """Attempt to create a fast LLM for weak sub-query rewriting."""
        try:
            fast = LLMFactory.create_rate_limited(
                provider_name=getattr(settings, "LLM_PROVIDER", "groq"),
                model_name=getattr(settings, "GROQ_MODEL_FAST", None),
            )
            logger.info(
                "Fast LLM created for agent rewrites: %s/%s",
                fast.provider_name, fast.model_name,
            )
            return fast
        except Exception:
            logger.warning("Could not create fast LLM — agent rewrites will use strong LLM")
            return None

    def _try_create_fallback_llm(self) -> Optional[BaseLLM]:
        """Attempt to create a fallback LLM provider."""
        primary = self._llm.provider_name if self._llm else ""
        fallback_provider = "openai" if primary == "gemini" else "gemini"

        try:
            fallback = LLMFactory.create_rate_limited(provider_name=fallback_provider)
            logger.info(
                "Fallback LLM created: %s/%s",
                fallback.provider_name, fallback.model_name,
            )
            return fallback
        except Exception:
            logger.warning(
                "Could not create fallback LLM (provider=%s), continuing without",
                fallback_provider,
            )
            return None

    def _log_query_metrics(
        self,
        request: RAGRequest,
        response: RAGResponse,
        query_start: float,
        fallback: bool = False,
    ) -> None:
        """Log query execution metrics."""
        total_ms = (time.perf_counter() - query_start) * 1000

        logger.info(
            "Pipeline query complete: request_id=%s total_ms=%.1f "
            "cache_hit=%s variant=%s confidence=%.3f "
            "prompt_tokens=%s completion_tokens=%s fallback=%s",
            request.request_id,
            total_ms,
            response.cache_hit,
            response.rag_variant,
            response.confidence.value if response.confidence else 0.0,
            response.prompt_tokens,
            response.completion_tokens,
            fallback,
        )
