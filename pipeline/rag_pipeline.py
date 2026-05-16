"""
RAGPipeline — single entry point for the entire RAG system.

Design:
    Facade pattern over all subsystems (LLM, vector store, cache, RAG
    variants, agent layer). Owns lifecycle (init/shutdown), query routing,
    document ingestion, health checks, and fallback logic. Delegates all
    processing to existing layers — the pipeline is glue and lifecycle,
    not business logic.

Chain of Responsibility:
    External callers (FastAPI, CLI, tests) → RAGPipeline.query()
    → if complex: AgentOrchestrator.execute() else: BaseRAG.query()
    RAGPipeline.query_raw() provides raw RAGRequest access for advanced callers.

Dependencies:
    agents.agent_orchestrator, cache.cache_manager, chunking.*,
    llm.llm_factory, rag.rag_factory, vectorstore.qdrant_store
"""

# stdlib
import asyncio
import time
from typing import Optional

# internal
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
from utils.logger import get_logger

logger = get_logger(__name__)

# Fallback variant used when the requested variant fails.
_FALLBACK_VARIANT = "simple"


class RAGPipeline:
    """Single entry point for the entire RAG system.

    Manages lifecycle of all subsystems, routes queries to the
    appropriate RAG variant, handles fallbacks on failure, and
    provides document ingestion and health check capabilities.

    Attributes:
        _initialized: Whether initialize() has been called successfully.
        _llm: Primary LLM provider instance.
        _fallback_llm: Optional secondary LLM for failover.
        _store: QdrantStore instance.
        _cache: Optional CacheManager instance.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        fallback_llm: Optional[BaseLLM] = None,
        store: Optional[QdrantStore] = None,
        cache: Optional[CacheManager] = None,
    ) -> None:
        """Initialize RAGPipeline.

        Dependencies can be injected for testing. When None, the
        pipeline creates them from settings during initialize().

        Args:
            llm: Optional primary LLM provider. Built from settings if None.
            fallback_llm: Optional fallback LLM provider.
            store: Optional QdrantStore. Built from settings if None.
            cache: Optional CacheManager. Built from settings if None.
        """
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._store = store
        self._cache = cache
        self._initialized = False
        self._agent_orchestrator: Optional[AgentOrchestrator] = None
        self._collections: dict[str, str] = {}
        # Cache of QdrantStore instances keyed by collection name.
        # All share self._store._client so they query the same Qdrant database.
        self._collection_stores: dict[str, QdrantStore] = {}

    # Lifecycle

    async def initialize(self) -> None:
        """Boot all subsystems in dependency order.

        Order: LLM providers → Vector store → Cache.
        LLM first because semantic cache needs the embedding model,
        and health checks need the LLM available.

        Raises:
            PipelineInitError: If any critical subsystem fails to start.
        """
        if self._initialized:
            logger.info("Pipeline already initialized, skipping")
            return

        logger.info("Pipeline initializing subsystems")
        init_start = time.perf_counter()

        try:
            # LLM providers — LLMFactory.create_from_settings() wraps the
            # provider in LLMRateLimiter automatically; no manual wrapping needed.
            if self._llm is None:
                self._llm = LLMFactory.create_from_settings()
                logger.info(
                    "Primary LLM created: %s/%s",
                    self._llm.provider_name, self._llm.model_name,
                )

            if self._fallback_llm is None:
                self._fallback_llm = self._try_create_fallback_llm()

            # Vector store
            if self._store is None:
                self._store = QdrantStore(
                    in_memory=settings.debug,
                    search_mode=settings.RAG_RETRIEVAL_MODE,
                )
            await self._store.initialize()
            logger.info("Vector store initialized")

            # Cache is non-critical — failure degrades gracefully, not crashes.
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

        # Warm-up forces all heavy models to load now so the first real
        # user query pays zero cold-start penalty.
        await self._run_warmup()

        elapsed = (time.perf_counter() - init_start) * 1000
        self._initialized = True
        logger.info("Pipeline initialized in %.1fms", elapsed)

    async def shutdown(self) -> None:
        """Tear down all subsystems in reverse dependency order.

        Order: Cache → Vector store. LLM providers have no cleanup.
        Safe to call even if initialize() was never called or failed.
        """
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
        """Force all heavy models to load before the first real query.

        Runs four warm-up tasks in parallel so the total boot cost is
        max(all four), not sum(all four). Each task is silenced with
        return_exceptions=True — a warm-up failure never crashes the
        pipeline; it just logs a warning and the first real query pays
        that cold-start penalty instead.

        What each task fixes:
            embedding  — BGE ONNX model load + JIT kernel compile (3-8s)
            splade     — SPLADE sparse model load via FastEmbed (1-3s)
            qdrant     — HNSW graph segments loaded into RAM (200-500ms)
            llm        — TCP + TLS handshake to provider API (500-1500ms)
        """
        warmup_start = time.perf_counter()
        logger.info("Pipeline warm-up starting...")

        async def _warmup_embeddings() -> None:
            model = get_embeddings()
            await asyncio.to_thread(model.embed_query, "warmup")
            logger.debug("Warm-up: embedding model ready")

        async def _warmup_splade() -> None:
            if self._store and hasattr(self._store, "_get_sparse_embeddings"):
                sparse = self._store._get_sparse_embeddings()
                await asyncio.to_thread(sparse.embed_query, "warmup")
                logger.debug("Warm-up: SPLADE model ready")

        async def _warmup_qdrant() -> None:
            if self._store:
                try:
                    await self._store.similarity_search_with_vectors("warmup", k=1)
                    logger.debug("Warm-up: Qdrant HNSW index ready")
                except Exception:
                    pass  # empty collection is fine — HNSW still loads

        async def _warmup_llm() -> None:
            if self._llm:
                try:
                    await self._llm.generate("Reply with: OK", max_tokens=2)
                    logger.debug("Warm-up: LLM connection pool ready")
                except Exception:
                    pass  # non-critical — connection opens on first real query

        results = await asyncio.gather(
            _warmup_embeddings(),
            _warmup_splade(),
            _warmup_qdrant(),
            _warmup_llm(),
            return_exceptions=True,
        )

        warmup_names = ["embedding", "splade", "qdrant", "llm"]
        for name, result in zip(warmup_names, results):
            if isinstance(result, Exception):
                logger.warning("Warm-up failed for %s: %s", name, result)

        elapsed = (time.perf_counter() - warmup_start) * 1000
        logger.info("Pipeline warm-up complete in %.1fms", elapsed)

    async def health_check(self) -> PipelineHealthStatus:
        """Check health of all subsystems.

        Returns:
            PipelineHealthStatus with per-subsystem status.
        """
        llm_status = await self._check_llm_health()
        store_status = await self._check_store_health()
        cache_status = await self._check_cache_health()

        # Ready only if LLM and vector store are both healthy.
        # Cache is non-critical — degraded state is acceptable.
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
        """Configure the agent layer for query decomposition.

        Must be called after initialize(). Creates AgentOrchestrator with
        a strong LLM (planning + synthesis) and a fast LLM (weak sub-query
        rewriting). The store_factory provides per-collection QdrantStore
        instances for parallel retrieval.

        Args:
            collections: Dict of collection_name -> description.
                Used by the planner to route sub-queries.
            max_concurrent: Max concurrent sub-query retrieval executions.
        """
        self._ensure_initialized()
        self._collections = collections

        # Fast LLM for weak sub-query rewriting — cheaper model, bounded use.
        # Falls back silently to strong LLM if creation fails.
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

    # Query — the main entry point

    async def query(
        self,
        pipeline_query: PipelineQuery,
    ) -> RAGResponse:
        """Execute a query through the full RAG pipeline.

        Validates input, selects the appropriate RAG variant,
        executes, and falls back on failure.

        Args:
            pipeline_query: Simplified external query model.

        Returns:
            RAGResponse with answer, sources, timings, confidence.

        Raises:
            PipelineValidationError: If input validation fails.
            PipelineFallbackExhaustedError: If all strategies fail.
            LLMAuthError: If LLM authentication fails (propagated as-is).
            LLMRateLimitError: If LLM rate limit hit (propagated as-is).
        """
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
            # Auth and rate-limit errors are not retryable via fallback;
            # the caller must handle these (fix credentials, back off, etc.).
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
        """Execute a query using a raw RAGRequest (advanced usage).

        Bypasses PipelineQuery validation — the caller is responsible
        for constructing a valid RAGRequest. Useful for internal
        callers (agents, tests) that need full RAGRequest control.

        Args:
            request: Fully constructed RAGRequest.

        Returns:
            RAGResponse with answer, sources, timings, confidence.

        Raises:
            PipelineValidationError: If pipeline is not initialized.
            PipelineFallbackExhaustedError: If all strategies fail.
        """
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

    # Document ingestion

    async def ingest(
        self,
        file_path: str,
        collection: str,
        user_id: str = "",
        doc_id: str = "",
    ) -> IngestionResult:
        """Ingest a document into the vector store.

        Runs the full ingestion pipeline: load → clean → preserve
        structure → chunk → embed → store.

        Args:
            file_path: Path to the document file.
            collection: Target Qdrant collection name.
            user_id: Owning user ID written to every chunk's payload.
                Empty string means no per-user scoping.
            doc_id: Stable identifier for the source document (e.g. a UUID
                generated by the backend). Written to every chunk's payload
                so the document can be deleted/updated as a unit later.

        Returns:
            IngestionResult with chunk counts and timing.

        Raises:
            PipelineValidationError: If pipeline is not initialized.
            PipelineIngestionError: If any ingestion step fails.
        """
        self._ensure_initialized()

        logger.info(
            "Pipeline ingesting file='%s' into collection='%s' | user_id=%s | doc_id=%s",
            file_path, collection, user_id or "<none>", doc_id or "<none>",
        )
        ingest_start = time.perf_counter()

        try:
            # Load and clean — DocumentCleaner.load_and_clean handles both steps.
            cleaner = DocumentCleaner()
            raw_docs = await asyncio.to_thread(cleaner.load_and_clean, file_path)
            logger.info("Loaded %d pages from '%s'", len(raw_docs), file_path)

            # Tag headings, tables, and other structural elements.
            preserver = StructurePreserver()
            structured_docs = await asyncio.to_thread(preserver.preserve, raw_docs)

            # Chunk into retrieval-sized pieces.
            chunker = Chunker()
            chunks = await asyncio.to_thread(chunker.split_documents, structured_docs)
            total_chunks = len(chunks)
            logger.info("Produced %d chunks", total_chunks)

            # Stamp tenancy metadata on every chunk so QdrantStore's setdefault
            # preserves these (instead of writing "" placeholders).
            if user_id or doc_id:
                for chunk in chunks:
                    if user_id:
                        chunk.metadata["user_id"] = user_id
                    if doc_id:
                        chunk.metadata["doc_id"] = doc_id

            # Reuse the pipeline's existing Qdrant client so the ingested
            # collection is visible to all subsequent queries. Creating a new
            # QdrantStore without sharing the client would produce an isolated
            # in-memory database unreachable by queries.
            ingest_store = QdrantStore(
                collection_name=collection,
                client=self._store._client,
                search_mode=settings.RAG_RETRIEVAL_MODE,
            )
            await ingest_store.initialize()
            point_ids = await ingest_store.add_documents(chunks)

            # Post-ingest HNSW warmup — forces Qdrant to load the freshly
            # built HNSW graph segments into RAM right now, so the first
            # real query against this collection pays zero cold-start penalty.
            # BGE and SPLADE are already hot (used above to encode chunks),
            # so this one dummy search costs only the Qdrant RTT (~110ms).
            if point_ids:  # only worth warming if new data was actually stored
                try:
                    await ingest_store.similarity_search_with_vectors("warmup", k=1)
                    logger.debug(
                        "Post-ingest HNSW warmup complete for collection='%s'", collection,
                    )
                except Exception:
                    pass  # non-fatal — first query pays cold-start instead

            elapsed = (time.perf_counter() - ingest_start) * 1000
            stored = len(point_ids)
            duplicates = total_chunks - stored

            result = IngestionResult(
                file_path=file_path,
                collection=collection,
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
                    "collection": collection,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ) from exc

    async def list_collections(self) -> list[dict]:
        """Return Qdrant collections enriched with description (from the agent
        registry) and per-collection document counts."""
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

    # Internal — query execution and routing

    def _should_use_agent(self, request: RAGRequest) -> bool:
        """Decide whether to route this request through the agent layer.

        Respects force_agent override in RAGConfig:
            True  → always use agent (if configured)
            False → never use agent
            None  → auto-detect via should_decompose()
        """
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
        """Build the RAG variant and execute the query.

        Routes complex queries through the agent layer (with cache check).
        Simple queries go directly to a RAG variant.

        Args:
            request: Validated RAGRequest.
            llm: LLM provider to use for this execution.

        Returns:
            RAGResponse from the selected execution path.
        """
        if self._should_use_agent(request):
            logger.info(
                "Routing to agent decomposition | request_id=%s",
                request.request_id,
            )
            # Cache check before the expensive planner + parallel retrieval + synthesis.
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
                # Agent path failed (all sub-queries failed retrieval or verification,
                # or synthesis produced no output). Fall through to direct RAG so the
                # user always receives an answer rather than a hard error.
                logger.warning(
                    "Agent path failed | request_id=%s | error=%s | reason=%s | "
                    "falling back to direct RAG",
                    request.request_id,
                    type(exc).__name__,
                    exc,
                )

        # Direct RAG path — used for simple queries and as agent fallback.
        rag = await self._build_rag_for_request(request, llm)
        return await rag.query(request)

    async def _try_agent_cache_read(self, request: RAGRequest) -> RAGResponse | None:
        """Attempt to read a cached agent response. Returns None on miss or error.

        Uses "__agent__" as the system_prompt discriminator so agent cache
        keys never collide with SimpleRAG keys for the same query.
        """
        try:
            result = await self._cache.get_or_wait(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=0.0,          # synthesis temperature
                system_prompt="__agent__",
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
        # Never cache a degraded response — both LLMs failed, answer is a stub.
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
            )
            await self._cache.resolve_in_flight(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=0.0,
                system_prompt="__agent__",
            )
        except Exception as exc:
            logger.warning(
                "Agent cache write failed | request_id=%s | error=%s",
                request.request_id, exc,
            )

    async def _get_store_for_collection(self, collection_name: str) -> QdrantStore:
        """Return a QdrantStore scoped to the given collection.

        Returns self._store directly for its own collection. Otherwise
        creates (once) and caches a new QdrantStore that shares the same
        Qdrant client — same pattern as ingest(). All collections live in
        one database; each store just targets a different collection name.

        Args:
            collection_name: Target Qdrant collection name.

        Returns:
            Initialized QdrantStore targeting collection_name.
        """
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
        """Build the appropriate RAG variant for this request.

        Resolves the correct QdrantStore for the request's collection,
        then delegates to RAGFactory.create_from_request().

        Args:
            request: RAGRequest with config specifying the variant.
            llm: LLM provider to inject into the RAG variant.

        Returns:
            Configured BaseRAG subclass instance.
        """
        store = await self._get_store_for_collection(request.collection_name)
        return RAGFactory.create_from_request(
            request=request,
            store=store,
            llm=llm,
            cache=self._cache,
            embeddings_fn=get_embeddings,
        )

    # Internal — fallback logic

    async def _handle_fallback(
        self,
        request: RAGRequest,
        original_error: Exception,
        query_start: float,
    ) -> RAGResponse:
        """Attempt recovery after primary execution fails.

        Fallback strategy (in order):
        1. Retry with SimpleRAG if a complex variant failed.
        2. Retry with fallback LLM if primary LLM timed out.
        3. Give up — raise PipelineFallbackExhaustedError.

        Args:
            request: The original RAGRequest.
            original_error: The exception that triggered fallback.
            query_start: perf_counter timestamp for total latency.

        Returns:
            RAGResponse from a fallback strategy.

        Raises:
            PipelineFallbackExhaustedError: If no fallback succeeds.
        """
        # Strategy 1 — retry with simpler variant using the same LLM.
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

        # Strategy 2 — retry with fallback LLM using the simpler variant.
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

        # All fallbacks exhausted — surface to caller.
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
        """Create a copy of the request with the variant downgraded to simple.

        Preserves all other config fields. Builds a new RAGRequest
        since the model may be frozen.

        Args:
            request: Original RAGRequest.

        Returns:
            New RAGRequest with variant set to _FALLBACK_VARIANT.
        """
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
        )

    def _describe_fallbacks(self, original_variant: str) -> list[str]:
        """Describe which fallback strategies were attempted.

        Args:
            original_variant: The originally requested variant.

        Returns:
            List of fallback descriptions for logging.
        """
        attempted = []
        if original_variant != _FALLBACK_VARIANT:
            attempted.append(f"variant_downgrade: {original_variant} -> {_FALLBACK_VARIANT}")
        if self._fallback_llm:
            attempted.append(
                f"llm_fallback: {self._fallback_llm.provider_name}/{self._fallback_llm.model_name}"
            )
        return attempted

    # Internal — validation

    def _ensure_initialized(self) -> None:
        """Guard against using the pipeline before initialization.

        Raises:
            PipelineValidationError: If pipeline is not initialized.
        """
        if not self._initialized:
            raise PipelineValidationError(
                message="Pipeline not initialized. Call await pipeline.initialize() first.",
                details={"state": "not_initialized"},
            )

    def _validate_and_convert(
        self,
        pipeline_query: PipelineQuery,
    ) -> RAGRequest:
        """Validate external query and convert to internal RAGRequest.

        Args:
            pipeline_query: External PipelineQuery model.

        Returns:
            Validated RAGRequest.

        Raises:
            PipelineValidationError: If validation fails.
        """
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
        """Extract the variant name from a request, with default.

        Args:
            request: RAGRequest to inspect.

        Returns:
            Variant name string.
        """
        if request.config and request.config.rag_variant:
            return request.config.rag_variant
        return settings.RAG_DEFAULT_VARIANT

    # Internal — health checks

    async def _check_llm_health(self) -> str:
        """Check LLM provider availability.

        Returns:
            "ok" if available, error description otherwise.
        """
        if not self._llm:
            return "not configured"
        try:
            available = await self._llm.is_available()
            return "ok" if available else "unavailable"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_store_health(self) -> str:
        """Check vector store availability.

        Returns:
            "ok" if accessible, error description otherwise.
        """
        if not self._store:
            return "not configured"
        try:
            # get_collection_stats is a lightweight read-only operation.
            await self._store.get_collection_stats()
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_cache_health(self) -> str:
        """Check cache subsystem availability.

        Cache is non-critical — "degraded" is an acceptable state.

        Returns:
            "ok", "disabled", "degraded", or error description.
        """
        if not settings.cache_enabled:
            return "disabled"
        if not self._cache:
            return "not configured"
        try:
            metrics = self._cache.get_metrics()
            return "ok" if metrics is not None else "degraded"
        except Exception as exc:
            return f"degraded: {exc}"

    # Internal — LLM factory helpers

    def _try_create_fast_llm(self) -> Optional[BaseLLM]:
        """Attempt to create a fast LLM for weak sub-query rewriting.

        Uses GROQ_MODEL_FAST from settings when available. Non-fatal —
        configure_agents() falls back to the strong LLM if this fails.

        Returns:
            Fast BaseLLM instance, or None if creation fails.
        """
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
        """Attempt to create a fallback LLM provider.

        Tries the secondary provider (if primary is gemini → openai,
        if primary is openai → gemini). Failure is non-fatal — the
        pipeline works without a fallback, just loses one retry path.

        Returns:
            Fallback BaseLLM instance, or None if creation fails.
        """
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

    # Internal — metrics and logging

    def _log_query_metrics(
        self,
        request: RAGRequest,
        response: RAGResponse,
        query_start: float,
        fallback: bool = False,
    ) -> None:
        """Log query execution metrics.

        Args:
            request: The executed RAGRequest.
            response: The RAGResponse produced.
            query_start: perf_counter timestamp from query start.
            fallback: Whether this response came from a fallback path.
        """
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
