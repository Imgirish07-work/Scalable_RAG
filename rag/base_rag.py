"""
BaseRAG — Template Method pattern for the RAG pipeline.

Design:
    - query() is the SEALED algorithm skeleton. No variant can reorder,
      skip, or add steps. They can only override individual hooks.
    - Cache integration lives HERE, not in variants. Every variant
      inherits caching for free. Variants own retrieval logic, not
      orchestration concerns.
    - All dependencies are injected via constructor: retriever, LLM,
      cache, ranker, assembler. Fully testable with mocks.
    - Timing instrumentation is built into query() — every step is
      timed individually for RAGTimings diagnostics.

Sealed pipeline (query):
    1. Cache check (get_or_wait)
    2. pre_process()     — hook: normalize query, resolve pronouns
    3. retrieve()        — hook: ABSTRACT, every variant implements
    4. rank()            — hook: MMR diversification (default)
    5. assemble_context() — hook: token-bounded assembly (default)
    6. generate()        — hook: LLM call with context (default)
    7. build_response()  — sealed: construct RAGResponse
    8. Cache write (set + resolve_in_flight)

Overridable hooks:
    - pre_process():     QueryExpansionRAG overrides (HyDE, future)
    - retrieve():        ALL variants override (abstract)
    - rank():            CorrectiveRAG may override (adds eval step)
    - generate():        MultiAgentRAG would override (future)
    - assemble_context(): Rarely overridden

Integration:
    - CacheManager from cache/cache_manager.py (optional)
    - BaseLLM from llm/contracts/base_llm.py
    - BaseRetriever from rag/retrieval/base_retriever.py
    - ContextRanker from rag/context/context_ranker.py
    - ContextAssembler from rag/context/context_assembler.py
    - Prompt templates from rag/prompts/rag_prompt_templates.py
"""

import time
from abc import ABC, abstractmethod

from llm.contracts.base_llm import BaseLLM
from llm.models.llm_response import LLMResponse
from rag.models.rag_request import RAGRequest
from rag.models.rag_response import (
    RAGResponse,
    RetrievedChunk,
    ConfidenceScore,
    RAGTimings,
)
from rag.retrieval.base_retriever import BaseRetriever
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from vectorstore.embeddings import get_embeddings
from vectorstore.reranker import get_reranker
from rag.prompts.rag_prompt_templates import (
    build_rag_prompt,
    build_conversation_refinement_prompt,
    format_conversation_history,
)
from rag.exceptions.rag_exceptions import (
    RAGGenerationError,
    RAGContextError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseRAG(ABC):
    """Abstract base class for all RAG variants.

    Implements the Template Method pattern: query() is sealed and defines
    the algorithm skeleton. Variants override individual hooks to customize
    behavior without changing the pipeline flow.

    Subclasses MUST implement:
        - retrieve(query, top_k, filters) → list[RetrievedChunk]
        - variant_name (property) → str

    Subclasses MAY override:
        - pre_process(request) → str
        - rank(chunks, query) → list[RetrievedChunk]
        - assemble_context(chunks) → tuple[str, list[RetrievedChunk], int]
        - generate(context, query, request) → LLMResponse

    Attributes:
        _retriever: BaseRetriever for vector store access.
        _llm: BaseLLM for text generation and token counting.
        _cache: Optional CacheManager for response caching.
        _ranker: ContextRanker for post-retrieval reranking.
        _assembler: ContextAssembler for token-bounded context building.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache: object | None = None,
        ranker: ContextRanker | None = None,
        assembler: ContextAssembler | None = None,
    ) -> None:
        """Initialize BaseRAG with all dependencies.

        Args:
            retriever: Vector store retriever (dense or hybrid).
            llm: LLM provider for generation and token counting.
            cache: Optional CacheManager. If None, caching is skipped.
                Typed as object to avoid circular imports.
            ranker: Optional ContextRanker. If None, a default MMR
                ranker is created.
            assembler: Optional ContextAssembler. If None, a default
                assembler is created using the provided LLM for
                token counting.
        """
        self._retriever = retriever
        self._llm = llm
        self._cache = cache
        # Default ranker: inject reranker if available so per-request cross_encoder works
        self._ranker = ranker or ContextRanker(
            strategy="mmr",
            embeddings_fn=get_embeddings,
            reranker=get_reranker(),
            top_k=5,
        )
        self._assembler = assembler or ContextAssembler(llm=llm)

        logger.info(
            "BaseRAG initialized | variant=%s | retriever=%s | "
            "llm=%s | cache=%s",
            self.variant_name,
            self._retriever.retriever_type,
            self._llm.provider_name,
            "enabled" if self._cache else "disabled",
        )

    # ================================================================
    # Abstract — subclasses MUST implement
    # ================================================================

    @property
    @abstractmethod
    def variant_name(self) -> str:
        """Return the variant identifier string.

        Returns:
            Variant name e.g. 'simple', 'corrective'.
        """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from the vector store.

        This is the primary hook that every variant implements.
        SimpleRAG calls the retriever directly. CorrectiveRAG adds
        relevance evaluation and retry logic.

        Args:
            query: Processed query string (output of pre_process).
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters from RAGConfig.

        Returns:
            List of RetrievedChunk ordered by relevance.

        Raises:
            RAGRetrievalError: If retrieval fails.
        """

    # ================================================================
    # Sealed pipeline — query() orchestrates everything
    # ================================================================

    async def query(self, request: RAGRequest) -> RAGResponse:
        """Execute the full RAG pipeline.

        This method is SEALED — variants do not override it. The pipeline
        flow is fixed: cache → pre_process → retrieve → rank → assemble →
        generate → cache_write → return.

        Variants customize behavior by overriding individual hooks
        (retrieve, pre_process, rank, generate).

        Args:
            request: RAGRequest with query, collection, config, and
                optional conversation history.

        Returns:
            RAGResponse with answer, sources, timings, and diagnostics.

        Raises:
            RAGRetrievalError: If retrieval fails.
            RAGContextError: If context assembly fails.
            RAGGenerationError: If LLM generation fails.
        """
        total_start = time.perf_counter()
        config = request.config

        logger.info(
            "RAG query started | variant=%s | request_id=%s | "
            "query_len=%d | collection=%s",
            self.variant_name,
            request.request_id,
            len(request.query),
            request.collection_name,
        )

        # ---- Step 1: Cache check ----
        if self._cache:
            cache_result = await self._try_cache_read(request)
            if cache_result is not None:
                return cache_result

        # ---- Step 2: Pre-process ----
        processed_query = await self.pre_process(request)

        # ---- Step 3: Retrieve ----
        # For cross_encoder: fetch RERANKER_COARSE_TOP_K candidates (e.g. 10) so
        # the reranker has enough to score; otherwise fetch config.top_k directly.
        active_strategy = config.rerank_strategy
        if active_strategy == "cross_encoder" and self._ranker._reranker is not None:
            from config.settings import settings as _s
            retrieval_k = getattr(_s, "RERANKER_COARSE_TOP_K", config.top_k * 2)
        else:
            retrieval_k = config.top_k

        retrieval_start = time.perf_counter()
        chunks = await self.retrieve(
            query=processed_query,
            top_k=retrieval_k,
            filters=config.metadata_filters,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # ---- Step 4: Rank ----
        ranking_start = time.perf_counter()
        ranked_chunks = await self.rank(chunks, processed_query, strategy=active_strategy)
        ranking_ms = (time.perf_counter() - ranking_start) * 1000

        # ---- Step 4b: Reranker threshold guard ----
        # If all cross-encoder scores are near-zero, the reranker found nothing
        # relevant. Assembling context from irrelevant chunks causes hallucination.
        # Return a transparent "no context" response instead.
        if ranked_chunks:
            reranker_scores = [
                c.reranker_score for c in ranked_chunks if c.reranker_score is not None
            ]
            if reranker_scores:
                from config.settings import settings as _s
                threshold = getattr(_s, "RERANKER_SCORE_THRESHOLD", 0.1)
                top_reranker_score = max(reranker_scores)
                if top_reranker_score < threshold:
                    total_ms = (time.perf_counter() - total_start) * 1000
                    logger.warning(
                        "Reranker threshold not met | top_score=%.4f | "
                        "threshold=%.2f | returning low-confidence response",
                        top_reranker_score,
                        threshold,
                    )
                    # Unblock any coalesced requests waiting on this key.
                    # The normal path calls resolve_in_flight inside _try_cache_write.
                    # The early-return path skips that, leaving the event unset and
                    # causing the next caller to wait the full 10s coalescing timeout.
                    if self._cache:
                        try:
                            await self._cache.resolve_in_flight(
                                query=request.query,
                                model_name=self._llm.model_name,
                                temperature=request.config.temperature,
                                system_prompt=request.config.system_prompt or "",
                            )
                        except Exception:
                            pass

                    no_context_answer = (
                        "I couldn't find sufficiently relevant information in the "
                        "provided documents to answer this question confidently. "
                        "Please try rephrasing your query or check that the relevant "
                        "document has been indexed."
                    )
                    from llm.models.llm_response import LLMResponse as _LLMResponse
                    stub_llm_response = _LLMResponse(
                        text=no_context_answer,
                        model=self._llm.model_name,
                        provider=self._llm.provider_name,
                        finish_reason="stop",
                        prompt_tokens=0,
                        completion_tokens=len(no_context_answer.split()),
                        tokens_used=len(no_context_answer.split()),
                        latency_ms=0.0,
                    )
                    return RAGResponse.from_generation(
                        answer=no_context_answer,
                        llm_response=stub_llm_response,
                        sources=[],
                        timings=RAGTimings(
                            retrieval_ms=round(retrieval_ms, 2),
                            ranking_ms=round(ranking_ms, 2),
                            total_ms=round(total_ms, 2),
                        ),
                        confidence=ConfidenceScore(value=top_reranker_score, method="reranker"),
                        request_id=request.request_id,
                        rag_variant=self.variant_name,
                        low_confidence=True,
                    )

        # ---- Step 5: Assemble context ----
        context_str, updated_chunks, context_tokens = await self.assemble_context(
            ranked_chunks
        )

        # ---- Step 6: Generate ----
        generation_start = time.perf_counter()
        llm_response = await self.generate(context_str, processed_query, request)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        # ---- Step 7: Build response ----
        total_ms = (time.perf_counter() - total_start) * 1000

        timings = RAGTimings(
            retrieval_ms=round(retrieval_ms, 2),
            ranking_ms=round(ranking_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
        )

        confidence = self._compute_confidence(
            chunks=updated_chunks,
            method=config.confidence_method,
        )

        # Filter sources if include_sources is False
        sources = updated_chunks if config.include_sources else []

        rag_response = RAGResponse.from_generation(
            answer=llm_response.text,
            llm_response=llm_response,
            sources=sources,
            timings=timings,
            confidence=confidence,
            request_id=request.request_id,
            rag_variant=self.variant_name,
            context_tokens_used=context_tokens,
            low_confidence=self._get_low_confidence_flag(),
        )

        # ---- Step 8: Cache write ----
        if self._cache:
            await self._try_cache_write(request, llm_response, sources, confidence)

        logger.info(
            "RAG query complete | variant=%s | request_id=%s | "
            "sources=%d | confidence=%.2f | tokens=%d | total_ms=%.1f",
            self.variant_name,
            request.request_id,
            len(sources),
            confidence.value,
            llm_response.tokens_used,
            total_ms,
        )

        return rag_response

    # ================================================================
    # Overridable hooks — sensible defaults, variants customize
    # ================================================================

    async def pre_process(self, request: RAGRequest) -> str:
        """Pre-process the query before retrieval.

        Default behavior:
            - If conversation_history exists, use LLM to resolve pronouns
              and make the query self-contained.
            - Otherwise, return the query as-is (already stripped by
              RAGRequest validator).

        QueryExpansionRAG (future) would override this to generate
        a hypothetical answer for HyDE embedding.

        Args:
            request: Full RAGRequest with query and optional history.

        Returns:
            Processed query string ready for retrieval.
        """
        # If no conversation history, return query directly
        chat_messages = request.get_chat_messages()
        if not chat_messages:
            return request.query

        # Conversation-aware query refinement
        history_str = format_conversation_history(chat_messages)
        system_prompt, user_prompt = build_conversation_refinement_prompt(
            query=request.query,
            conversation_history=history_str,
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._llm.chat(
                messages,
                temperature=0.0,
                max_tokens=200,
            )

            refined = response.text.strip()
            if refined:
                logger.info(
                    "Query refined via conversation context | "
                    "original=%s | refined=%s",
                    request.query[:80],
                    refined[:80],
                )
                return refined

        except Exception as exc:
            logger.warning(
                "Query refinement failed, using original query | error=%s",
                str(exc),
            )

        return request.query

    async def rank(
        self,
        chunks: list[RetrievedChunk],
        query: str,
        strategy: str | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks.

        Default behavior: delegate to the injected ContextRanker.

        CorrectiveRAG may override this to add relevance evaluation
        before or after reranking.

        Args:
            chunks:   Retrieved chunks from retrieve().
            query:    Processed query string.
            strategy: Per-request strategy override (e.g. 'cross_encoder').
                      Passed through to ContextRanker.rank().

        Returns:
            Reranked list of RetrievedChunk.
        """
        return await self._ranker.rank(chunks, query, strategy=strategy)

    async def assemble_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[RetrievedChunk], int]:
        """Assemble ranked chunks into a token-bounded context string.

        Default behavior: delegate to the injected ContextAssembler.

        Rarely overridden — the assembler handles token budgeting,
        source labeling, and used_in_context flagging.

        Args:
            chunks: Ranked chunks from rank().

        Returns:
            Tuple of (context_string, updated_chunks, tokens_used).

        Raises:
            RAGContextError: If no chunks fit the token budget.
        """
        return await self._assembler.assemble(chunks)

    async def generate(
        self,
        context: str,
        query: str,
        request: RAGRequest,
    ) -> LLMResponse:
        """Generate an answer using the LLM with assembled context.

        Default behavior:
            - Build system + user prompts from templates.
            - If custom system_prompt is set in RAGConfig, use that instead.
            - If conversation_history exists, include it in the prompt.
            - Call LLM via chat() with system + user messages.

        MultiAgentRAG (future) would override this to synthesize
        across sub-query answers.

        Args:
            context: Assembled context string from assemble_context().
            query: Processed query string.
            request: Full RAGRequest for conversation history and config.

        Returns:
            LLMResponse from the LLM provider.

        Raises:
            RAGGenerationError: If LLM returns empty or unusable output.
        """
        # Build conversation history string if available
        chat_messages = request.get_chat_messages()
        history_str = (
            format_conversation_history(chat_messages)
            if chat_messages
            else None
        )

        # Build prompt pair
        system_prompt, user_prompt = build_rag_prompt(
            query=query,
            context=context,
            conversation_history=history_str,
        )

        # Allow system prompt override from config
        if request.config.system_prompt:
            system_prompt = request.config.system_prompt

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._llm.chat(
                messages,
                temperature=request.config.temperature,
            )

            if not response.text or not response.text.strip():
                raise RAGGenerationError(
                    "LLM returned empty response for RAG query.",
                    details={
                        "request_id": request.request_id,
                        "model": self._llm.model_name,
                        "query_length": len(query),
                        "context_length": len(context),
                    },
                )

            return response

        except RAGGenerationError:
            raise

        except Exception as exc:
            # Don't wrap LLM-layer errors (LLMAuthError, etc.) — let them
            # propagate as-is for provider-specific handling
            from llm.exceptions.llm_exceptions import LLMError
            if isinstance(exc, LLMError):
                raise

            raise RAGGenerationError(
                f"RAG generation failed: {exc}",
                details={
                    "request_id": request.request_id,
                    "model": self._llm.model_name,
                },
            ) from exc

    # ================================================================
    # Private helpers
    # ================================================================

    async def _try_cache_read(self, request: RAGRequest) -> RAGResponse | None:
        """Attempt to read from cache. Returns None on miss or error.

        Cache errors are caught and logged — they never propagate to
        the caller. A cache failure means we just run the full pipeline.

        Args:
            request: RAGRequest for cache key generation.

        Returns:
            RAGResponse if cache hit, None otherwise.
        """
        try:
            result = await self._cache.get_or_wait(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                system_prompt=request.config.system_prompt or "",
            )

            if result.hit:
                logger.info(
                    "Cache hit | request_id=%s | layer=%s | "
                    "similarity=%.3f | latency=%.1f ms",
                    request.request_id,
                    result.layer,
                    result.similarity_score or 0.0,
                    result.lookup_latency_ms,
                )

                cached_sources = [RetrievedChunk(**s) for s in result.sources]
                return RAGResponse.from_cache(
                    cached_response=result.response,
                    request_id=request.request_id,
                    rag_variant=self.variant_name,
                    cache_layer=result.layer,
                    lookup_latency_ms=result.lookup_latency_ms,
                    sources=cached_sources,
                    confidence_value=result.confidence_value,
                )

        except Exception as exc:
            logger.warning(
                "Cache read failed, proceeding without cache | "
                "request_id=%s | error=%s",
                request.request_id,
                str(exc),
            )

        return None

    async def _try_cache_write(
        self,
        request: RAGRequest,
        llm_response: LLMResponse,
        sources: list[RetrievedChunk] = [],
        confidence: ConfidenceScore | None = None,
    ) -> None:
        """Attempt to write to cache. Errors are caught and logged.

        Cache write happens AFTER the response is built and returned
        conceptually — in practice it's the last step before return
        but failures don't affect the response.

        Args:
            request: RAGRequest for cache key generation.
            llm_response: LLMResponse to cache.
            sources: Retrieved chunks to store alongside the response.
        """
        try:
            await self._cache.set(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                response=llm_response,
                system_prompt=request.config.system_prompt or "",
                sources=[chunk.model_dump() for chunk in sources],
                confidence_value=confidence.value if confidence is not None else 0.0,
            )
            await self._cache.resolve_in_flight(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                system_prompt=request.config.system_prompt or "",
            )
        except Exception as exc:
            logger.warning(
                "Cache write failed | request_id=%s | error=%s",
                request.request_id,
                str(exc),
            )

    def _get_low_confidence_flag(self) -> bool:
        """Return whether the current query has low confidence.

        Default: always False. CorrectiveRAG overrides this via
        its _is_low_confidence instance variable.

        Returns:
            True if the variant flagged low confidence.
        """
        return getattr(self, "_is_low_confidence", False)

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute confidence score from retrieval results.

        Default implementation: average relevance score of chunks
        that were included in the context (used_in_context=True).

        This is the 'retrieval' method — free, no extra LLM calls.
        CorrectiveRAG may compute its own confidence based on the
        relevance evaluation step.

        Args:
            chunks: Updated chunks with used_in_context flags.
            method: Confidence scoring method from RAGConfig.

        Returns:
            ConfidenceScore with value and method.
        """
        # Only count chunks that were actually used in context
        used_chunks = [c for c in chunks if c.used_in_context]

        if not used_chunks:
            return ConfidenceScore(value=0.0, method=method)

        avg_score = sum(c.relevance_score for c in used_chunks) / len(used_chunks)

        # Clamp to valid range
        avg_score = max(0.0, min(1.0, avg_score))

        return ConfidenceScore(value=round(avg_score, 4), method=method)

    def __repr__(self) -> str:
        """Human-readable representation for logging.

        Returns:
            String like 'SimpleRAG(retriever=dense, llm=gemini)'.
        """
        return (
            f"{self.__class__.__name__}("
            f"retriever={self._retriever.retriever_type}, "
            f"llm={self._llm.provider_name})"
        )