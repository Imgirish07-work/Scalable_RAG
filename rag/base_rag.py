"""
Abstract base class for all RAG variants using the Template Method pattern.

Design:
    Template Method pattern: query() is a sealed algorithm skeleton.
    No variant can reorder, skip, or add pipeline steps — they can only
    override individual hooks. Cache integration, timing instrumentation,
    and confidence computation all live here and are inherited for free.
    All dependencies (retriever, LLM, cache, ranker, assembler) are
    constructor-injected for full testability with mocks.

Chain of Responsibility:
    RAGFactory creates and injects dependencies → BaseRAG.query() orchestrates
    the sealed pipeline → retrieve() calls BaseRetriever → rank() calls
    ContextRanker → assemble_context() calls ContextAssembler →
    generate() calls BaseLLM.chat().

Dependencies:
    llm.contracts.base_llm (BaseLLM)
    llm.models.llm_response (LLMResponse)
    llm.provider_health (provider_health)
    rag.models.rag_request (RAGRequest)
    rag.models.rag_response (RAGResponse, RetrievedChunk, ConfidenceScore, RAGTimings)
    rag.retrieval.base_retriever (BaseRetriever)
    rag.context.context_assembler (ContextAssembler)
    rag.context.context_ranker (ContextRanker)
    rag.prompts.rag_prompt_templates
    rag.exceptions.rag_exceptions
"""

import time
from abc import ABC, abstractmethod

from llm.contracts.base_llm import BaseLLM
from llm.models.llm_response import LLMResponse
from llm.provider_health import provider_health
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
        - retrieve(query, top_k, filters) -> list[RetrievedChunk]
        - variant_name (property) -> str

    Subclasses MAY override:
        - pre_process(request) -> str
        - rank(chunks, query) -> list[RetrievedChunk]
        - assemble_context(chunks) -> tuple[str, list[RetrievedChunk], int]
        - generate(context, query, request) -> LLMResponse

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
        fallback_llm: BaseLLM | None = None,
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
            fallback_llm: Optional secondary LLM used when the primary
                provider enters cooldown or fails with a provider error.
        """
        self._retriever = retriever
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._cache = cache
        # Inject reranker at construction so per-request cross_encoder works
        # without re-instantiating the ranker on every query.
        self._ranker = ranker or ContextRanker(
            strategy="mmr",
            embeddings_fn=get_embeddings,
            reranker=get_reranker(),
            top_k=5,
        )
        self._assembler = assembler or ContextAssembler(llm=llm)

        logger.info(
            "BaseRAG initialized | variant=%s | retriever=%s | "
            "llm=%s | fallback_llm=%s | cache=%s",
            self.variant_name,
            self._retriever.retriever_type,
            self._llm.provider_name,
            self._fallback_llm.provider_name if self._fallback_llm else "none",
            "enabled" if self._cache else "disabled",
        )

    # Abstract — subclasses MUST implement

    @property
    @abstractmethod
    def variant_name(self) -> str:
        """Return the variant identifier string.

        Returns:
            Variant name e.g. 'simple', 'chain'.
        """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list | None = None,
        request: RAGRequest | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from the vector store.

        This is the primary hook every variant implements. SimpleRAG
        calls the retriever directly.

        Args:
            query: Processed query string (output of pre_process).
            top_k: Maximum chunks to retrieve.
            filters: Optional metadata filters from RAGConfig.
            request: Full RAGRequest for any variant-specific config.
                SimpleRAG ignores it.

        Returns:
            List of RetrievedChunk ordered by relevance.

        Raises:
            RAGRetrievalError: If retrieval fails.
        """

    # Sealed pipeline — query() orchestrates everything

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

        # Step 1: Cache check
        if self._cache:
            cache_result = await self._try_cache_read(request)
            if cache_result is not None:
                return cache_result

        # Step 2: Pre-process
        processed_query = await self.pre_process(request)

        # Step 3: Retrieve
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
            request=request,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 4: Rank
        ranking_start = time.perf_counter()
        ranked_chunks = await self.rank(chunks, processed_query, strategy=active_strategy)
        ranking_ms = (time.perf_counter() - ranking_start) * 1000

        # Step 4b: Reranker quality check + MMR recovery.
        #
        # When cross-encoder scores fall below threshold (common in narrative/
        # domain-specific corpora where ms-marco-MiniLM is miscalibrated), we
        # do NOT immediately fail. Instead, re-rank the same coarse candidates
        # with MMR — no extra retrieval call, zero added latency for the normal
        # path. Only if MMR also returns nothing do we return "no context".
        #
        # This eliminates the retrieval-reranker domain-mismatch failure mode
        # without adding any network calls.
        if ranked_chunks:
            reranker_scores = [
                c.reranker_score for c in ranked_chunks if c.reranker_score is not None
            ]
            if reranker_scores:
                from config.settings import settings as _s
                threshold = getattr(_s, "RERANKER_SCORE_THRESHOLD", 0.05)
                top_reranker_score = max(reranker_scores)
                if top_reranker_score < threshold:
                    logger.warning(
                        "Cross-encoder threshold not met | top_score=%.4f | "
                        "threshold=%.2f | re-ranking coarse candidates with MMR",
                        top_reranker_score,
                        threshold,
                    )
                    # Re-rank the original coarse candidates (still in `chunks`)
                    # with MMR — MMR is corpus-agnostic and does not depend on
                    # ms-marco calibration.
                    ranked_chunks = await self._ranker.rank(
                        chunks, processed_query, strategy="mmr"
                    )
                    ranking_ms = (time.perf_counter() - ranking_start) * 1000

                    if not ranked_chunks:
                        # Genuine zero-retrieval failure — nothing to recover from.
                        total_ms = (time.perf_counter() - total_start) * 1000
                        logger.warning(
                            "MMR fallback returned no chunks — returning "
                            "low-confidence response | request_id=%s",
                            request.request_id,
                        )
                        # Unblock any coalesced requests waiting on this key.
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

                    logger.info(
                        "MMR fallback succeeded | chunks_recovered=%d | "
                        "original_top_score=%.4f",
                        len(ranked_chunks),
                        top_reranker_score,
                    )

        # Step 5: Assemble context
        context_str, updated_chunks, context_tokens = await self.assemble_context(
            ranked_chunks
        )

        # Step 6: Generate
        generation_start = time.perf_counter()
        llm_response = await self.generate(context_str, processed_query, request)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        # Step 7: Build response
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

        # Omit source chunks when the caller has disabled source inclusion
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

        # Step 8: Cache write
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

    # Overridable hooks — sensible defaults, variants customize

    async def pre_process(self, request: RAGRequest) -> str:
        """Pre-process the query before retrieval.

        Default behavior: if conversation_history exists, use the LLM to
        resolve pronouns and make the query self-contained. Otherwise,
        return the query as-is (already stripped by RAGRequest validator).

        QueryExpansionRAG (future) would override this to generate a
        hypothetical answer for HyDE embedding.

        Args:
            request: Full RAGRequest with query and optional history.

        Returns:
            Processed query string ready for retrieval.
        """
        # Without history there is nothing to resolve — return directly
        chat_messages = request.get_chat_messages()
        if not chat_messages:
            return request.query

        # Build a conversation-aware refinement prompt
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
        """Rerank retrieved chunks using the configured strategy.

        Default behavior: delegate to the injected ContextRanker.
        Subclasses may override this to add variant-specific evaluation
        before or after reranking.

        Args:
            chunks: Retrieved chunks from retrieve().
            query: Processed query string.
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
            - Call the LLM via chat() with system + user messages.
            - If the primary LLM is in cooldown, route directly to the
              fallback LLM without paying the timeout penalty.
            - On provider-level failure, mark primary unhealthy and retry
              generation with fallback (saves re-running full retrieval).

        MultiAgentRAG (future) would override this to synthesize answers
        across multiple sub-query results.

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

        # Build prompt pair from templates
        system_prompt, user_prompt = build_rag_prompt(
            query=query,
            context=context,
            conversation_history=history_str,
        )

        # Allow per-request system prompt override
        if request.config.system_prompt:
            system_prompt = request.config.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            from llm.exceptions.llm_exceptions import LLMError, LLMProviderError

            # Skip primary LLM if it is in cooldown (e.g. blocked by Zscaler).
            # Routes directly to fallback without paying the timeout penalty.
            _skip_primary = (
                not provider_health.is_available(self._llm.provider_name)
                and self._fallback_llm is not None
            )
            if _skip_primary:
                logger.info(
                    "Primary LLM '%s' in cooldown — routing directly to fallback '%s'",
                    self._llm.provider_name,
                    self._fallback_llm.provider_name,
                )
                response = await self._fallback_llm.chat(
                    messages,
                    temperature=request.config.temperature,
                )
            else:
                response = await self._llm.chat(
                    messages,
                    temperature=request.config.temperature,
                )
                # Successful call — clear any prior failure state immediately.
                provider_health.mark_recovered(self._llm.provider_name)

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
            # Any LLM-layer error (timeout, rate-limit, provider failure) with a
            # fallback configured → retry generation with the fallback LLM only.
            # Re-retrieval is skipped — this saves ~3-4s vs re-running the pipeline.
            # Only LLMProviderError (hard failure) marks the primary as unavailable;
            # transient errors (timeout, rate-limit) let the pool self-recover.
            if isinstance(exc, LLMError) and self._fallback_llm is not None:
                if isinstance(exc, LLMProviderError):
                    provider_health.mark_failed(self._llm.provider_name)
                logger.warning(
                    "Primary LLM failed (%s), retrying generation with fallback | "
                    "primary=%s | fallback=%s | error=%s",
                    type(exc).__name__,
                    self._llm.provider_name,
                    self._fallback_llm.provider_name,
                    str(exc),
                )
                try:
                    response = await self._fallback_llm.chat(
                        messages,
                        temperature=request.config.temperature,
                    )
                    if not response.text or not response.text.strip():
                        raise RAGGenerationError(
                            "Fallback LLM returned empty response.",
                            details={"request_id": request.request_id},
                        )
                    return response
                except RAGGenerationError:
                    raise
                except Exception as fallback_exc:
                    raise RAGGenerationError(
                        f"Both primary and fallback LLM failed: {fallback_exc}",
                        details={"request_id": request.request_id},
                    ) from fallback_exc

            # No fallback configured — propagate LLM errors as-is, wrap the rest
            if isinstance(exc, LLMError):
                raise

            raise RAGGenerationError(
                f"RAG generation failed: {exc}",
                details={
                    "request_id": request.request_id,
                    "model": self._llm.model_name,
                },
            ) from exc

    # Private helpers

    async def _try_cache_read(self, request: RAGRequest) -> RAGResponse | None:
        """Attempt to read a cached response. Returns None on miss or error.

        Cache errors are caught and logged — they never propagate to the
        caller. A cache failure means the full pipeline runs instead.

        Args:
            request: RAGRequest used to compute the cache key.

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
                if result.strategy.value == "semantic":
                    logger.info(
                        "Cache hit | request_id=%s | layer=%s | strategy=%s | "
                        "similarity=%.3f | latency=%.1f ms",
                        request.request_id,
                        result.layer,
                        result.strategy,
                        result.similarity_score,
                        result.lookup_latency_ms,
                    )
                else:
                    logger.info(
                        "Cache hit | request_id=%s | layer=%s | strategy=%s | "
                        "latency=%.1f ms",
                        request.request_id,
                        result.layer,
                        result.strategy,
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
        sources: list[RetrievedChunk] | None = None,
        confidence: ConfidenceScore | None = None,
    ) -> None:
        """Attempt to write a response to cache. Errors are caught and logged.

        Cache write is the last step before return. Failures do not affect
        the response already built — they are surfaced only via a warning log.

        Args:
            request: RAGRequest used to compute the cache key.
            llm_response: LLMResponse to store.
            sources: Retrieved chunks to store alongside the response.
            confidence: Confidence score to persist for cache hit responses.
        """
        try:
            await self._cache.set(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                response=llm_response,
                system_prompt=request.config.system_prompt or "",
                sources=[chunk.model_dump() for chunk in (sources or [])],
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
        """Return whether the current query result has low confidence.

        Default: always False. Subclasses override this via the
        _is_low_confidence instance variable set during retrieve().

        Returns:
            True if the variant flagged low confidence for this query.
        """
        return getattr(self, "_is_low_confidence", False)

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute a confidence score from retrieval results.

        Averages relevance scores of chunks that were included in the
        context (used_in_context=True). Prefers reranker scores when
        available — cross-encoder attends jointly to (query, chunk) and
        is a stronger relevance signal than cosine distance alone.

        Subclasses may override this to use LLM-evaluated relevance scores.

        Args:
            chunks: Updated chunks with used_in_context flags set.
            method: Confidence scoring method string from RAGConfig.

        Returns:
            ConfidenceScore with value and method.
        """
        # Only count chunks that were actually used in context
        used_chunks = [c for c in chunks if c.used_in_context]

        if not used_chunks:
            return ConfidenceScore(value=0.0, method=method)

        import math

        # Prefer reranker scores when available — the cross-encoder reads
        # (query, chunk) jointly via full attention, making it a direct
        # semantic relevance signal rather than a vector distance proxy.
        # Fall back to cosine/RRF relevance scores when reranker didn't run.
        reranker_scores = [
            c.reranker_score for c in used_chunks if c.reranker_score is not None
        ]
        if reranker_scores:
            scores = sorted(reranker_scores, reverse=True)
            method = "reranker"
        else:
            # Average the top-⌈k/2⌉ scores — avoids skew from low-scoring tail
            # chunks (especially pronounced with hybrid RRF scoring).
            scores = sorted(
                (c.relevance_score for c in used_chunks), reverse=True
            )

        top_n = max(1, math.ceil(len(scores) / 2))
        avg_score = sum(scores[:top_n]) / top_n

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
