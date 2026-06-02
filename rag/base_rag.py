"""Abstract base class for RAG variants using Template Method; query() is sealed."""

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
    """Abstract base class for all RAG variants."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache: object | None = None,
        ranker: ContextRanker | None = None,
        assembler: ContextAssembler | None = None,
        fallback_llm: BaseLLM | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._cache = cache
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

    @property
    @abstractmethod
    def variant_name(self) -> str:
        """Return the variant identifier string."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: list | None = None,
        request: RAGRequest | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from the vector store."""

    async def query(self, request: RAGRequest) -> RAGResponse:
        """Execute the full RAG pipeline; sealed, do not override."""
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

        if self._cache:
            cache_result = await self._try_cache_read(request)
            if cache_result is not None:
                return cache_result

        processed_query = await self.pre_process(request)

        # cross_encoder needs a larger coarse candidate pool before reranking
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

        ranking_start = time.perf_counter()
        ranked_chunks = await self.rank(chunks, processed_query, strategy=active_strategy)
        ranking_ms = (time.perf_counter() - ranking_start) * 1000

        if ranked_chunks:
            reranker_scores = [
                c.reranker_score for c in ranked_chunks if c.reranker_score is not None
            ]
            if reranker_scores:
                threshold = config.reranker_score_threshold
                top_reranker_score = max(reranker_scores)
                if top_reranker_score < threshold:
                    logger.warning(
                        "Cross-encoder threshold not met | top_score=%.4f | "
                        "threshold=%.2f | re-ranking coarse candidates with MMR",
                        top_reranker_score,
                        threshold,
                    )
                    ranked_chunks = await self._ranker.rank(
                        chunks, processed_query, strategy="mmr"
                    )
                    ranking_ms = (time.perf_counter() - ranking_start) * 1000

                    if not ranked_chunks:
                        total_ms = (time.perf_counter() - total_start) * 1000
                        logger.warning(
                            "MMR fallback returned no chunks — returning "
                            "low-confidence response | request_id=%s",
                            request.request_id,
                        )
                        if self._cache:
                            try:
                                await self._cache.resolve_in_flight(
                                    query=request.query,
                                    model_name=self._llm.model_name,
                                    temperature=request.config.temperature,
                                    system_prompt=request.config.system_prompt or "",
                                    user_id=request.user_id or "",
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

        _min_ctx = config.min_context_chunks
        if ranked_chunks and len(ranked_chunks) < _min_ctx:
            already_ids = {c.chunk_id for c in ranked_chunks}
            backfill = sorted(
                (c for c in chunks if c.chunk_id not in already_ids),
                key=lambda c: c.relevance_score,
                reverse=True,
            )
            needed = _min_ctx - len(ranked_chunks)
            if backfill:
                ranked_chunks = ranked_chunks + backfill[:needed]
                logger.info(
                    "Min-context backfill | added=%d | total=%d | min_required=%d",
                    min(needed, len(backfill)), len(ranked_chunks), _min_ctx,
                )
            elif active_strategy != "cross_encoder":
                expanded_k = config.top_k * 2
                logger.info(
                    "Adaptive top_k expansion | original_k=%d | expanded_k=%d",
                    config.top_k,
                    expanded_k,
                )
                expanded_chunks = await self.retrieve(
                    query=processed_query,
                    top_k=expanded_k,
                    filters=config.metadata_filters,
                    request=request,
                )
                ranked_chunks = await self.rank(
                    expanded_chunks, processed_query, strategy=active_strategy
                )
                ranking_ms = (time.perf_counter() - ranking_start) * 1000

        context_str, updated_chunks, context_tokens = await self.assemble_context(
            ranked_chunks
        )

        generation_start = time.perf_counter()
        llm_response = await self.generate(context_str, processed_query, request)
        generation_ms = (time.perf_counter() - generation_start) * 1000

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

    async def pre_process(self, request: RAGRequest) -> str:
        """Pre-process the query; resolves pronouns via LLM when history exists."""
        chat_messages = request.get_chat_messages()
        if not chat_messages:
            return request.query

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
        """Rerank retrieved chunks using the configured strategy."""
        return await self._ranker.rank(chunks, query, strategy=strategy)

    async def assemble_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[RetrievedChunk], int]:
        """Assemble ranked chunks into a token-bounded context string."""
        return await self._assembler.assemble(chunks)

    async def generate(
        self,
        context: str,
        query: str,
        request: RAGRequest,
    ) -> LLMResponse:
        """Generate an answer using the LLM with assembled context."""
        chat_messages = request.get_chat_messages()
        history_str = (
            format_conversation_history(chat_messages)
            if chat_messages
            else None
        )

        system_prompt, user_prompt = build_rag_prompt(
            query=query,
            context=context,
            conversation_history=history_str,
        )

        if request.config.system_prompt:
            system_prompt = request.config.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            from llm.exceptions.llm_exceptions import LLMError, LLMProviderError

            # skip primary if in cooldown to avoid unnecessary timeout latency
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
            # hard LLMProviderError marks primary unavailable; other LLMErrors do not
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

            if isinstance(exc, LLMError):
                raise

            raise RAGGenerationError(
                f"RAG generation failed: {exc}",
                details={
                    "request_id": request.request_id,
                    "model": self._llm.model_name,
                },
            ) from exc

    async def _try_cache_read(self, request: RAGRequest) -> RAGResponse | None:
        """Attempt to read a cached response. Returns None on miss or error."""
        try:
            result = await self._cache.get_or_wait(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                system_prompt=request.config.system_prompt or "",
                user_id=request.user_id or "",
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
        """Attempt to write a response to cache. Errors are caught and logged."""
        try:
            await self._cache.set(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                response=llm_response,
                system_prompt=request.config.system_prompt or "",
                sources=[chunk.model_dump() for chunk in (sources or [])],
                confidence_value=confidence.value if confidence is not None else 0.0,
                user_id=request.user_id or "",
            )
            await self._cache.resolve_in_flight(
                query=request.query,
                model_name=self._llm.model_name,
                temperature=request.config.temperature,
                system_prompt=request.config.system_prompt or "",
                user_id=request.user_id or "",
            )
        except Exception as exc:
            logger.warning(
                "Cache write failed | request_id=%s | error=%s",
                request.request_id,
                str(exc),
            )

    def _get_low_confidence_flag(self) -> bool:
        """Return whether the current query result has low confidence."""
        return getattr(self, "_is_low_confidence", False)

    def _compute_confidence(
        self,
        chunks: list[RetrievedChunk],
        method: str = "retrieval",
    ) -> ConfidenceScore:
        """Compute a confidence score from retrieval results."""
        used_chunks = [c for c in chunks if c.used_in_context]

        if not used_chunks:
            return ConfidenceScore(value=0.0, method=method)

        import math

        # cross-encoder scores are preferred: joint (query, chunk) attention is stronger than cosine
        reranker_scores = [
            c.reranker_score for c in used_chunks if c.reranker_score is not None
        ]
        if reranker_scores:
            scores = sorted(reranker_scores, reverse=True)
            method = "reranker"
        else:
            # average top-ceil(k/2) to avoid skew from low-scoring tail chunks
            scores = sorted(
                (c.relevance_score for c in used_chunks), reverse=True
            )

        top_n = max(1, math.ceil(len(scores) / 2))
        avg_score = sum(scores[:top_n]) / top_n

        avg_score = max(0.0, min(1.0, avg_score))

        return ConfidenceScore(value=round(avg_score, 4), method=method)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"retriever={self._retriever.retriever_type}, "
            f"llm={self._llm.provider_name})"
        )
