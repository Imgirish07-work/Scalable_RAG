"""
Factory for creating RAG variant instances with injected dependencies.

Design:
    Registry-based factory pattern. A class-level dict maps variant name
    strings to their implementation classes. Adding a new variant is one
    line in the registry — nothing else changes. All public methods are
    classmethods; the factory is never instantiated. The factory handles
    two separate concerns: which RAG variant to use and which retriever to
    inject, both resolved from settings or per-request config.

Chain of Responsibility:
    Called by the pipeline layer → creates BaseRAG variant → injects
    BaseRetriever, BaseLLM, CacheManager, ContextRanker, ContextAssembler.
    create_from_settings() uses global settings; create_from_request()
    resolves variant and retriever from RAGConfig.

Dependencies:
    rag.variants (SimpleRAG, CorrectiveRAG, ChainRAG)
    rag.retrieval (DenseRetriever, HybridRetriever)
    rag.context (ContextRanker, ContextAssembler)
    llm.contracts.base_llm (BaseLLM)
    vectorstore.reranker (get_reranker)
    config.settings (settings)
"""

from typing import Optional

from llm.contracts.base_llm import BaseLLM
from rag.base_rag import BaseRAG
from rag.variants.chain_rag import ChainRAG
from rag.variants.simple_rag import SimpleRAG
from rag.variants.corrective_rag import CorrectiveRAG
from rag.retrieval.base_retriever import BaseRetriever
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.models.rag_request import RAGRequest
from rag.exceptions.rag_exceptions import RAGConfigError
from vectorstore.reranker import get_reranker
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGFactory:
    """Factory for creating RAG variant instances with all dependencies injected.

    Class-level registries map variant names and retrieval modes to their
    implementation classes. All public methods are classmethods — no
    instantiation of RAGFactory itself.

    Attributes:
        _variant_registry: Dict mapping variant name string to BaseRAG subclass.
        _retriever_registry: Dict mapping mode name string to BaseRetriever subclass.
    """

    # Variant registry — maps name → class
    _variant_registry: dict[str, type[BaseRAG]] = {
        "simple": SimpleRAG,
        "corrective": CorrectiveRAG,
        "chain": ChainRAG,
    }

    # Retriever registry — maps mode → class
    _retriever_registry: dict[str, type[BaseRetriever]] = {
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
    }

    # Public creation methods

    @classmethod
    def create(
        cls,
        variant_name: str,
        retriever: BaseRetriever,
        llm: BaseLLM,
        cache: object | None = None,
        ranker: ContextRanker | None = None,
        assembler: ContextAssembler | None = None,
        **kwargs,
    ) -> BaseRAG:
        """Create a RAG variant with pre-built dependencies.

        Use this when the retriever, LLM, and other dependencies have
        already been constructed by the caller.

        Args:
            variant_name: Variant to create ('simple', 'corrective', 'chain').
            retriever: Pre-built BaseRetriever instance.
            llm: Pre-built BaseLLM instance.
            cache: Optional CacheManager instance.
            ranker: Optional ContextRanker. Default MMR ranker created if None.
            assembler: Optional ContextAssembler. Default created if None.
            **kwargs: Extra kwargs forwarded to the variant constructor.
                For CorrectiveRAG: pass_threshold, retry_threshold,
                max_retries, eval_chunk_count.

        Returns:
            BaseRAG instance — always the abstract contract, never concrete.

        Raises:
            RAGConfigError: If variant_name is not in the registry.
        """
        cleaned = cls._validate_variant(variant_name)
        variant_class = cls._variant_registry[cleaned]

        logger.info(
            "Creating RAG variant | variant=%s | retriever=%s | llm=%s",
            cleaned,
            retriever.retriever_type,
            llm.provider_name,
        )

        return variant_class(
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
            **kwargs,
        )

    @classmethod
    def create_retriever(
        cls,
        store: object,
        mode: str = "dense",
        **kwargs,
    ) -> BaseRetriever:
        """Create a retriever from a QdrantStore and mode string.

        Convenience method so callers do not need to import retriever classes.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
            mode: Retrieval mode ('dense' or 'hybrid').
            **kwargs: Extra kwargs forwarded to the retriever constructor.
                For HybridRetriever: dense_weight, sparse_weight.

        Returns:
            BaseRetriever instance.

        Raises:
            RAGConfigError: If mode is not in the retriever registry.
        """
        cleaned = mode.strip().lower()

        if cleaned not in cls._retriever_registry:
            raise RAGConfigError(
                f"Retrieval mode '{mode}' not supported. "
                f"Must be one of: {sorted(cls._retriever_registry.keys())}",
                details={"mode": mode},
            )

        retriever_class = cls._retriever_registry[cleaned]

        logger.info("Creating retriever | mode=%s", cleaned)

        return retriever_class(store=store, **kwargs)

    @classmethod
    def create_from_settings(
        cls,
        store: object,
        llm: BaseLLM,
        cache: object | None = None,
        embeddings_fn: object | None = None,
    ) -> BaseRAG:
        """Create a fully wired RAG instance from global settings.

        Reads RAG_DEFAULT_VARIANT, RAG_RETRIEVAL_MODE, RAG_RERANK_STRATEGY,
        and RAG_MAX_CONTEXT_TOKENS from settings. Constructs retriever,
        ranker, assembler, and variant in one call.

        Use this during pipeline startup for zero-config RAG initialization.

        Args:
            store: QdrantStore instance.
            llm: BaseLLM instance.
            cache: Optional CacheManager instance.
            embeddings_fn: Optional callable returning the embedding model.
                Required for MMR reranking. Pass get_embeddings from
                vectorstore/embeddings.py.

        Returns:
            Fully configured BaseRAG instance.

        Raises:
            RAGConfigError: If settings contain invalid variant or mode values.
        """
        variant_name = getattr(settings, "RAG_DEFAULT_VARIANT", "simple")
        retrieval_mode = getattr(settings, "RAG_RETRIEVAL_MODE", "dense")
        rerank_strategy = getattr(settings, "RAG_RERANK_STRATEGY", "mmr")
        max_context_tokens = getattr(settings, "RAG_MAX_CONTEXT_TOKENS", 3072)

        logger.info(
            "Creating RAG from settings | variant=%s | retrieval=%s | "
            "rerank=%s | max_context=%d",
            variant_name,
            retrieval_mode,
            rerank_strategy,
            max_context_tokens,
        )

        # Build retriever
        retriever = cls.create_retriever(store=store, mode=retrieval_mode)

        # Build ranker — inject reranker only when cross_encoder is configured
        top_k = getattr(settings, "RAG_TOP_K", 5)
        reranker = get_reranker() if rerank_strategy == "cross_encoder" else None
        ranker = ContextRanker(
            strategy=rerank_strategy,
            embeddings_fn=embeddings_fn,
            reranker=reranker,
            top_k=top_k,
        )

        # Build assembler
        assembler = ContextAssembler(
            llm=llm,
            max_tokens=max_context_tokens,
        )

        # Build variant with all wired dependencies
        return cls.create(
            variant_name=variant_name,
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
        )

    @classmethod
    def create_from_request(
        cls,
        request: RAGRequest,
        store: object,
        llm: BaseLLM,
        cache: object | None = None,
        embeddings_fn: object | None = None,
    ) -> BaseRAG:
        """Create a RAG instance tailored to a specific request's config.

        Resolves the variant from RAGConfig.resolve_variant(). If
        config.rag_variant is set, it wins. Otherwise falls back to
        settings.RAG_DEFAULT_VARIANT (the smart default pattern).

        Use this when different requests need different RAG variants.

        Args:
            request: RAGRequest whose config carries variant preference.
            store: QdrantStore instance.
            llm: BaseLLM instance.
            cache: Optional CacheManager instance.
            embeddings_fn: Optional callable for MMR reranking.

        Returns:
            BaseRAG instance configured for this specific request.

        Raises:
            RAGConfigError: If the resolved variant is not in the registry.
        """
        config = request.config
        variant_name = config.resolve_variant()

        logger.info(
            "Creating RAG from request | variant=%s | retrieval=%s | "
            "request_id=%s",
            variant_name,
            config.retrieval_mode,
            request.request_id,
        )

        # Build retriever from request config
        retriever = cls.create_retriever(
            store=store,
            mode=config.retrieval_mode,
        )

        # Build ranker from request config — inject reranker when needed
        reranker = get_reranker() if config.rerank_strategy == "cross_encoder" else None
        ranker = ContextRanker(
            strategy=config.rerank_strategy,
            embeddings_fn=embeddings_fn,
            reranker=reranker,
            top_k=config.top_k,
        )

        # Build assembler from request config
        assembler = ContextAssembler(
            llm=llm,
            max_tokens=config.max_context_tokens,
        )

        return cls.create(
            variant_name=variant_name,
            retriever=retriever,
            llm=llm,
            cache=cache,
            ranker=ranker,
            assembler=assembler,
        )

    # Registry management

    @classmethod
    def available_variants(cls) -> list[str]:
        """Return all registered RAG variant names.

        Returns:
            Sorted list of variant name strings.
        """
        return sorted(cls._variant_registry.keys())

    @classmethod
    def available_retrieval_modes(cls) -> list[str]:
        """Return all registered retrieval mode names.

        Returns:
            Sorted list of retrieval mode strings.
        """
        return sorted(cls._retriever_registry.keys())

    @classmethod
    def register_variant(
        cls,
        variant_name: str,
        variant_class: type[BaseRAG],
    ) -> None:
        """Register a new RAG variant at runtime.

        Allows plugins, tests, or future layers (RLM) to add variants
        without modifying factory source code.

        Args:
            variant_name: Unique name string for the new variant.
            variant_class: Class that extends BaseRAG.

        Raises:
            RAGConfigError: If variant_class does not extend BaseRAG.
        """
        if not (isinstance(variant_class, type) and issubclass(variant_class, BaseRAG)):
            raise RAGConfigError(
                f"Cannot register '{variant_name}'. "
                f"Variant class must extend BaseRAG.",
                details={"class": str(variant_class)},
            )

        cls._variant_registry[variant_name.strip().lower()] = variant_class
        logger.info("Registered RAG variant | variant=%s", variant_name)

    # Private helpers

    @classmethod
    def _validate_variant(cls, variant_name: str) -> str:
        """Validate a variant name against the registry.

        Args:
            variant_name: Raw variant name string from the caller.

        Returns:
            Cleaned lowercase variant name.

        Raises:
            RAGConfigError: If the name is empty or not in the registry.
        """
        if not variant_name or not variant_name.strip():
            raise RAGConfigError(
                "Variant name cannot be empty. "
                f"Available variants: {cls.available_variants()}",
            )

        cleaned = variant_name.strip().lower()

        if cleaned not in cls._variant_registry:
            raise RAGConfigError(
                f"RAG variant '{variant_name}' is not registered. "
                f"Available variants: {cls.available_variants()}",
                details={"requested": variant_name},
            )

        return cleaned
