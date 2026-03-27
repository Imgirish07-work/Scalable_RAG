"""
RAG factory — creates RAG variant instances with all dependencies injected.

Design:
    - Registry dict maps variant name → variant class. Adding a new variant
      is 1 line in _registry, nothing else changes.
    - Unlike LLMFactory (which creates self-contained providers), RAGFactory
      must inject dependencies (retriever, LLM, cache, ranker, assembler).
      The caller provides infrastructure, the factory provides the variant.
    - create_from_settings() reads RAG_DEFAULT_VARIANT and RAG_RETRIEVAL_MODE
      from settings, creates the matching retriever, and wires everything.
    - create_from_request() resolves the variant from RAGConfig.resolve_variant()
      — this is the smart default pattern in action.

Retriever creation:
    - The factory also creates the retriever based on retrieval_mode.
      This keeps retriever construction out of pipeline code.
    - Dense is default. Hybrid requires QdrantStore with SPLADE configured.

Integration:
    - SimpleRAG, CorrectiveRAG from rag/variants/
    - DenseRetriever, HybridRetriever from rag/retrieval/
    - ContextRanker from rag/context/context_ranker.py
    - ContextAssembler from rag/context/context_assembler.py
    - BaseLLM from llm/contracts/base_llm.py
    - CacheManager from cache/cache_manager.py (optional)
    - get_embeddings from vectorstore/embeddings.py
    - Settings from config/settings.py
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
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGFactory:
    """Factory for creating RAG variant instances with dependencies.

    Class-level registry maps variant names to their implementation
    classes. All public methods are classmethods — no instantiation.

    The factory handles two separate creation concerns:
        1. Which RAG variant to use (simple, corrective)
        2. Which retriever to inject (dense, hybrid)

    Both are resolved from settings or per-request config.

    Attributes:
        _variant_registry: Dict mapping variant name → BaseRAG subclass.
        _retriever_registry: Dict mapping mode name → BaseRetriever subclass.
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

    # ================================================================
    # Public creation methods
    # ================================================================

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

        Use this when you've already constructed the retriever, LLM,
        and other dependencies yourself.

        Args:
            variant_name: Variant to create ('simple', 'corrective').
            retriever: Pre-built BaseRetriever instance.
            llm: Pre-built BaseLLM instance.
            cache: Optional CacheManager instance.
            ranker: Optional ContextRanker. Default MMR created if None.
            assembler: Optional ContextAssembler. Default created if None.
            **kwargs: Extra kwargs passed to variant constructor.
                For CorrectiveRAG: pass_threshold, retry_threshold,
                max_retries, eval_chunk_count.

        Returns:
            BaseRAG instance — always the contract, never concrete.

        Raises:
            RAGConfigError: If variant_name is not registered.
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

        Convenience method so callers don't import retriever classes.

        Args:
            store: QdrantStore instance from vectorstore/qdrant_store.py.
            mode: Retrieval mode ('dense' or 'hybrid').
            **kwargs: Extra kwargs for retriever constructor.
                For HybridRetriever: dense_weight, sparse_weight.

        Returns:
            BaseRetriever instance.

        Raises:
            RAGConfigError: If mode is not registered.
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
        """Create a complete RAG instance from settings.

        Reads RAG_DEFAULT_VARIANT, RAG_RETRIEVAL_MODE, RAG_RERANK_STRATEGY,
        and RAG_MAX_CONTEXT_TOKENS from settings. Constructs retriever,
        ranker, assembler, and variant — fully wired.

        Use this in pipeline startup for zero-config RAG initialization.

        Args:
            store: QdrantStore instance.
            llm: BaseLLM instance.
            cache: Optional CacheManager instance.
            embeddings_fn: Optional callable returning embedding model.
                Required for MMR reranking. Pass get_embeddings from
                vectorstore/embeddings.py.

        Returns:
            Fully configured BaseRAG instance.

        Raises:
            RAGConfigError: If settings contain invalid values.
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

        # Build ranker
        ranker = ContextRanker(
            strategy=rerank_strategy,
            embeddings_fn=embeddings_fn,
        )

        # Build assembler
        assembler = ContextAssembler(
            llm=llm,
            max_tokens=max_context_tokens,
        )

        # Build variant
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
        """Create a RAG instance based on a specific request's config.

        Resolves the variant from RAGConfig.resolve_variant() — this is
        the smart default pattern. If config.rag_variant is set, use it.
        Otherwise fall back to settings.RAG_DEFAULT_VARIANT.

        Use this when different requests need different variants.

        Args:
            request: RAGRequest with config containing variant preference.
            store: QdrantStore instance.
            llm: BaseLLM instance.
            cache: Optional CacheManager instance.
            embeddings_fn: Optional callable for MMR reranking.

        Returns:
            BaseRAG instance configured for this specific request.

        Raises:
            RAGConfigError: If resolved variant is not registered.
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

        # Build ranker from request config
        ranker = ContextRanker(
            strategy=config.rerank_strategy,
            embeddings_fn=embeddings_fn,
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

    # ================================================================
    # Registry management
    # ================================================================

    @classmethod
    def available_variants(cls) -> list[str]:
        """Return list of all registered variant names.

        Returns:
            Sorted list of variant name strings.
        """
        return sorted(cls._variant_registry.keys())

    @classmethod
    def available_retrieval_modes(cls) -> list[str]:
        """Return list of all registered retrieval modes.

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
            variant_name: Unique name string.
            variant_class: Class that extends BaseRAG.

        Raises:
            RAGConfigError: If class does not extend BaseRAG.
        """
        if not (isinstance(variant_class, type) and issubclass(variant_class, BaseRAG)):
            raise RAGConfigError(
                f"Cannot register '{variant_name}'. "
                f"Variant class must extend BaseRAG.",
                details={"class": str(variant_class)},
            )

        cls._variant_registry[variant_name.strip().lower()] = variant_class
        logger.info("Registered RAG variant | variant=%s", variant_name)

    # ================================================================
    # Private helpers
    # ================================================================

    @classmethod
    def _validate_variant(cls, variant_name: str) -> str:
        """Validate variant name exists in registry.

        Args:
            variant_name: Raw variant name string.

        Returns:
            Cleaned lowercase variant name.

        Raises:
            RAGConfigError: If variant not found in registry.
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