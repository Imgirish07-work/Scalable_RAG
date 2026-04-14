"""
RAG layer public API.

This module exposes the complete RAG interface for all downstream layers
(pipeline, agents, API). Import from here — never from internal submodules.

Usage:
    from rag import RAGFactory, RAGRequest, RAGResponse, RAGConfig

    rag = RAGFactory.create_from_settings(store=qdrant, llm=llm)
    response = await rag.query(RAGRequest(query="What is RAG?", collection_name="docs"))

Build status:
    Step 1: Models + Exceptions  ✅ Complete
    Step 2: Retrievers           ✅ Complete
    Step 3: Context + Prompts    ✅ Complete
    Step 4: BaseRAG + SimpleRAG  ✅ Complete
    Step 5: RAGFactory + Tests   ✅ Complete

Import order follows project convention:
    models → exceptions → retrievers → context → prompts → base → variants → factory
"""

# Models — request/response DTOs and supporting types
from rag.models.rag_request import (
    RAGRequest,
    RAGConfig,
    ConversationTurn,
    MetadataFilter,
    SUPPORTED_RAG_VARIANTS,
    SUPPORTED_RETRIEVAL_MODES,
    SUPPORTED_RERANK_STRATEGIES,
    SUPPORTED_CONFIDENCE_METHODS,
    SUPPORTED_FILTER_OPERATORS,
)
from rag.models.rag_response import (
    RAGResponse,
    RetrievedChunk,
    ConfidenceScore,
    RAGTimings,
)

# Exceptions — full RAG error hierarchy
from rag.exceptions.rag_exceptions import (
    RAGError,
    RAGConfigError,
    RAGRetrievalError,
    RAGContextError,
    RAGGenerationError,
    RAGQualityError,
)

# Retrievers — Strategy pattern for vector store access
from rag.retrieval.base_retriever import BaseRetriever
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever

# Context — assembly and reranking
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker

# Prompts — templates and builder functions
from rag.prompts.rag_prompt_templates import (
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CONCISE,
    build_rag_prompt,
    build_conversation_refinement_prompt,
    format_conversation_history,
)

# Base + Variants
from rag.base_rag import BaseRAG
from rag.variants.simple_rag import SimpleRAG

# Factory — the primary entry point for creating RAG instances
from rag.rag_factory import RAGFactory

__all__ = [
    # Request models
    "RAGRequest",
    "RAGConfig",
    "ConversationTurn",
    "MetadataFilter",
    # Response models
    "RAGResponse",
    "RetrievedChunk",
    "ConfidenceScore",
    "RAGTimings",
    # Exceptions
    "RAGError",
    "RAGConfigError",
    "RAGRetrievalError",
    "RAGContextError",
    "RAGGenerationError",
    "RAGQualityError",
    # Retrievers
    "BaseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    # Context
    "ContextAssembler",
    "ContextRanker",
    # Prompts
    "RAG_SYSTEM_PROMPT",
    "RAG_SYSTEM_PROMPT_CONCISE",
    "build_rag_prompt",
    "build_conversation_refinement_prompt",
    "format_conversation_history",
    # Base + Variants
    "BaseRAG",
    "SimpleRAG",
    # Factory
    "RAGFactory",
    # Constants
    "SUPPORTED_RAG_VARIANTS",
    "SUPPORTED_RETRIEVAL_MODES",
    "SUPPORTED_RERANK_STRATEGIES",
    "SUPPORTED_CONFIDENCE_METHODS",
    "SUPPORTED_FILTER_OPERATORS",
]