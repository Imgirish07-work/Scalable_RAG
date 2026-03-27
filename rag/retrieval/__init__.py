"""
Retrieval subpackage — Strategy pattern for vector store access.

Provides interchangeable retrieval strategies that wrap the existing
QdrantStore from Layer 3. BaseRAG takes a BaseRetriever via constructor
injection — variants never know the concrete retriever type.

Usage:
    from rag.retrieval import DenseRetriever, HybridRetriever

    retriever = DenseRetriever(store=qdrant_store)
    chunks = await retriever.retrieve("What is RAG?", top_k=5)

Available retrievers:
    - DenseRetriever: Dense-only vector similarity (default, fast)
    - HybridRetriever: Dense + SPLADE sparse (keyword-aware, 1.5-2x slower)
"""

from rag.retrieval.base_retriever import BaseRetriever
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "HybridRetriever",
]