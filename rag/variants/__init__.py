"""
RAG variants subpackage — concrete implementations of BaseRAG.

Each variant overrides specific hooks in the BaseRAG template method.
The sealed query() pipeline remains unchanged across all variants.

Available variants:
    - SimpleRAG: Direct retrieval, single-pass generation. Production default.

Usage:
    from rag.variants import SimpleRAG

    rag = SimpleRAG(retriever=retriever, llm=llm)
    response = await rag.query(request)
"""

from rag.variants.simple_rag import SimpleRAG

__all__ = [
    "SimpleRAG",
]
