"""
RAG variants subpackage — concrete implementations of BaseRAG.

Each variant overrides specific hooks in the BaseRAG template method.
The sealed query() pipeline remains unchanged across all variants.

Available variants:
    - SimpleRAG: Direct retrieval, no evaluation. Production default.
    - CorrectiveRAG: Relevance evaluation + query rewrite retry.
    - ChainRAG: Multi-hop retrieval following dependency chains (CoRAG).

Usage:
    from rag.variants import SimpleRAG, CorrectiveRAG, ChainRAG

    rag = SimpleRAG(retriever=retriever, llm=llm)
    rag = CorrectiveRAG(retriever=retriever, llm=llm)
    rag = ChainRAG(retriever=retriever, llm=llm)
    response = await rag.query(request)
"""

from rag.variants.simple_rag import SimpleRAG
from rag.variants.corrective_rag import CorrectiveRAG
from rag.variants.chain_rag import ChainRAG

__all__ = [
    "SimpleRAG",
    "CorrectiveRAG",
    "ChainRAG",
]