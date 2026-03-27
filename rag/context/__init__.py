"""
Context subpackage — assembly and reranking of retrieved chunks.

Provides:
    - ContextAssembler: Formats chunks into token-bounded context strings.
    - ContextRanker: Post-retrieval reranking (MMR, cross-encoder, none).

Usage:
    from rag.context import ContextAssembler, ContextRanker

    ranker = ContextRanker(strategy="mmr", embeddings_fn=get_embeddings)
    assembler = ContextAssembler(llm=llm, max_tokens=3072)

    ranked = await ranker.rank(chunks, query)
    context_str, updated_chunks, tokens = await assembler.assemble(ranked)
"""

from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker

__all__ = [
    "ContextAssembler",
    "ContextRanker",
]