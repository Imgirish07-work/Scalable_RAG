"""
Prompts subpackage — RAG prompt templates and builder functions.

All templates are plain Python strings with {placeholders}, filled via
str.format(). No Jinja2 or other template engine dependency.

Builder functions return (system_prompt, user_prompt) tuples ready
for BaseLLM.generate() or BaseLLM.chat().

Usage:
    from rag.prompts import build_rag_prompt, build_relevance_eval_prompt

    system, user = build_rag_prompt(query="What is RAG?", context=ctx)
    response = await llm.generate(user, system_prompt=system)
"""

from rag.prompts.rag_prompt_templates import (
    # Template constants
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CONCISE,
    RAG_USER_PROMPT,
    RAG_USER_PROMPT_WITH_HISTORY,
    RELEVANCE_EVAL_SYSTEM_PROMPT,
    RELEVANCE_EVAL_USER_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
    CONVERSATION_QUERY_REFINEMENT_PROMPT,
    # CoRAG template constants
    CHAIN_DRAFT_SYSTEM_PROMPT,
    CHAIN_DRAFT_USER_PROMPT,
    CHAIN_COMPLETENESS_SYSTEM_PROMPT,
    CHAIN_COMPLETENESS_USER_PROMPT,
    # Builder functions
    build_rag_prompt,
    build_relevance_eval_prompt,
    build_query_rewrite_prompt,
    build_conversation_refinement_prompt,
    format_conversation_history,
    # CoRAG builder functions
    build_chain_draft_prompt,
    build_chain_completeness_prompt,
)

__all__ = [
    # Templates
    "RAG_SYSTEM_PROMPT",
    "RAG_SYSTEM_PROMPT_CONCISE",
    "RAG_USER_PROMPT",
    "RAG_USER_PROMPT_WITH_HISTORY",
    "RELEVANCE_EVAL_SYSTEM_PROMPT",
    "RELEVANCE_EVAL_USER_PROMPT",
    "QUERY_REWRITE_SYSTEM_PROMPT",
    "QUERY_REWRITE_USER_PROMPT",
    "CONVERSATION_QUERY_REFINEMENT_PROMPT",
    # CoRAG templates
    "CHAIN_DRAFT_SYSTEM_PROMPT",
    "CHAIN_DRAFT_USER_PROMPT",
    "CHAIN_COMPLETENESS_SYSTEM_PROMPT",
    "CHAIN_COMPLETENESS_USER_PROMPT",
    # Builders
    "build_rag_prompt",
    "build_relevance_eval_prompt",
    "build_query_rewrite_prompt",
    "build_conversation_refinement_prompt",
    "format_conversation_history",
    # CoRAG builders
    "build_chain_draft_prompt",
    "build_chain_completeness_prompt",
]