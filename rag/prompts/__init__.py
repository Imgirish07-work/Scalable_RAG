"""Prompts subpackage: RAG prompt templates and builder functions."""

from rag.prompts.rag_prompt_templates import (
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CONCISE,
    RAG_USER_PROMPT,
    RAG_USER_PROMPT_WITH_HISTORY,
    CONVERSATION_QUERY_REFINEMENT_PROMPT,
    build_rag_prompt,
    build_conversation_refinement_prompt,
    format_conversation_history,
)

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_SYSTEM_PROMPT_CONCISE",
    "RAG_USER_PROMPT",
    "RAG_USER_PROMPT_WITH_HISTORY",
    "CONVERSATION_QUERY_REFINEMENT_PROMPT",
    "build_rag_prompt",
    "build_conversation_refinement_prompt",
    "format_conversation_history",
]
