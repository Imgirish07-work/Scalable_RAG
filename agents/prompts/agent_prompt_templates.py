"""
Prompt templates for the agent layer.

Design:
    Two prompt pairs: planning and synthesis. Each is a module-level constant
    (static system prompt) plus a builder function that injects runtime values
    into the user prompt. All builders return (system_prompt, user_prompt)
    tuples compatible with BaseLLM.chat().

    A third lightweight prompt handles weak sub-query rewriting — used with
    the fast LLM to reformulate low-quality sub-queries before re-retrieval.

Chain of Responsibility:
    QueryPlanner calls build_planning_prompt().
    AgentOrchestrator calls build_rewrite_prompt() for weak sub-queries (fast LLM).
    AgentOrchestrator calls build_synthesis_prompt() for final generation (strong LLM).

Dependencies:
    None (stdlib only).
"""

# Planning — decompose query into independent sub-queries

PLANNING_SYSTEM_PROMPT = (
    "You are a query decomposition planner. Your job is to break a complex "
    "question into independent sub-queries that can be answered separately "
    "and then combined.\n\n"
    "RULES:\n"
    "1. Each sub-query must be self-contained and answerable with a single "
    "document retrieval.\n"
    "2. Sub-queries should target specific collections from the available list.\n"
    "3. Keep sub-queries focused — one aspect per sub-query.\n"
    "4. ALWAYS produce a minimum of 2 sub-queries. By the time this prompt is called "
    "the query has already been classified as complex — find at least two distinct "
    "aspects that benefit from independent retrieval.\n"
    "5. Maximum 3 sub-queries — if you need more, consolidate.\n\n"
    "Respond with ONLY a JSON object — no markdown fences, no preamble:\n"
    '{{\n'
    '  "reasoning": "Brief explanation of your decomposition strategy",\n'
    '  "parallel_safe": true/false,\n'
    '  "sub_queries": [\n'
    '    {{\n'
    '      "query": "Specific sub-query text",\n'
    '      "collection": "target_collection_name",\n'
    '      "purpose": "What this sub-query resolves"\n'
    '    }}\n'
    "  ]\n"
    "}}"
)

PLANNING_USER_PROMPT = (
    "Original query: {query}\n\n"
    "Available collections:\n"
    "{collections}\n\n"
    "Decompose this query into sub-queries. Respond with JSON only."
)


# Sub-query rewrite — reformulate a weak sub-query using partial results (fast LLM)

REWRITE_SYSTEM_PROMPT = (
    "You are a search query optimizer. Given a search query and partial results "
    "that were retrieved with low relevance, rewrite the query to retrieve better "
    "information.\n\n"
    "RULES:\n"
    "1. Return ONLY the rewritten query — no explanation, no preamble.\n"
    "2. Keep it concise and specific.\n"
    "3. Use terminology from the partial results if helpful.\n"
    "4. Do not change the core intent of the original query."
)

REWRITE_USER_PROMPT = (
    "Original sub-query: {query}\n"
    "Purpose: {purpose}\n\n"
    "Partial results (low relevance):\n"
    "{chunk_preview}\n\n"
    "Rewrite the sub-query to find better information. Return only the rewritten query."
)


# Synthesis — generate the final answer from structured fused context (strong LLM)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. The context is organized by sub-topics.\n\n"
    "RULES:\n"
    "1. Use ONLY information from the provided context — do not fabricate.\n"
    "2. Where context is marked as a gap or limited, acknowledge it explicitly.\n"
    "3. Structure your answer to cover all sub-topics from the context.\n"
    "4. Be thorough but concise — do not repeat information.\n"
    "5. If sub-topics provide conflicting information, note the discrepancy."
)

SYNTHESIS_USER_PROMPT = (
    "Question: {query}\n\n"
    "Context:\n"
    "{structured_context}\n\n"
    "Answer the question using the provided context."
)


# Builder functions

def build_planning_prompt(
    query: str,
    collections: dict[str, str],
) -> tuple[str, str]:
    """Build planning prompts for query decomposition.

    Args:
        query: The original user query.
        collections: Dict of collection_name → description.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    collection_lines = [
        f"- {name}: {description}"
        for name, description in collections.items()
    ]
    collections_str = "\n".join(collection_lines) if collection_lines else "- default: General documents"

    user_prompt = PLANNING_USER_PROMPT.format(
        query=query,
        collections=collections_str,
    )
    return PLANNING_SYSTEM_PROMPT, user_prompt


def build_rewrite_prompt(
    query: str,
    purpose: str,
    best_chunk_content: str,
) -> tuple[str, str]:
    """Build rewrite prompts for weak sub-query reformulation.

    Called with the fast LLM. The best available chunk is used as context
    to guide the rewrite even though its relevance score is low.

    Args:
        query: The original sub-query that returned weak results.
        purpose: What the sub-query was meant to resolve.
        best_chunk_content: Content of the highest-scored chunk retrieved.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    # Limit chunk preview to avoid token waste on the rewrite call.
    preview = best_chunk_content[:400] if best_chunk_content else "No results found."

    user_prompt = REWRITE_USER_PROMPT.format(
        query=query,
        purpose=purpose,
        chunk_preview=preview,
    )
    return REWRITE_SYSTEM_PROMPT, user_prompt


def build_synthesis_prompt(
    query: str,
    structured_context: str,
) -> tuple[str, str]:
    """Build synthesis prompts for final answer generation.

    The structured_context is pre-formatted by ContextFusion with
    [Sub-query N: ...] labels grouping chunks by sub-topic.

    Args:
        query: The original user query.
        structured_context: Fused context string from ContextFusion.fuse().

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = SYNTHESIS_USER_PROMPT.format(
        query=query,
        structured_context=structured_context,
    )
    return SYNTHESIS_SYSTEM_PROMPT, user_prompt
