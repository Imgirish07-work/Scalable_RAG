"""
Prompt templates for the agent layer.

Design:
    Three prompt pairs (planning, verification, synthesis), each as a
    module-level constant plus a builder function. Constants hold the
    static system prompt; builder functions inject runtime values into
    the user prompt. All builders return (system_prompt, user_prompt)
    tuples compatible with BaseLLM.chat().

Chain of Responsibility:
    QueryPlanner calls build_planning_prompt() → ResultVerifier calls
    build_verification_prompt() → AnswerSynthesizer calls
    build_synthesis_prompt().

Dependencies:
    None (stdlib only).
"""

# Planning — decompose query into sub-queries

PLANNING_SYSTEM_PROMPT = (
    "You are a query decomposition planner. Your job is to break a complex "
    "question into independent sub-queries that can be answered separately "
    "and then combined.\n\n"
    "RULES:\n"
    "1. Each sub-query must be self-contained and answerable with a single "
    "document retrieval.\n"
    "2. Sub-queries should target specific collections from the available list.\n"
    "3. Keep sub-queries focused — one aspect per sub-query.\n"
    "4. If the query is already simple enough, return a single sub-query.\n"
    "5. Maximum 6 sub-queries — if you need more, consolidate.\n\n"
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


# Verification — check whether a sub-query answer is adequate

VERIFICATION_SYSTEM_PROMPT = (
    "You are a result quality verifier. Evaluate whether a sub-query answer "
    "adequately addresses the question it was meant to answer.\n\n"
    "Respond with ONLY a JSON object — no markdown fences, no preamble:\n"
    '{{\n'
    '  "is_adequate": true/false,\n'
    '  "reasoning": "Brief explanation"\n'
    "}}"
)

VERIFICATION_USER_PROMPT = (
    "Sub-query: {sub_query}\n"
    "Purpose: {purpose}\n\n"
    "Answer provided:\n"
    "{answer}\n\n"
    "Is this answer adequate for the stated purpose? Respond with JSON only."
)


# Synthesis — combine sub-query results into a final answer

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an answer synthesizer. Combine multiple sub-query results into "
    "a single coherent, comprehensive answer to the original question.\n\n"
    "RULES:\n"
    "1. Use ONLY information from the provided sub-query results.\n"
    "2. If a sub-query failed or returned incomplete information, explicitly "
    "note the gap — do NOT fabricate information to fill it.\n"
    "3. Structure the answer logically — group related information together.\n"
    "4. Be concise but thorough. Do not repeat information across sections.\n"
    "5. If sub-query results conflict, note the discrepancy."
)

SYNTHESIS_USER_PROMPT = (
    "Original query: {query}\n\n"
    "Sub-query results:\n"
    "{sub_results}\n\n"
    "Synthesize a comprehensive answer to the original query."
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
    collection_lines = []
    for name, description in collections.items():
        collection_lines.append(f"- {name}: {description}")
    collections_str = "\n".join(collection_lines) if collection_lines else "- default: General documents"

    user_prompt = PLANNING_USER_PROMPT.format(
        query=query,
        collections=collections_str,
    )
    return PLANNING_SYSTEM_PROMPT, user_prompt


def build_verification_prompt(
    sub_query: str,
    purpose: str,
    answer: str,
) -> tuple[str, str]:
    """Build verification prompts for sub-query result checking.

    Args:
        sub_query: The sub-query that was executed.
        purpose: What the sub-query was meant to resolve.
        answer: The answer produced by the RAG pipeline.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = VERIFICATION_USER_PROMPT.format(
        sub_query=sub_query,
        purpose=purpose,
        answer=answer,
    )
    return VERIFICATION_SYSTEM_PROMPT, user_prompt


def build_synthesis_prompt(
    query: str,
    sub_results: list[dict],
) -> tuple[str, str]:
    """Build synthesis prompts for combining sub-query results.

    Args:
        query: The original user query.
        sub_results: List of dicts with sub-query info and answers.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    result_lines = []
    for i, result in enumerate(sub_results, 1):
        status = "SUCCESS" if result.get("success") else "FAILED"
        line = (
            f"[{i}] ({status}) Sub-query: {result.get('query', '')}\n"
            f"    Purpose: {result.get('purpose', '')}\n"
            f"    Answer: {result.get('answer', 'No answer available')}"
        )
        result_lines.append(line)

    sub_results_str = "\n\n".join(result_lines)

    user_prompt = SYNTHESIS_USER_PROMPT.format(
        query=query,
        sub_results=sub_results_str,
    )
    return SYNTHESIS_SYSTEM_PROMPT, user_prompt
