"""
Prompt templates for RAG generation, evaluation, and query refinement.

Design:
    Each template is a module-level string constant with {placeholders}
    filled via str.format(). No template engine dependency — plain Python,
    easy to diff and debug. Separate templates per task: generation, relevance
    evaluation, query rewriting, and conversation refinement. Builder
    functions pair each system prompt with its user prompt and handle
    placeholder substitution, keeping call sites clean.

    Grounding rule: every RAG system prompt instructs the LLM to answer
    ONLY from the provided context. Without this rule the LLM hallucinates
    confidently — it is the single most important RAG prompt engineering
    decision.

Chain of Responsibility:
    build_rag_prompt() called by BaseRAG.generate()
    build_conversation_refinement_prompt() called by BaseRAG.pre_process()
    build_chain_draft_prompt() / build_chain_completeness_prompt() called by ChainRAG

Dependencies:
    None (stdlib only).
"""


# 1. System prompts — define LLM role and grounding rules

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on "
    "the provided context. Follow these rules:\n"
    "\n"
    "1. Answer ONLY using information from the provided context.\n"
    "2. If the context does not contain enough information to answer the "
    "question, say: \"I don't have enough information in the provided "
    "documents to answer this question.\"\n"
    "3. Do NOT use prior knowledge or make assumptions beyond what the "
    "context states.\n"
    "4. When referencing information, cite the source number "
    "(e.g., [Source 1], [Source 2]).\n"
    "5. Be concise and direct. Do not repeat the question.\n"
    "6. If multiple sources provide conflicting information, acknowledge "
    "the conflict and present both perspectives with their sources."
)

RAG_SYSTEM_PROMPT_CONCISE = (
    "Answer the question using ONLY the provided context. "
    "If the context lacks the answer, say so. "
    "Cite sources as [Source N]. Be concise."
)


# 2. User prompts — carry query and assembled context

RAG_USER_PROMPT = (
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {query}\n"
    "\n"
    "Answer the question based on the context above."
)

RAG_USER_PROMPT_WITH_HISTORY = (
    "Conversation so far:\n"
    "{conversation_history}\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {query}\n"
    "\n"
    "Answer the question based on the context above, considering "
    "the conversation history for context."
)


# 3. ChainRAG prompts — draft generation and completeness evaluation

CHAIN_DRAFT_SYSTEM_PROMPT = (
    "You are a precise information extractor. Generate a concise draft answer "
    "using ONLY the provided context.\n\n"
    "CRITICAL RULES:\n"
    "1. Use only information explicitly stated in the context.\n"
    "2. If the context references external documents, regulations, appendices, "
    "or sections that are NOT included in the context, explicitly flag them. "
    'Example: "As referenced in Regulation 7.3 (not available in current context)."\n'
    "3. Keep the draft concise — focus on directly answering the query.\n"
    "4. Do NOT fabricate information to fill gaps."
)

CHAIN_DRAFT_USER_PROMPT = (
    "Context:\n"
    "{context}\n\n"
    "Query: {query}\n\n"
    "Generate a concise draft answer. Explicitly flag any references to external "
    "documents or sections not present in the context above."
)

CHAIN_COMPLETENESS_SYSTEM_PROMPT = (
    "You are a completeness evaluator. Assess whether a draft answer fully "
    "resolves the user's query or has unresolved references.\n\n"
    "Respond with ONLY a JSON object — no markdown fences, no preamble:\n"
    '{\n'
    '  "is_complete": true/false,\n'
    '  "reasoning": "Brief explanation of what is resolved or missing",\n'
    '  "follow_up_query": "Specific search query to retrieve missing information "\n'
    "}\n\n"
    "RULES:\n"
    '1. Set "is_complete" to true ONLY if the draft fully answers the query '
    "with no dangling references.\n"
    '2. Set "follow_up_query" to an empty string if is_complete is true.\n'
    '3. Make "follow_up_query" specific and targeted — it will be used for '
    "vector similarity search.\n"
    "4. Focus on unresolved references to regulations, appendices, sections, "
    "or external documents mentioned but not included."
)

CHAIN_COMPLETENESS_USER_PROMPT = (
    "Original query: {query}\n\n"
    "Draft answer:\n"
    "{draft_answer}\n\n"
    "Evaluate completeness and respond with JSON only."
)


# 4. ChainRAG combined prompt — draft + completeness in one LLM call

CHAIN_COMBINED_SYSTEM_PROMPT = (
    "You are a precise answer generator and completeness evaluator.\n\n"
    "Given a query and retrieved context, do TWO things in ONE response:\n"
    "1. Write a concise draft answer using ONLY the provided context.\n"
    "2. Evaluate whether your draft fully resolves the query.\n\n"
    "Respond with ONLY a JSON object — no markdown fences, no preamble:\n"
    "{\n"
    '  "draft": "Your concise answer here",\n'
    '  "is_complete": true/false,\n'
    '  "reasoning": "Brief explanation of what is resolved or missing",\n'
    '  "follow_up_query": "Targeted search query for missing info, or empty string"\n'
    "}\n\n"
    "RULES:\n"
    "1. Answer ONLY from the provided context. Flag external references explicitly.\n"
    '2. Set "is_complete" to true ONLY if the draft fully answers the query.\n'
    '3. "follow_up_query" must be an empty string when is_complete is true.\n'
    '4. "follow_up_query" must be a SHORT, SPECIFIC factual question — at most 12 words.\n'
    "   - Identify exactly ONE fact or definition that is missing from your draft.\n"
    "   - Write a precise retrieval question for that single missing fact.\n"
    "   - Do NOT write topic phrases or broad expansions.\n"
    '   - Good: "What replication factor does Cassandra use by default?"\n'
    '   - Bad: "distributed replication and consistency trade-offs"'
)

CHAIN_COMBINED_USER_PROMPT = (
    "Context:\n"
    "{context}\n\n"
    "Query: {query}\n\n"
    "Generate the draft answer and evaluate completeness. Respond with JSON only."
)


# 5. Utility prompts — conversation-aware query refinement

CONVERSATION_QUERY_REFINEMENT_PROMPT = (
    "Given the conversation history and the latest user query, "
    "rewrite the query to be self-contained (resolve pronouns like "
    "'it', 'that', 'they' and references to previous turns). "
    "If the query is already self-contained, return it unchanged. "
    "Respond with ONLY the rewritten query.\n"
    "\n"
    "Conversation history:\n"
    "{conversation_history}\n"
    "\n"
    "Latest query: {query}\n"
    "\n"
    "Self-contained query:"
)


# Template builder functions

def build_rag_prompt(
    query: str,
    context: str,
    conversation_history: str | None = None,
) -> tuple[str, str]:
    """Build the system + user prompt pair for RAG generation.

    Selects the history-aware template when conversation_history is provided,
    otherwise uses the plain template.

    Args:
        query: The user's question.
        context: Assembled context string from ContextAssembler.
        conversation_history: Optional formatted conversation history string.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = RAG_SYSTEM_PROMPT

    if conversation_history:
        user = RAG_USER_PROMPT_WITH_HISTORY.format(
            conversation_history=conversation_history,
            context=context,
            query=query,
        )
    else:
        user = RAG_USER_PROMPT.format(
            context=context,
            query=query,
        )

    return system, user


def build_conversation_refinement_prompt(
    query: str,
    conversation_history: str,
) -> tuple[str, str]:
    """Build the prompt pair for conversation-aware query refinement.

    Used in BaseRAG.pre_process() when the RAGRequest has conversation_history.
    Resolves pronouns and trailing references to make the query self-contained
    for retrieval embedding.

    Args:
        query: The latest user query, possibly containing pronouns or references.
        conversation_history: Formatted string of previous conversation turns.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = (
        "Rewrite the query to be self-contained. "
        "Respond with ONLY the rewritten query."
    )
    user = CONVERSATION_QUERY_REFINEMENT_PROMPT.format(
        conversation_history=conversation_history,
        query=query,
    )
    return system, user


def format_conversation_history(
    turns: list[dict],
    max_turns: int = 10,
) -> str:
    """Format conversation turns into a human-readable string.

    Takes the most recent turns to prevent context window bloat when
    conversation history is long.

    Args:
        turns: List of {"role": str, "content": str} dicts.
            Output of RAGRequest.get_chat_messages().
        max_turns: Maximum number of turns to include. Default 10 —
            enough for context without blowing the token budget.

    Returns:
        Formatted conversation string. Empty string if no turns.
    """
    if not turns:
        return ""

    # Use only the most recent turns to cap token usage
    recent = turns[-max_turns:]

    lines = []
    for turn in recent:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_chain_draft_prompt(context: str, query: str) -> tuple[str, str]:
    """Build system and user prompts for ChainRAG draft generation.

    Args:
        context: Assembled context string from accumulated retrieved chunks.
        query: The current query (original or follow-up hop query).

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = CHAIN_DRAFT_USER_PROMPT.format(
        context=context,
        query=query,
    )
    return CHAIN_DRAFT_SYSTEM_PROMPT, user_prompt


def build_chain_combined_prompt(context: str, query: str) -> tuple[str, str]:
    """Build system and user prompts for the combined ChainRAG draft + completeness call.

    Replaces the two-call pattern (build_chain_draft_prompt + build_chain_completeness_prompt)
    with a single LLM call that returns draft, is_complete, reasoning, and follow_up_query.

    Args:
        context: Assembled context string from accumulated retrieved chunks.
        query: The original user query.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = CHAIN_COMBINED_USER_PROMPT.format(context=context, query=query)
    return CHAIN_COMBINED_SYSTEM_PROMPT, user_prompt


def build_chain_completeness_prompt(
    query: str,
    draft_answer: str,
) -> tuple[str, str]:
    """Build system and user prompts for ChainRAG completeness evaluation.

    Args:
        query: The original user query (not a follow-up hop query).
        draft_answer: The draft answer to evaluate for completeness.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = CHAIN_COMPLETENESS_USER_PROMPT.format(
        query=query,
        draft_answer=draft_answer,
    )
    return CHAIN_COMPLETENESS_SYSTEM_PROMPT, user_prompt
