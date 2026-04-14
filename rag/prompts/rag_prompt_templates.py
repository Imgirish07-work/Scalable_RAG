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


# 3. Utility prompts — conversation-aware query refinement

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


