"""
RAG prompt templates — plain Python string templates

Design:
    - Each template is a module-level constant string with {placeholders}.
    - Templates are filled via str.format() — simple, debuggable, no
      template engine dependency.
    - Separate templates for each RAG variant and sub-task. Variants
      pick the template they need — no giant multi-purpose template.
    - System prompts define the LLM's role and grounding rules.
      User prompts carry the actual query + context.

Template categories:
    1. System prompts — define LLM behavior for RAG generation
    2. User prompts — carry query + assembled context
    3. CorrectiveRAG prompts — relevance evaluation + query rewriting
    4. Utility prompts — conversation-aware query refinement

Grounding philosophy:
    - Every RAG system prompt instructs the LLM to answer ONLY from
      the provided context. If the context doesn't contain the answer,
      the LLM must say so explicitly.
    - This is the single most important prompt engineering decision
      for RAG — without it, the LLM will hallucinate confidently.

Integration:
    - BaseRAG.generate() uses SYSTEM_PROMPT + USER_PROMPT
    - CorrectiveRAG.retrieve() uses RELEVANCE_EVAL_PROMPT
    - CorrectiveRAG.retrieve() uses QUERY_REWRITE_PROMPT on retry
    - Pre-process step uses CONVERSATION_QUERY_REFINEMENT_PROMPT
      when conversation_history is present
"""


# 1. System prompts — LLM role and grounding rules

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


# 2. User prompts — query + context for generation

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


# 3. CorrectiveRAG prompts — relevance evaluation and query rewriting

RELEVANCE_EVAL_SYSTEM_PROMPT = (
    "You are a relevance evaluator. Your job is to assess whether a "
    "retrieved document is relevant to the user's query. "
    "Respond with ONLY a JSON object, no other text."
)

RELEVANCE_EVAL_USER_PROMPT = (
    "Query: {query}\n"
    "\n"
    "Document:\n"
    "{document}\n"
    "\n"
    "Rate the relevance of this document to the query on a scale of "
    "0.0 to 1.0, where:\n"
    "- 0.0 = completely irrelevant\n"
    "- 0.5 = partially relevant (some related info but not directly answering)\n"
    "- 1.0 = highly relevant (directly answers or contains key information)\n"
    "\n"
    'Respond with ONLY: {{"relevance": <score>, "reason": "<brief reason>"}}'
)

QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriter. Given a query that failed to retrieve "
    "relevant documents, rewrite it to improve retrieval. "
    "Make the query more specific, use alternative terms, or "
    "rephrase to better match document language. "
    "Respond with ONLY the rewritten query, nothing else."
)

QUERY_REWRITE_USER_PROMPT = (
    "Original query: {query}\n"
    "\n"
    "This query failed to retrieve sufficiently relevant documents. "
    "Rewrite it to improve retrieval. Consider:\n"
    "- Using more specific terminology\n"
    "- Adding relevant context or qualifiers\n"
    "- Rephrasing to match how the answer might be stated in documents\n"
    "\n"
    "Rewritten query:"
)

# CoRAG (Chain-of-RAG) — Draft + Completeness

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
    '{{\n'
    '  "is_complete": true/false,\n'
    '  "reasoning": "Brief explanation of what is resolved or missing",\n'
    '  "follow_up_query": "Specific search query to retrieve missing information "\n'
    "}}\n\n"
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


# 4. Utility prompts — conversation-aware query refinement

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

    Selects the appropriate user prompt template based on whether
    conversation history is provided.

    Args:
        query: The user's question.
        context: Assembled context string from ContextAssembler.
        conversation_history: Optional formatted conversation history.
            If provided, uses the history-aware template.

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


def build_relevance_eval_prompt(
    query: str,
    document: str,
) -> tuple[str, str]:
    """Build the prompt pair for CorrectiveRAG relevance evaluation.

    Args:
        query: The user's original question.
        document: Single chunk content to evaluate.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = RELEVANCE_EVAL_SYSTEM_PROMPT
    user = RELEVANCE_EVAL_USER_PROMPT.format(
        query=query,
        document=document,
    )
    return system, user


def build_query_rewrite_prompt(query: str) -> tuple[str, str]:
    """Build the prompt pair for CorrectiveRAG query rewriting.

    Args:
        query: The original query that failed retrieval.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = QUERY_REWRITE_SYSTEM_PROMPT
    user = QUERY_REWRITE_USER_PROMPT.format(query=query)
    return system, user


def build_conversation_refinement_prompt(
    query: str,
    conversation_history: str,
) -> tuple[str, str]:
    """Build the prompt pair for conversation-aware query refinement.

    Used in pre_process() when the RAGRequest has conversation_history.
    Resolves pronouns and references to make the query self-contained
    for retrieval.

    Args:
        query: The latest user query (may contain pronouns).
        conversation_history: Formatted previous turns.

    Returns:
        Tuple of (system_prompt, user_prompt) where system is a
            simple instruction and user carries the history + query.
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
    """Format conversation turns into a readable string.

    Limits the number of turns to prevent context window bloat.
    Takes the most recent turns (tail of the list).

    Args:
        turns: List of {"role": str, "content": str} dicts.
            Output of RAGRequest.get_chat_messages().
        max_turns: Maximum number of turns to include.
            Default 10 — enough for context, not enough to
            blow the token budget.

    Returns:
        Formatted conversation string. Empty string if no turns.
    """
    if not turns:
        return ""

    # Take the most recent turns
    recent = turns[-max_turns:]

    lines = []
    for turn in recent:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_chain_draft_prompt(context: str, query: str) -> tuple[str, str]:
    """Build system and user prompts for CoRAG draft generation.

    Args:
        context: Assembled context string from retrieved chunks.
        query: The current query (original or follow-up).

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = CHAIN_DRAFT_USER_PROMPT.format(
        context=context,
        query=query,
    )
    return CHAIN_DRAFT_SYSTEM_PROMPT, user_prompt


def build_chain_completeness_prompt(
    query: str,
    draft_answer: str,
) -> tuple[str, str]:
    """Build system and user prompts for CoRAG completeness evaluation.

    Args:
        query: The original user query (not follow-up).
        draft_answer: The draft answer to evaluate.

    Returns:
        Tuple of (system_prompt, user_prompt) for BaseLLM.chat().
    """
    user_prompt = CHAIN_COMPLETENESS_USER_PROMPT.format(
        query=query,
        draft_answer=draft_answer,
    )
    return CHAIN_COMPLETENESS_SYSTEM_PROMPT, user_prompt