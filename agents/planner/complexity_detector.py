"""
Rule-based complexity detector.

Design:
    Additive scoring heuristic — each signal adds points. When the
    total meets _COMPLEXITY_THRESHOLD the query is routed to the agent
    layer. Deliberately conservative: a false positive (simple query
    decomposed unnecessarily) wastes ~2 LLM calls but still yields a
    good answer; a false negative (complex query sent to single RAG)
    produces a worse answer.

Chain of Responsibility:
    RAGPipeline._execute_query() calls should_decompose() before deciding
    whether to invoke AgentOrchestrator. No LLM calls are made here.

Dependencies:
    re (stdlib only).
"""

# stdlib
import re

# internal
from utils.logger import get_logger

logger = get_logger(__name__)

# Heuristic thresholds

# Queries shorter than this are almost never complex enough for decomposition.
_MIN_QUERY_LENGTH = 40

# Multiple explicit conjunctions suggest multi-part questions.
_CONJUNCTION_PATTERN = re.compile(
    r"\b(and also|as well as|in addition to|along with|furthermore)\b",
    re.IGNORECASE,
)

# Comparison language strongly signals multi-entity queries.
_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|contrast"
    r"|across|between|relative to|compared to|how does .+ stack up)\b",
    re.IGNORECASE,
)

# Multi-part question markers: double question marks, ordinals, numbered lists, repeated conjunctions.
_MULTI_QUESTION_PATTERN = re.compile(
    r"(\?.*\?)"
    r"|(\b(firstly|secondly|thirdly)\b)"
    r"|(1\.|2\.|3\.)"
    r"|(\band\b.*\band\b)",
    re.IGNORECASE,
)

# Signals that the query targets multiple entities simultaneously.
_MULTI_ENTITY_PATTERN = re.compile(
    r"\b(each|every|all|both|respective|individually)\b",
    re.IGNORECASE,
)

# Minimum score to trigger agent decomposition.
_COMPLEXITY_THRESHOLD = 3


def should_decompose(query: str) -> bool:
    """Determine whether a query needs agent decomposition.

    Uses a simple scoring system: each heuristic signal adds points.
    If the total score meets the threshold, the query is routed to
    the agent layer.

    Args:
        query: The user's original query text.

    Returns:
        True if the query should be decomposed by the agent planner.
    """
    if len(query) < _MIN_QUERY_LENGTH:
        return False

    score = 0
    signals = []

    # Comparison language — strong signal (+3).
    if _COMPARISON_PATTERN.search(query):
        score += 3
        signals.append("comparison")

    # Explicit conjunctions suggesting multiple distinct parts (+2).
    if _CONJUNCTION_PATTERN.search(query):
        score += 2
        signals.append("conjunction")

    # Multiple questions or enumerated parts (+2).
    if _MULTI_QUESTION_PATTERN.search(query):
        score += 2
        signals.append("multi_question")

    # Multi-entity language (+2).
    if _MULTI_ENTITY_PATTERN.search(query):
        score += 2
        signals.append("multi_entity")

    # Long queries often contain multiple implicit sub-questions (+1).
    if len(query) > 150:
        score += 1
        signals.append("long_query")

    needs_decomposition = score >= _COMPLEXITY_THRESHOLD

    logger.info(
        "Complexity check: score=%d threshold=%d decompose=%s signals=%s query='%s'",
        score, _COMPLEXITY_THRESHOLD, needs_decomposition,
        signals, query[:80],
    )

    return needs_decomposition
