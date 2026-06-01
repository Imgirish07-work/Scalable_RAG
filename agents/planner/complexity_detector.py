"""Rule-based complexity detector; additive scoring heuristic routes queries to the agent layer."""

import re

from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_QUERY_LENGTH = 40

_CONJUNCTION_PATTERN = re.compile(
    r"\b(and also|as well as|in addition to|along with|furthermore)\b",
    re.IGNORECASE,
)

_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|contrast"
    r"|across|between|relative to|compared to|how does .+ stack up)\b",
    re.IGNORECASE,
)

_MULTI_QUESTION_PATTERN = re.compile(
    r"(\?.*\?)"
    r"|(\b(firstly|secondly|thirdly)\b)"
    r"|(1\.|2\.|3\.)"
    r"|(\band\b.*\band\b)",
    re.IGNORECASE,
)

_MULTI_ENTITY_PATTERN = re.compile(
    r"\b(each|every|all|both|respective|individually)\b",
    re.IGNORECASE,
)

_COMPLEXITY_THRESHOLD = 3


def should_decompose(query: str) -> bool:
    """Determine whether a query needs agent decomposition."""
    if len(query) < _MIN_QUERY_LENGTH:
        return False

    score = 0
    signals = []

    if _COMPARISON_PATTERN.search(query):
        score += 3
        signals.append("comparison")

    if _CONJUNCTION_PATTERN.search(query):
        score += 2
        signals.append("conjunction")

    if _MULTI_QUESTION_PATTERN.search(query):
        score += 2
        signals.append("multi_question")

    if _MULTI_ENTITY_PATTERN.search(query):
        score += 2
        signals.append("multi_entity")

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
