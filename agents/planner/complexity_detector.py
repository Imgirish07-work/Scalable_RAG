"""Rule-based complexity detector.

Decides whether a query needs agent decomposition or can go directly
to a single RAG variant. Uses heuristic signals — no LLM calls.

The detector is deliberately conservative: it's better to send a
borderline query through the cheaper RAG path than to waste tokens
on unnecessary decomposition. False negatives (complex query goes
to RAG) produce a worse answer. False positives (simple query goes
to agent) waste ~2 extra LLM calls but still produce a good answer.
"""

# stdlib
import re

# internal
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Heuristic thresholds ──

# queries shorter than this are almost never complex enough for decomposition
_MIN_QUERY_LENGTH = 40

# multiple explicit conjunctions suggest multi-part questions
_CONJUNCTION_PATTERN = re.compile(
    r"\b(and also|as well as|in addition to|along with|furthermore)\b",
    re.IGNORECASE,
)

# comparison language strongly signals multi-entity queries
_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|contrast"
    r"|across|between|relative to|compared to|how does .+ stack up)\b",
    re.IGNORECASE,
)

# multi-part question markers
_MULTI_QUESTION_PATTERN = re.compile(
    r"(\?.*\?)"
    r"|(\b(firstly|secondly|thirdly)\b)"
    r"|(1\.|2\.|3\.)"
    r"|(\band\b.*\band\b)",
    re.IGNORECASE,
)

# signals that the query targets multiple entities
_MULTI_ENTITY_PATTERN = re.compile(
    r"\b(each|every|all|both|respective|individually)\b",
    re.IGNORECASE,
)

# minimum score to trigger agent decomposition
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

    # signal 1 — comparison language (strong signal, +3)
    if _COMPARISON_PATTERN.search(query):
        score += 3
        signals.append("comparison")

    # signal 2 — explicit conjunctions suggesting multiple parts (+2)
    if _CONJUNCTION_PATTERN.search(query):
        score += 2
        signals.append("conjunction")

    # signal 3 — multiple questions or enumerated parts (+2)
    if _MULTI_QUESTION_PATTERN.search(query):
        score += 2
        signals.append("multi_question")

    # signal 4 — multi-entity language (+2)
    if _MULTI_ENTITY_PATTERN.search(query):
        score += 2
        signals.append("multi_entity")

    # signal 5 — query length suggests complexity (+1)
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
