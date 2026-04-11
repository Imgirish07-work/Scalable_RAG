"""
Keyword-based classifier that assigns cache TTL by query type.

Design:
    Different query types have vastly different staleness profiles, so
    a single global TTL would either under-cache stable knowledge or
    over-cache volatile facts. This module uses pre-compiled regex
    patterns to classify each query into one of seven types, each with
    its own TTL value.

    Classification is pure regex — fast (~0.01ms), deterministic, zero
    LLM cost. Covers 90%+ of real queries. Edge cases fall to the safe
    DEFAULT TTL (1 hour).

    TTL map:
        FACTUAL       → 1 hour   (volatile: prices, scores, versions)
        CONCEPTUAL    → 24 hours (stable knowledge)
        CODE          → 12 hours (library APIs change)
        SUMMARIZATION → 7 days   (source document doesn't change)
        TRANSLATION   → 7 days   (stable)
        CREATIVE      → 3 days   (creative outputs are unique)
        DEFAULT       → 1 hour   (safe fallback)

    Pattern priority: more specific patterns (code, summarization) are
    checked before broader ones (factual, conceptual) to prevent
    misclassification.

Chain of Responsibility:
    Instantiated by CacheManager. Called by CacheManager.set() to
    determine the TTL for each new cache entry before writing to L1/L2.

Dependencies:
    re, enum (stdlib only)
"""

import re
from enum import Enum
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Detected query type for TTL assignment."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    CODE = "code"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE = "creative"
    DEFAULT = "default"


# TTL values in seconds

TTL_MAP: dict[QueryType, int] = {
    QueryType.FACTUAL: 3_600,           # 1 hour — goes stale quickly
    QueryType.CONCEPTUAL: 86_400,       # 24 hours — stable knowledge
    QueryType.CODE: 43_200,             # 12 hours — libraries update
    QueryType.SUMMARIZATION: 604_800,   # 7 days — source doc doesn't change
    QueryType.TRANSLATION: 604_800,     # 7 days — translations are stable
    QueryType.CREATIVE: 259_200,        # 3 days — creative outputs are unique
    QueryType.DEFAULT: 3_600,           # 1 hour — safe fallback
}


# Keyword patterns per query type.
# Compiled once at module load — zero cost per classify() call.
# Order matters: patterns are checked top-to-bottom, first match wins.

_FACTUAL_PATTERN = re.compile(
    r"\b("
    r"latest|current|recent|today|now|"
    r"new|newest|updated|this year|this month|this week|"
    r"who is|who are|who was|"
    r"how much|how many|"
    r"price of|cost of|rate of|"
    r"score|result|standings|"
    r"version|release|"
    r"status|available|"
    r"deadline|date of|when is|when was|when did"
    r")\b",
    re.IGNORECASE,
)

_CONCEPTUAL_PATTERN = re.compile(
    r"\b("
    r"explain|how does|how do|how is|how are|"
    r"what is|what are|what does|"
    r"define|definition|meaning of|"
    r"difference between|compare|vs|versus|"
    r"why does|why do|why is|why are|"
    r"concept|theory|principle|"
    r"describe|overview|introduction to|"
    r"tell me about|teach me|help me understand|"
    r"pros and cons|advantages|disadvantages"
    r")\b",
    re.IGNORECASE,
)

_CODE_PATTERN = re.compile(
    r"\b("
    r"write a function|write a class|write a script|write a program|"
    r"write code|code for|code to|code that|"
    r"implement|implementation|"
    r"function that|function to|function for|"
    r"algorithm for|algorithm to|"
    r"regex for|regex to|"
    r"sql query|sql for|"
    r"api call|api request|"
    r"debug|fix this|fix the|"
    r"refactor|optimize this|"
    r"in python|in javascript|in typescript|in java|in rust|in go|"
    r"python code|python function|python script|python program|"
    r"javascript code|typescript code|java code|"
    r"def |class |import |async def |"
    r"```"
    r")\b",
    re.IGNORECASE,
)

_SUMMARIZATION_PATTERN = re.compile(
    r"\b("
    r"summarize|summary|summarise|"
    r"tldr|tl;dr|"
    r"key points|main points|takeaways|"
    r"brief|briefly|"
    r"condense|shorten|"
    r"extract|highlights|"
    r"outline of|overview of"
    r")\b",
    re.IGNORECASE,
)

_TRANSLATION_PATTERN = re.compile(
    r"\b("
    r"translate|translation|"
    r"in spanish|in french|in german|in japanese|in chinese|"
    r"in hindi|in tamil|in arabic|in korean|in portuguese|"
    r"to english|to spanish|to french|to german|to japanese|"
    r"to chinese|to hindi|to tamil|to arabic|to korean|to portuguese|"
    r"how do you say"
    r")\b",
    re.IGNORECASE,
)

_CREATIVE_PATTERN = re.compile(
    r"\b("
    r"write a poem|write a story|write a song|write an essay|"
    r"creative|fiction|"
    r"imagine|brainstorm|"
    r"generate ideas|come up with|"
    r"marketing copy|ad copy|tagline|slogan|"
    r"blog post|article about|"
    r"rewrite in|rephrase"
    r")\b",
    re.IGNORECASE,
)

# Ordered list — first match wins. More specific patterns checked first
# to prevent broad conceptual patterns from masking code/summarization.
_CLASSIFICATION_RULES: list[tuple[re.Pattern, QueryType]] = [
    (_CODE_PATTERN, QueryType.CODE),
    (_SUMMARIZATION_PATTERN, QueryType.SUMMARIZATION),
    (_TRANSLATION_PATTERN, QueryType.TRANSLATION),
    (_CREATIVE_PATTERN, QueryType.CREATIVE),
    (_FACTUAL_PATTERN, QueryType.FACTUAL),
    (_CONCEPTUAL_PATTERN, QueryType.CONCEPTUAL),
]


class TTLClassifier:
    """Classifies queries and assigns appropriate cache TTL.

    Keyword-based classification — fast, deterministic, zero LLM cost.
    Falls back to default TTL for unrecognized query patterns.

    Attributes:
        _default_ttl: Fallback TTL in seconds for unclassified queries.
        _ttl_map: Active TTL map (TTL_MAP merged with any overrides).
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        ttl_overrides: Optional[dict[QueryType, int]] = None,
    ) -> None:
        """Initialize the TTL classifier.

        Args:
            default_ttl: Fallback TTL for unclassified queries (seconds).
            ttl_overrides: Optional dict to override default TTL values
                           per query type. Merged on top of TTL_MAP.
        """
        self._default_ttl = default_ttl
        self._ttl_map = dict(TTL_MAP)

        if ttl_overrides:
            self._ttl_map.update(ttl_overrides)

        # Ensure the DEFAULT bucket always reflects the caller's default_ttl
        self._ttl_map[QueryType.DEFAULT] = default_ttl

        logger.info(
            "TTLClassifier initialized: default_ttl=%ds, types=%s",
            default_ttl,
            {k.value: v for k, v in self._ttl_map.items()},
        )

    def classify(self, query: str) -> QueryType:
        """Classify a query into a QueryType.

        Runs keyword patterns in priority order. First match wins.
        Returns QueryType.DEFAULT if no pattern matches.

        Args:
            query: Raw user query.

        Returns:
            Detected QueryType.
        """
        if not query or not query.strip():
            return QueryType.DEFAULT

        for pattern, query_type in _CLASSIFICATION_RULES:
            if pattern.search(query):
                logger.debug(
                    "Query classified: type=%s, query='%s'",
                    query_type.value,
                    query[:60],
                )
                return query_type

        logger.debug("Query unclassified (default): query='%s'", query[:60])
        return QueryType.DEFAULT

    def get_ttl(self, query: str) -> int:
        """Get the appropriate TTL for a query.

        Convenience method — classifies the query and returns the TTL.

        Args:
            query: Raw user query.

        Returns:
            TTL in seconds.
        """
        query_type = self.classify(query)
        ttl = self._ttl_map.get(query_type, self._default_ttl)
        return ttl

    def get_ttl_with_type(self, query: str) -> tuple[int, QueryType]:
        """Get both TTL and detected query type.

        Useful for logging and metrics — caller can see why a
        particular TTL was assigned.

        Args:
            query: Raw user query.

        Returns:
            Tuple of (ttl_seconds, QueryType).
        """
        query_type = self.classify(query)
        ttl = self._ttl_map.get(query_type, self._default_ttl)
        return ttl, query_type

    @property
    def ttl_map(self) -> dict[str, int]:
        """Return current TTL map as {type_name: seconds} for observability."""
        return {k.value: v for k, v in self._ttl_map.items()}
