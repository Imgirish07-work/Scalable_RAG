"""
Query normalization chain — Chain of Responsibility pattern.

Transforms raw user queries into a canonical form before cache key
generation. The goal is to maximize cache hits by ensuring that
semantically identical queries produce identical cache keys.


Step execution order matters:
    1. WhitespaceNormalizer  — collapse spaces, strip edges
    2. CaseNormalizer        — lowercase everything
    3. PunctuationNormalizer — strip trailing punctuation
    4. UnicodeNormalizer     — normalize unicode to NFC form
    5. (Applied externally)  — parameter canonicalization in make_key()

Usage:
    chain = QueryNormalizerChain()
    normalized = chain.normalize("  What   is  RAG? ")
    # Returns: "what is rag"

    fingerprint = chain.build_cache_fingerprint(
        query="What is RAG?",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        system_prompt_hash="abc123..."
    )
    # Returns: "what is rag|gemini-2.5-flash|0.0|abc123..."
"""

import re
import unicodedata
from typing import Optional

from utils.logger import get_logger
from cache.normalizers.base_normalizer import BaseNormalizer

logger = get_logger(__name__)

class WhitespaceNormalizer(BaseNormalizer):
    """Collapse multiple whitespace characters into single spaces and strip edges."""

    @property
    def name(self) -> str:
        return "whitespace" 

    def normalize(self, text: str) -> str:
        if not text:
            return text

        result = re.sub(r"\s+", " ", text).strip()
        return result

class CaseNormalizer(BaseNormalizer):
    """Convert text to lowercase for case-insensitive matching."""

    @property
    def name(self) -> str:
        return "case"

    def normalize(self, text: str) -> str:
        if not text:
            return text

        return text.lower()
    
class PunctuationNormalizer(BaseNormalizer):
    """Strip trailing punctuation that doesn't change query semantics."""

    TRAILING_PATTERN = re.compile(r"[?.!,;:\s]+$")

    @property
    def name(self) -> str:
        return "punctuation"

    def normalize(self, text: str) -> str:
        if not text:
            return text

        result = re.sub(self.TRAILING_PATTERN, "", text)

        return result if result else text

class UnicodeNormalizer(BaseNormalizer):
    """Normalize Unicode to NFC form for consistent byte representation.

    Different Unicode representations of the same character produce
    different SHA-256 hashes. NFC normalization ensures that
    "café" (composed) and "café" (decomposed: e + combining acute)
    produce the same hash.

    NFC (Canonical Decomposition, followed by Canonical Composition) is
    the standard form used by most systems and databases.

    Also strips zero-width characters that are invisible but affect hashing:
        Zero-width space (U+200B)
        Zero-width non-joiner (U+200C)
        Zero-width joiner (U+200D)
        Byte order mark (U+FEFF)
    """

    ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")

    @property
    def name(self) -> str:
        return "unicode"

    def normalize(self, text: str) -> str:
        if not text:
            return text

        # First normalize to NFC form
        normalized = unicodedata.normalize("NFC", text)

        # Then remove zero-width characters
        result = re.sub(self.ZERO_WIDTH_PATTERN, "", normalized)

        return result
    
class QueryNormalizerChain:
    """Composes normalization steps into a sequential pipeline.

    The chain runs each step in order on the query text.
    Steps are stateless — the chain can be shared across threads safely.

    The chain also provides build_cache_fingerprint() which combines
    the normalized query with model parameters into a single string
    suitable for hashing by the exact strategy.

    Attributes:
        _steps: Ordered list of normalizer steps.
    """

    PARAM_SEPARATOR = "|"

    def __init__(self, steps: Optional[list[BaseNormalizer]] = None):
        self._steps = steps or self._build_default_chain()
        step_names = [s.name for s in self._steps]
        logger.info(
            "QueryNormalizerChain initialized: steps=%s", step_names
        )

    def _build_default_chain(self) -> list[BaseNormalizer]:
        """Build the default production normalization chain.

        Order matters:
            1. Whitespace first — clean foundation for all other steps
            2. Case second — must happen before punctuation (no edge cases)
            3. Punctuation third — strip trailing noise after case normalization
            4. Unicode last — final byte-level canonicalization before hashing
        """
        return [
            WhitespaceNormalizer(),
            CaseNormalizer(),
            PunctuationNormalizer(),
            UnicodeNormalizer(),
        ]

    @property
    def steps(self) -> list[BaseNormalizer]:
        """Read-only access to the step list."""
        return list(self._steps)
    
    def normalize(self, text: str) -> str:
        """Run the full normalization chain on a query string.

        Each step transforms the text in sequence. If any step fails
        (which shouldn't happen — steps catch their own errors), the
        chain logs a warning and continues with the text as-is.

        Args:
            text: Raw user query.

        Returns:
            Fully normalized query string.
        """
        if not text:
            return text

        result = text   
        for step in self._steps:
            try:
                result = step.normalize(result)
            except Exception as e:
                logger.exception(
                    "Normalizer step '%s' failed, skipping: input='%s'",
                    step.name,
                    result[:100],
                )
        return result
    
    def build_cache_fingerprint(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> str:
        """Build the canonical string that gets hashed into a cache key.
        Args:
            query: Raw user query (will be normalized).
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of system prompt.

        Returns:
            Canonical fingerprint string like:
            "what is rag|gemini-2.5-flash|0.0|abc123def456..."
        """

        normalized_query = self.normalize(query)
        nanonical_model = model_name.strip().lower()
        canonical_temp = f"{temperature:.1f}" 

        parts = [
            normalized_query,
            nanonical_model,
            canonical_temp,
        ]

        if system_prompt_hash:
            parts.append(system_prompt_hash)

        fingerprint  =  self.PARAM_SEPARATOR.join(parts)

        logger.debug(
            "Cache fingerprint built: query='%s' → fingerprint='%s'",
            query[:80],
            fingerprint[:120],
        )

        return fingerprint