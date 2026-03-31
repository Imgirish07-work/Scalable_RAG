"""Result verifier — quality checks on sub-query results.

Two verification modes:
  - Heuristic (default): fast, zero LLM calls, checks basic quality signals.
  - LLM-verified: one LLM call per sub-query result for deeper relevance check.

Heuristic mode is the production default. LLM mode is available for
high-stakes queries where the extra cost is justified.
"""

# stdlib
import json
import re
from typing import Optional

# internal
from agents.models.agent_response import SubQueryResult
from agents.prompts.agent_prompt_templates import build_verification_prompt
from llm.contracts.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

# heuristic thresholds
_MIN_ANSWER_LENGTH = 20
_MIN_CONFIDENCE = 0.3

# phrases that indicate non-answers
_NON_ANSWER_PATTERNS = [
    "i don't know",
    "i cannot",
    "i'm unable",
    "no information",
    "not found",
    "no relevant",
    "i could not find",
]

_VERIFICATION_MAX_TOKENS = 128


class ResultVerifier:
    """Verifies quality of sub-query results before synthesis.

    Marks each result as verified or flagged. Flagged results are
    still passed to the synthesizer — but the synthesizer is told
    which results are unreliable so it can note gaps.

    Attributes:
        _llm: Optional LLM for deep verification.
        _use_llm: Whether to use LLM verification.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        use_llm: bool = False,
    ) -> None:
        """Initialize ResultVerifier.

        Args:
            llm: LLM provider for deep verification. Required if use_llm=True.
            use_llm: Whether to use LLM-based verification.
        """
        self._llm = llm
        self._use_llm = use_llm and llm is not None

    async def verify(
        self,
        results: list[SubQueryResult],
    ) -> list[SubQueryResult]:
        """Verify all sub-query results.

        Failed sub-queries are passed through unchanged — they're
        already marked as failed. Only successful results are checked.

        Args:
            results: List of SubQueryResult from the retriever.

        Returns:
            Same list with failed results potentially updated.
        """
        verified = []
        for result in results:
            if not result.success:
                # already failed — pass through
                verified.append(result)
                continue

            is_adequate = await self._check_result(result)

            if is_adequate:
                verified.append(result)
            else:
                # downgrade to failed
                logger.warning(
                    "Sub-query '%s' failed verification: answer too weak",
                    result.sub_query_id,
                )
                verified.append(SubQueryResult(
                    sub_query_id=result.sub_query_id,
                    query=result.query,
                    collection=result.collection,
                    answer=result.answer,
                    confidence=result.confidence,
                    sources=result.sources,
                    success=False,
                    failure_reason="Failed quality verification",
                    latency_ms=result.latency_ms,
                ))

        passed = sum(1 for r in verified if r.success)
        logger.info(
            "Verification complete: %d/%d passed", passed, len(verified),
        )
        return verified

    async def _check_result(self, result: SubQueryResult) -> bool:
        """Check a single sub-query result for quality.

        Runs heuristic checks first (free). If use_llm is enabled
        and heuristics pass, runs an LLM relevance check.

        Args:
            result: SubQueryResult to verify.

        Returns:
            True if the result is adequate.
        """
        # heuristic checks (always run, zero cost)
        if not self._heuristic_check(result):
            return False

        # LLM check (optional, costs one call)
        if self._use_llm:
            return await self._llm_check(result)

        return True

    def _heuristic_check(self, result: SubQueryResult) -> bool:
        """Fast heuristic quality checks.

        Args:
            result: SubQueryResult to check.

        Returns:
            True if the result passes all heuristic checks.
        """
        # check 1 — answer is not empty or too short
        if len(result.answer.strip()) < _MIN_ANSWER_LENGTH:
            logger.debug(
                "Sub-query '%s' failed: answer too short (%d chars)",
                result.sub_query_id, len(result.answer),
            )
            return False

        # check 2 — confidence above minimum
        if result.confidence < _MIN_CONFIDENCE:
            logger.debug(
                "Sub-query '%s' failed: confidence %.3f below threshold %.3f",
                result.sub_query_id, result.confidence, _MIN_CONFIDENCE,
            )
            return False

        # check 3 — answer is not a non-answer
        answer_lower = result.answer.lower()
        for pattern in _NON_ANSWER_PATTERNS:
            if pattern in answer_lower:
                logger.debug(
                    "Sub-query '%s' failed: non-answer pattern '%s' detected",
                    result.sub_query_id, pattern,
                )
                return False

        return True

    async def _llm_check(self, result: SubQueryResult) -> bool:
        """LLM-based relevance verification.

        Asks the LLM whether the answer adequately addresses the
        sub-query's purpose. Falls back to True on failure — we
        don't want a verification error to discard good results.

        Args:
            result: SubQueryResult to verify.

        Returns:
            True if the LLM considers the answer adequate.
        """
        system_prompt, user_prompt = build_verification_prompt(
            sub_query=result.query,
            purpose="Answering the sub-query",
            answer=result.answer,
        )

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_VERIFICATION_MAX_TOKENS,
            )
            return _parse_verification_response(response.text)

        except Exception:
            logger.exception(
                "LLM verification failed for sub-query '%s', defaulting to pass",
                result.sub_query_id,
            )
            return True


def _parse_verification_response(text: str) -> bool:
    """Parse the LLM verification response.

    Expects JSON with an "is_adequate" boolean field. Falls back
    to True if parsing fails — conservative default.

    Args:
        text: Raw LLM response text.

    Returns:
        True if the answer is adequate.
    """
    cleaned = text.strip()

    # try direct parse
    parsed = _try_json_parse(cleaned)
    if parsed is None:
        stripped = re.sub(r"^```(?:json)?\s*", "", cleaned)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        parsed = _try_json_parse(stripped)

    if parsed is None:
        logger.warning("Verification parse failed, defaulting to adequate")
        return True

    is_adequate = parsed.get("is_adequate", True)
    if isinstance(is_adequate, bool):
        return is_adequate
    return str(is_adequate).lower() in ("true", "1", "yes")


def _try_json_parse(text: str) -> Optional[dict]:
    """Attempt JSON parsing, returning None on failure.

    Args:
        text: String to parse.

    Returns:
        Parsed dict or None.
    """
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return None
