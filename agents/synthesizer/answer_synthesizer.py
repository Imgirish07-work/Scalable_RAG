"""
Answer synthesizer — combines sub-query results into a final answer.

Design:
    Single LLM call that receives the original query and all sub-query
    results (successful and failed). The prompt instructs the LLM to
    note gaps from failed sub-queries explicitly rather than hallucinating
    to fill them. Only successful results contribute to synthesis, but
    failed results appear in the prompt so the LLM can acknowledge gaps.

Chain of Responsibility:
    AgentOrchestrator.execute() → AnswerSynthesizer.synthesize()
    → LLM call → answer string returned to orchestrator.

Dependencies:
    agents.prompts.agent_prompt_templates, llm.contracts.base_llm
"""

# internal
from agents.exceptions.agent_exceptions import AgentSynthesisError
from agents.models.agent_response import SubQueryResult
from agents.prompts.agent_prompt_templates import build_synthesis_prompt
from llm.contracts.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

_SYNTHESIS_MAX_TOKENS = 2048


class AnswerSynthesizer:
    """Combines sub-query results into a single coherent answer.

    Does not retrieve new information — only combines existing
    sub-query answers. The prompt explicitly instructs the LLM
    to note gaps from failed sub-queries rather than hallucinating.

    Attributes:
        _llm: LLM provider for the synthesis call.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """Initialize AnswerSynthesizer.

        Args:
            llm: LLM provider for synthesis.
        """
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        sub_results: list[SubQueryResult],
    ) -> str:
        """Synthesize sub-query results into a final answer.

        Args:
            query: The original user query.
            sub_results: Verified sub-query results.

        Returns:
            Synthesized answer string.

        Raises:
            AgentSynthesisError: If synthesis fails or produces empty output.
        """
        # Guard — nothing to synthesize if every sub-query failed.
        successful = [r for r in sub_results if r.success]
        if not successful:
            logger.warning("No successful sub-results to synthesize")
            raise AgentSynthesisError(
                message="All sub-queries failed — nothing to synthesize",
                details={
                    "total_sub_queries": len(sub_results),
                    "failures": [r.failure_reason for r in sub_results],
                },
            )

        # Include all results (including failures) so the LLM can note gaps.
        formatted_results = _format_sub_results(sub_results)
        system_prompt, user_prompt = build_synthesis_prompt(
            query=query,
            sub_results=formatted_results,
        )

        logger.info(
            "Synthesizing %d results (%d successful, %d failed)",
            len(sub_results), len(successful),
            len(sub_results) - len(successful),
        )

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=_SYNTHESIS_MAX_TOKENS,
            )
        except Exception as exc:
            raise AgentSynthesisError(
                message=f"Synthesis LLM call failed: {exc}",
                details={"query": query[:200], "error_type": type(exc).__name__},
            ) from exc

        answer = response.text.strip()
        if not answer:
            raise AgentSynthesisError(
                message="Synthesis produced empty answer",
                details={"query": query[:200]},
            )

        logger.info("Synthesis complete, answer length=%d chars", len(answer))
        return answer


def _format_sub_results(sub_results: list[SubQueryResult]) -> list[dict]:
    """Format sub-query results for the synthesis prompt.

    Failed results are represented with a neutral placeholder so internal
    system details (error types, failure reasons) are never exposed in the
    user-facing answer.

    Args:
        sub_results: List of SubQueryResult.

    Returns:
        List of dicts with query, purpose, answer, success fields.
    """
    formatted = []
    for result in sub_results:
        entry = {
            "query": result.query,
            "purpose": result.purpose,
            "answer": result.answer if result.success else "[Information not available for this aspect of the query.]",
            "success": result.success,
        }
        formatted.append(entry)
    return formatted
