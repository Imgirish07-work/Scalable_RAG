"""
Query planner — decomposes complex queries into sub-queries.

Design:
    Single LLM call with a structured JSON prompt. The LLM returns a
    decomposition plan; this module parses, validates, and caps it.
    Falls back to a single-sub-query plan if the LLM output is
    unparseable, ensuring the agent path always produces a result.

Chain of Responsibility:
    AgentOrchestrator.execute() → QueryPlanner.plan() → LLM call →
    _parse_plan_response() → DecompositionPlan returned to orchestrator.

Dependencies:
    agents.prompts.agent_prompt_templates, llm.contracts.base_llm
"""

# stdlib
import json
import re
from typing import Optional

# internal
from agents.exceptions.agent_exceptions import AgentPlanningError
from agents.models.agent_request import DecompositionPlan, SubQuery
from agents.prompts.agent_prompt_templates import build_planning_prompt
from llm.contracts.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

# Safety caps to prevent runaway decomposition.
_MAX_SUB_QUERIES = 6
_PLANNING_MAX_TOKENS = 1024


class QueryPlanner:
    """Decomposes complex queries into sub-queries via LLM.

    The planner receives a query and a collection registry (name → description),
    and produces a DecompositionPlan with targeted sub-queries.

    Attributes:
        _llm: LLM provider for the planning call.
        _collections: Registry of available collections.
    """

    def __init__(
        self,
        llm: BaseLLM,
        collections: dict[str, str],
    ) -> None:
        """Initialize QueryPlanner.

        Args:
            llm: LLM provider for decomposition.
            collections: Dict of collection_name → description.
        """
        self._llm = llm
        self._collections = collections

    async def plan(self, query: str) -> DecompositionPlan:
        """Decompose a query into sub-queries.

        Makes a single LLM call with the planning prompt. Parses
        the structured JSON response into a DecompositionPlan.

        Args:
            query: The original user query.

        Returns:
            DecompositionPlan with sub-queries and metadata.

        Raises:
            AgentPlanningError: If planning fails or produces no sub-queries.
        """
        logger.info("Planning decomposition for query: '%s'", query[:100])

        system_prompt, user_prompt = build_planning_prompt(
            query=query,
            collections=self._collections,
        )

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_PLANNING_MAX_TOKENS,
                response_mime_type="application/json",
                thinking_budget=0,
            )
        except Exception as exc:
            raise AgentPlanningError(
                message=f"Planning LLM call failed: {exc}",
                details={"query": query[:200], "error_type": type(exc).__name__},
            ) from exc

        default_collection = next(iter(self._collections), "default")
        plan = _parse_plan_response(response.text, query, default_collection)

        logger.info(
            "Plan produced %d sub-queries, parallel_safe=%s",
            len(plan.sub_queries), plan.parallel_safe,
        )
        return plan


def _parse_plan_response(
    text: str,
    original_query: str,
    default_collection: str = "default",
) -> DecompositionPlan:
    """Parse the LLM's planning response into a DecompositionPlan.

    Two-stage parsing: try clean JSON, then strip markdown fences.
    Falls back to a single-subquery plan targeting the given collection
    if parsing fails completely.

    Args:
        text: Raw LLM response text.
        original_query: The original query for fallback.
        default_collection: Collection name to use in the fallback plan.

    Returns:
        Validated DecompositionPlan.

    Raises:
        AgentPlanningError: If parsing fails and fallback is not viable.
    """
    parsed = _try_json_parse(text)

    if parsed is None:
        # Strip markdown code fences that some LLMs add despite instructions.
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        parsed = _try_json_parse(stripped)

    if parsed is None:
        logger.warning("Plan parsing failed, falling back to single sub-query")
        return _fallback_plan(original_query, default_collection)

    return _validate_plan(parsed, original_query, default_collection)


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


def _validate_plan(
    raw: dict,
    original_query: str,
    default_collection: str = "default",
) -> DecompositionPlan:
    """Validate and normalize a parsed plan response.

    Ensures sub-queries exist, caps the count, and validates
    each sub-query has required fields.

    Args:
        raw: Parsed JSON dict from the LLM.
        original_query: For fallback if sub-queries are invalid.

    Returns:
        Validated DecompositionPlan.
    """
    raw_sub_queries = raw.get("sub_queries", [])

    if not raw_sub_queries or not isinstance(raw_sub_queries, list):
        logger.warning("Plan has no sub-queries, falling back")
        return _fallback_plan(original_query, default_collection)

    # Cap to prevent excessive parallelism and token usage.
    if len(raw_sub_queries) > _MAX_SUB_QUERIES:
        logger.warning(
            "Plan produced %d sub-queries, capping at %d",
            len(raw_sub_queries), _MAX_SUB_QUERIES,
        )
        raw_sub_queries = raw_sub_queries[:_MAX_SUB_QUERIES]

    sub_queries = []
    for sq in raw_sub_queries:
        if not isinstance(sq, dict):
            continue
        query_text = str(sq.get("query", "")).strip()
        collection = str(sq.get("collection", "default")).strip()
        purpose = str(sq.get("purpose", ""))

        if not query_text:
            continue

        sub_queries.append(SubQuery(
            query=query_text,
            collection=collection,
            purpose=purpose,
            variant=sq.get("variant"),
        ))

    if not sub_queries:
        logger.warning("No valid sub-queries after validation, falling back")
        return _fallback_plan(original_query, default_collection)

    reasoning = str(raw.get("reasoning", ""))
    parallel_safe = raw.get("parallel_safe", True)
    if not isinstance(parallel_safe, bool):
        parallel_safe = str(parallel_safe).lower() in ("true", "1", "yes")

    return DecompositionPlan(
        sub_queries=sub_queries,
        reasoning=reasoning,
        parallel_safe=parallel_safe,
    )


def _fallback_plan(query: str, collection: str = "default") -> DecompositionPlan:
    """Create a single-subquery fallback plan.

    Used when planning fails — routes the original query to the
    given collection as a single RAG call. This ensures the
    agent path always produces a result, even if planning breaks.

    Args:
        query: The original user query.
        collection: Collection name to target in the fallback sub-query.

    Returns:
        DecompositionPlan with one sub-query.
    """
    return DecompositionPlan(
        sub_queries=[
            SubQuery(
                query=query,
                collection=collection,
                purpose="Fallback — original query as single sub-query",
            ),
        ],
        reasoning="Planning failed — using original query as fallback",
        parallel_safe=True,
    )
