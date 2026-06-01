"""Decomposes complex queries into sub-queries via a single structured JSON LLM call."""

import json
import re
from typing import Optional

from agents.exceptions.agent_exceptions import AgentPlanningError
from agents.models.agent_request import DecompositionPlan, SubQuery
from agents.prompts.agent_prompt_templates import build_planning_prompt
from llm.contracts.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)

# must match the prompt's stated maximum
_MAX_SUB_QUERIES = 3
_PLANNING_MAX_TOKENS = 1024


class QueryPlanner:
    """Decomposes complex queries into sub-queries via LLM."""

    def __init__(
        self,
        llm: BaseLLM,
        collections: dict[str, str],
    ) -> None:
        self._llm = llm
        self._collections = collections

    async def plan(self, query: str) -> DecompositionPlan:
        """Decompose a query into sub-queries."""
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
    """Parse the LLM's planning response into a DecompositionPlan."""
    parsed = _try_json_parse(text)

    if parsed is None:
        # strip markdown code fences that some llms add despite instructions
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        parsed = _try_json_parse(stripped)

    if parsed is None:
        logger.warning("Plan parsing failed | falling back to single sub-query")
        return _fallback_plan(original_query, default_collection)

    return _validate_plan(parsed, original_query, default_collection)


def _try_json_parse(text: str) -> Optional[dict]:
    """Attempt JSON parsing, returning None on failure."""
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
    """Validate and normalize a parsed plan response."""
    raw_sub_queries = raw.get("sub_queries", [])

    if not raw_sub_queries or not isinstance(raw_sub_queries, list):
        logger.warning("Plan has no sub-queries | falling back to single sub-query")
        return _fallback_plan(original_query, default_collection)

    # cap to prevent excessive parallelism and token usage
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
        logger.warning("No valid sub-queries after validation | falling back to single sub-query")
        return _fallback_plan(original_query, default_collection)

    # enforce min 2 so agent path does not silently behave like simplerag
    if len(sub_queries) == 1:
        logger.warning(
            "Plan produced only 1 sub-query for complex query | "
            "adding supplementary context sub-query | query='%s'",
            original_query[:80],
        )
        sub_queries.append(SubQuery(
            query=f"Background context and key concepts related to: {original_query}",
            collection=sub_queries[0].collection,
            purpose="Supplementary context to broaden the primary answer",
        ))

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
    """Create a single-subquery fallback plan."""
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
