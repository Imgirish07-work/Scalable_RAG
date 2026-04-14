"""
ChunkQualityGate — deterministic quality evaluation for sub-query retrieval results.

Design:
    Evaluates retrieved chunks using average reranker/relevance scores and
    chunk count. Zero LLM calls — purely deterministic signal from the
    cross-encoder already applied by ChunkRetriever.

    Classification:
        strong  — enough chunks with good avg score → pass through as-is.
        weak    — chunks exist but avg score is below threshold → flag for
                  fast-LLM rewrite + one re-retrieval pass.
        failed  — zero chunks → accept as an information gap, synthesizer
                  acknowledges it; no retry attempted.

Chain of Responsibility:
    ChunkRetriever → ChunkQualityGate.evaluate()
    → AgentOrchestrator rewrites weak sub-queries (fast LLM, 1 call each).

Dependencies:
    agents.models.agent_response
"""

from agents.models.agent_response import SubQueryResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum chunk count for a sub-query to be considered strong.
_MIN_CHUNKS = 2

# Minimum average reranker/relevance score to pass as strong.
# Below this the sub-query is weak — has signal but not enough.
_MIN_AVG_SCORE = 0.25


class ChunkQualityGate:
    """Classifies each sub-query result as strong, weak, or failed.

    Weak results are flagged with is_weak=True so the orchestrator can
    rewrite the sub-query with the fast LLM and re-retrieve once.
    Failed results (0 chunks) have success=False — accepted as gaps.

    Attributes:
        _min_chunks: Minimum chunk count to be considered strong.
        _min_avg_score: Minimum average score to be considered strong.
    """

    def __init__(
        self,
        min_chunks: int = _MIN_CHUNKS,
        min_avg_score: float = _MIN_AVG_SCORE,
    ) -> None:
        """Initialize ChunkQualityGate.

        Args:
            min_chunks: Minimum chunk count threshold.
            min_avg_score: Minimum average reranker score threshold.
        """
        self._min_chunks = min_chunks
        self._min_avg_score = min_avg_score

    def evaluate(self, results: list[SubQueryResult]) -> list[SubQueryResult]:
        """Classify each sub-query result and update quality flags.

        Args:
            results: Sub-query results from ChunkRetriever.

        Returns:
            Same list with is_weak and success flags updated where needed.
        """
        evaluated = []

        for result in results:
            if not result.success:
                # Already failed at retrieval level — pass through unchanged.
                evaluated.append(result)
                continue

            if not result.chunks:
                # Zero chunks → mark as failed (information gap, no retry).
                logger.info(
                    "Quality gate | FAILED (0 chunks) | id=%s | query='%s'",
                    result.sub_query_id, result.query[:60],
                )
                evaluated.append(result.model_copy(update={
                    "success": False,
                    "failure_reason": "No chunks retrieved",
                }))
                continue

            avg_score = _avg_chunk_score(result.chunks)
            is_strong = (
                len(result.chunks) >= self._min_chunks
                and avg_score >= self._min_avg_score
            )

            if is_strong:
                logger.info(
                    "Quality gate | STRONG | id=%s | chunks=%d | avg_score=%.3f",
                    result.sub_query_id, len(result.chunks), avg_score,
                )
                evaluated.append(result)
            else:
                logger.info(
                    "Quality gate | WEAK | id=%s | chunks=%d | avg_score=%.3f "
                    "| min_chunks=%d | min_score=%.3f",
                    result.sub_query_id, len(result.chunks), avg_score,
                    self._min_chunks, self._min_avg_score,
                )
                evaluated.append(result.model_copy(update={"is_weak": True}))

        strong = sum(1 for r in evaluated if r.success and not r.is_weak)
        weak = sum(1 for r in evaluated if r.is_weak)
        failed = sum(1 for r in evaluated if not r.success)

        logger.info(
            "Quality gate complete | strong=%d | weak=%d | failed=%d | total=%d",
            strong, weak, failed, len(evaluated),
        )
        return evaluated


def _avg_chunk_score(chunks) -> float:
    """Compute average reranker_score (fallback to relevance_score) across chunks."""
    scores = [
        c.reranker_score if c.reranker_score is not None else c.relevance_score
        for c in chunks
    ]
    return sum(scores) / len(scores) if scores else 0.0
