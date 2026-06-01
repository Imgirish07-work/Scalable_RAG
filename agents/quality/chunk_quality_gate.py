"""Deterministic strong/weak/failed classifier for sub-query retrieval results."""

from agents.models.agent_response import SubQueryResult
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_CHUNKS = 2

# bge reranker base: relevant 0.40-0.95, moderate 0.25-0.55, irrelevant 0.02-0.20
_MIN_AVG_SCORE = 0.25


class ChunkQualityGate:
    """Classifies each sub-query result as strong, weak, or failed."""

    def __init__(
        self,
        min_chunks: int = _MIN_CHUNKS,
        min_avg_score: float = _MIN_AVG_SCORE,
    ) -> None:
        self._min_chunks = min_chunks
        self._min_avg_score = min_avg_score

    def evaluate(self, results: list[SubQueryResult]) -> list[SubQueryResult]:
        """Classify each sub-query result and update quality flags."""
        evaluated = []

        for result in results:
            if not result.success:
                evaluated.append(result)
                continue

            if not result.chunks:
                # zero chunks treated as info gap, no retry
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
