"""POST /v1/query — synchronous RAG query."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.auth.principal import Principal
from backend.deps import get_pipeline, get_principal
from backend.models.query import ApiQueryRequest
from backend.observability.metrics import queries_total
from pipeline.exceptions.pipeline_exceptions import PipelineValidationError
from pipeline.models.pipeline_request import PipelineQuery
from pipeline.rag_pipeline import RAGPipeline
from rag.models.rag_response import RAGResponse
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=RAGResponse)
async def query(
    body: ApiQueryRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> RAGResponse:
    """Execute a RAG query; user_id is pulled from the Principal, not the body."""
    request_id = getattr(request.state, "request_id", None)

    try:
        pipeline_query = PipelineQuery(
            query=body.query,
            collection=body.collection,
            variant=body.variant,
            conversation_history=body.conversation_history,
            temperature=body.temperature,
            top_k=body.top_k,
            include_sources=body.include_sources,
            domain=body.domain,
            request_id=request_id,
            user_id=principal.user_id,
        )
        response = await pipeline.query(pipeline_query)
    except PipelineValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Query failed | request_id=%s | user_id=%s | error=%s",
            request_id, principal.user_id, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {type(exc).__name__}",
        )

    try:
        queries_total.labels(
            variant=response.rag_variant or "unknown",
            cache_hit=str(response.cache_hit).lower(),
        ).inc()
    except Exception:
        logger.warning("Failed to record query metric", exc_info=True)

    return response
