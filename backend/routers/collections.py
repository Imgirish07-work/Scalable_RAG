"""GET /v1/collections — list Qdrant collections with descriptions and stats."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth.principal import Principal
from backend.deps import get_pipeline, get_principal
from backend.models.collection import CollectionInfo, CollectionListResponse
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["collections"])


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections(
    principal: Principal = Depends(get_principal),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> CollectionListResponse:
    """Return collections that exist in Qdrant, enriched with registry descriptions."""
    try:
        raw = await pipeline.list_collections()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to list collections | user_id=%s | error=%s",
            principal.user_id, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {type(exc).__name__}",
        )

    return CollectionListResponse(
        collections=[CollectionInfo(**item) for item in raw]
    )
