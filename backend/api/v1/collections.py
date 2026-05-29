"""GET /v1/collections — list Qdrant collections with descriptions and stats."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_pipeline
from backend.schemas.collection import CollectionListView, CollectionView
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["collections"])


@router.get("/collections", response_model=CollectionListView)
async def list_collections(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> CollectionListView:
    """Return collections that exist in Qdrant, enriched with registry descriptions."""
    try:
        raw = await pipeline.list_collections()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list collections | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {type(exc).__name__}",
        )

    return CollectionListView(
        collections=[CollectionView(**item) for item in raw]
    )
