"""GET /v1/collections — logical collections derived from the user's documents."""

from fastapi import APIRouter, Depends

from backend.dependencies import get_current_user_id
from backend.schemas.collection import CollectionListView
from backend.services.collection_service import (
    CollectionService,
    get_collection_service,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["collections"])


@router.get("/collections", response_model=CollectionListView)
async def list_collections(
    user_id: str = Depends(get_current_user_id),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionListView:
    """Return logical collections the user has at least one live document in."""
    return await service.list_for_user(user_id)
