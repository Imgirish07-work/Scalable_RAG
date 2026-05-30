"""Logical-collection listing — derived from the documents table per user."""

from fastapi import Depends

from backend.repositories.document_repository import (
    DocumentRepository,
    get_document_repository,
)
from backend.schemas.collection import CollectionListView, CollectionView
from utils.logger import get_logger

logger = get_logger(__name__)


class CollectionService:
    """Reads the user's live documents and projects them into collections."""

    def __init__(self, repo: DocumentRepository) -> None:
        self._repo = repo

    async def list_for_user(self, user_id: str) -> CollectionListView:
        rows = await self._repo.list_collections_for_user(user_id)
        collections = [
            CollectionView(name=name, document_count=count, last_updated=updated)
            for name, count, updated in rows
        ]
        logger.info(
            "Collections listed | user_id=%s | count=%d", user_id, len(collections),
        )
        return CollectionListView(collections=collections)


def get_collection_service(
    repo: DocumentRepository = Depends(get_document_repository),
) -> CollectionService:
    return CollectionService(repo)
