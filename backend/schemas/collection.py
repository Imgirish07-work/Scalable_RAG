"""Response DTOs for the collections resource (derived from the documents table)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CollectionView(BaseModel):
    """One logical collection (folder) the user has documents in."""

    model_config = ConfigDict(frozen=True)

    name: str
    document_count: int = Field(ge=0)
    last_updated: datetime


class CollectionListView(BaseModel):
    """Envelope for `GET /v1/collections`."""

    model_config = ConfigDict(frozen=True)

    collections: list[CollectionView]
