"""Response DTOs for the collections resource."""

from pydantic import BaseModel, ConfigDict, Field


class CollectionView(BaseModel):
    """Single collection entry as returned to the client."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    document_count: int = Field(default=0, ge=0)


class CollectionListView(BaseModel):
    """Envelope around a list of collections."""

    model_config = ConfigDict(frozen=True)

    collections: list[CollectionView]
