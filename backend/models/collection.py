"""Collection endpoint response models."""

from pydantic import BaseModel, ConfigDict, Field


class CollectionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    document_count: int = Field(default=0, ge=0)


class CollectionListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    collections: list[CollectionInfo]
