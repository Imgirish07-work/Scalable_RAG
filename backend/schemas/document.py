"""Request and response DTOs for the documents/ingest resource."""

from pydantic import BaseModel, ConfigDict

from pipeline.models.pipeline_request import IngestionResult


class DocumentCreatedView(BaseModel):
    """Response returned when a document has been ingested successfully."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    result: IngestionResult
