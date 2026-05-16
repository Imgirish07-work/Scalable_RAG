"""Ingest endpoint response model."""

from pydantic import BaseModel, ConfigDict

from pipeline.models.pipeline_request import IngestionResult


class ApiIngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    result: IngestionResult
