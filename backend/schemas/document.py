"""Request and response DTOs for the documents resource."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class UploadSessionRequest(BaseModel):
    """Client metadata for `POST /v1/ingest` — returns a presigned PUT URL."""

    file_name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    collection: str = Field(default="default", max_length=64)


class UploadSessionView(BaseModel):
    """Presigned URL the client must PUT the file against."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    s3_key: str
    presigned_url: str
    expires_at: datetime


class DownloadView(BaseModel):
    """Presigned URL the browser can GET to fetch the raw file for in-app viewing."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    file_name: str
    mime_type: str
    presigned_url: str
    expires_at: datetime


class FinalizeAck(BaseModel):
    """202 response from `POST /v1/documents/{id}/finalize`."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    status: str
    duplicate_of: Optional[str] = None


class DocumentDetailView(BaseModel):
    """Single document as returned by list/get."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    user_id: str
    file_name: str
    mime_type: str
    size_bytes: int
    collection: str
    status: str
    chunks_count: Optional[int] = None
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListView(BaseModel):
    """List envelope for `GET /v1/documents`."""

    model_config = ConfigDict(frozen=True)

    documents: list[DocumentDetailView]
    count: int
