"""Shared API response models."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str = Field(...)
    request_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None
