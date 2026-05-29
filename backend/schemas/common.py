"""Shared response envelopes used across multiple resources."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Generic error envelope. Will be replaced by RFC 7807 ProblemView in Phase 7."""

    model_config = ConfigDict(frozen=True)

    detail: str = Field(...)
    request_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None
