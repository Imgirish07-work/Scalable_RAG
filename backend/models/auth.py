"""Auth endpoint request/response models."""

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class IssueKeyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str = Field(..., min_length=3, max_length=255)
    name: Optional[str] = Field(default=None, max_length=128)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError(f"Invalid email format: {v}")
        return v


class IssueKeyResponse(BaseModel):
    """Issued once on key creation. The full `key` is never queryable later."""

    model_config = ConfigDict(frozen=True)

    key: str
    key_id: str
    key_prefix: str
    user_id: str
    email: str
