"""Authenticated identity attached to each request."""

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    """Identity resolved from the request's API key."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    key_id: str = Field(..., min_length=1)
    plan: str = Field(default="free")
