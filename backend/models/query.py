"""Query endpoint request model. `user_id` is set server-side from auth."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from rag.models.rag_request import ConversationTurn


class ApiQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, max_length=10000)
    collection: str = Field(..., min_length=1, max_length=256)
    variant: Optional[str] = None
    conversation_history: Optional[list[ConversationTurn]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    include_sources: bool = True
    domain: Optional[str] = None
