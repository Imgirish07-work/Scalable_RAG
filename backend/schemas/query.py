"""Request DTO for POST /v1/query."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from rag.models.rag_request import ConversationTurn


class QueryRequest(BaseModel):
    """Body shape accepted by the /v1/query endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(
        default=None,
        max_length=256,
        description=(
            "Logical collection ('folder') filter. Omit to search the "
            "caller's entire corpus."
        ),
    )
    variant: Optional[str] = None
    conversation_history: Optional[list[ConversationTurn]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    include_sources: bool = True
    domain: Optional[str] = None
