"""Agent request and sub-query models."""

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SubQuery(BaseModel):
    """A single decomposed sub-query produced by the planner."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ...,
        min_length=1,
        description="Sub-query text for retrieval.",
    )
    collection: str = Field(
        ...,
        min_length=1,
        description="Target Qdrant collection name.",
    )
    variant: Optional[str] = Field(
        default=None,
        description="RAG variant override. None uses settings default.",
    )
    purpose: str = Field(
        default="",
        description="Brief description of what this sub-query resolves.",
    )
    sub_query_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for tracing.",
    )


class DecompositionPlan(BaseModel):
    """The planner's output: a list of sub-queries with metadata."""

    model_config = ConfigDict(frozen=True)

    sub_queries: list[SubQuery] = Field(
        ...,
        min_length=1,
        description="Ordered list of sub-queries to execute.",
    )
    reasoning: str = Field(
        default="",
        description="Planner's explanation of the decomposition strategy.",
    )
    parallel_safe: bool = Field(
        default=True,
        description="Whether sub-queries are independent and can run concurrently.",
    )
