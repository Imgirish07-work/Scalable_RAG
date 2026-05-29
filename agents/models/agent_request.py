"""
Agent request and sub-query models.

Design:
    SubQuery is the atomic unit of work produced by the planner. Each
    SubQuery maps to exactly one RAG pipeline call. DecompositionPlan
    groups the sub-queries with routing metadata (parallel-safe flag,
    planner reasoning).

Chain of Responsibility:
    QueryPlanner produces DecompositionPlan → ChunkRetriever executes
    retrieval-only calls per SubQuery → SubQueryResult per sub-query.

Dependencies:
    pydantic
"""

# stdlib
from typing import Optional
from uuid import uuid4

# third-party
from pydantic import BaseModel, ConfigDict, Field


class SubQuery(BaseModel):
    """A single decomposed sub-query produced by the planner.

    Each SubQuery becomes one retrieval-only call inside the agent path.
    The planner decides the query text, the target collection (routed by
    the agent registry), and the variant.

    Attributes:
        query: The sub-query text for retrieval.
        collection: Target Qdrant collection name as picked by the planner.
        purpose: Brief description of what this sub-query resolves.
        sub_query_id: Unique ID for tracing.
    """

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
    """The planner's output — a list of sub-queries with metadata.

    Attributes:
        sub_queries: Ordered list of sub-queries to execute.
        reasoning: The planner's explanation of the decomposition.
        parallel_safe: Whether sub-queries can run concurrently.
    """

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
