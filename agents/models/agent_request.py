"""
Agent request and sub-query models.

Design:
    SubQuery is the atomic unit of work produced by the planner. Each
    SubQuery maps to exactly one RAG pipeline call. DecompositionPlan
    groups the sub-queries with routing metadata (parallel-safe flag,
    planner reasoning).

Chain of Responsibility:
    QueryPlanner produces DecompositionPlan → ParallelRetriever iterates
    SubQuery items → SubQuery.to_rag_request() converts each to RAGRequest
    → pipeline.query_raw() executes.

Dependencies:
    pydantic, rag.models.rag_request
"""

# stdlib
from typing import Optional
from uuid import uuid4

# third-party
from pydantic import BaseModel, ConfigDict, Field

# internal
from rag.models.rag_request import RAGConfig, RAGRequest


class SubQuery(BaseModel):
    """A single decomposed sub-query produced by the planner.

    Each SubQuery becomes one RAG pipeline call. The planner
    decides the query text, target collection, and variant.

    Attributes:
        query: The sub-query text for retrieval.
        collection: Target Qdrant collection name.
        variant: RAG variant to use. None uses settings default.
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
        description="RAG variant: 'simple', 'corrective', 'chain'. None uses default.",
    )
    purpose: str = Field(
        default="",
        description="Brief description of what this sub-query resolves.",
    )
    sub_query_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for tracing.",
    )

    def to_rag_request(self, parent_request_id: str) -> RAGRequest:
        """Convert to internal RAGRequest for pipeline execution.

        Args:
            parent_request_id: The parent query's request ID for tracing.

        Returns:
            RAGRequest ready for pipeline.query_raw().
        """
        config = RAGConfig(rag_variant=self.variant)

        return RAGRequest(
            query=self.query,
            collection_name=self.collection,
            config=config,
            request_id=f"{parent_request_id}::{self.sub_query_id}",
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
