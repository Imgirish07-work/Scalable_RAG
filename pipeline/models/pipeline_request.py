"""Simplified external-facing request and response models for the pipeline."""

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import settings
from rag.models.rag_request import ConversationTurn, RAGConfig, RAGRequest


class PipelineQuery(BaseModel):
    """Simplified query interface for external callers."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's natural language question.",
    )
    collection: Optional[str] = Field(
        default=None,
        max_length=256,
        description=(
            "Logical collection ('folder') filter. None searches the user's "
            "entire corpus."
        ),
    )
    variant: Optional[str] = Field(
        default=None,
        description="RAG variant: 'simple'. None uses default.",
    )
    conversation_history: Optional[list[ConversationTurn]] = Field(
        default=None,
        description="Prior conversation turns for multi-turn context.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature override.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of chunks to retrieve.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source chunks in response.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Caller-provided request ID for tracing.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain profile: 'technical' or 'story'. None = no profile.",
    )
    user_id: str = Field(
        default="",
        description="Authenticated user ID. Empty string = no user scoping.",
    )

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: Optional[str]) -> Optional[str]:
        """Validate variant name against known variants."""
        if v is None:
            return None
        allowed = {"simple"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            msg = f"Unknown variant '{v}'. Allowed: {sorted(allowed)}"
            raise ValueError(msg)
        return normalized

    def to_rag_request(self) -> RAGRequest:
        """Convert to internal RAGRequest + RAGConfig."""
        from rag.domain_profiles import apply_domain_profile

        # ragconfig top_k and temperature are non-optional, omit when none to use field defaults
        config_kwargs: dict = {
            "rag_variant": self.variant,
            "include_sources": self.include_sources,
            "domain": self.domain,
        }
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        # caller-supplied values above win over domain defaults
        config_kwargs = apply_domain_profile(config_kwargs, self.domain)

        config = RAGConfig(**config_kwargs)

        return RAGRequest(
            query=self.query,
            collection_name=settings.qdrant_collection_name,
            config=config,
            conversation_history=self.conversation_history,
            request_id=self.request_id or str(uuid4()),
            user_id=self.user_id,
            logical_collection=self.collection or "",
        )


class PipelineHealthStatus(BaseModel):
    """Health check result for the pipeline and its subsystems."""

    model_config = ConfigDict(frozen=True)

    ready: bool = Field(
        ...,
        description="True only if all critical subsystems are healthy.",
    )
    llm: str = Field(
        ...,
        description="LLM provider status: 'ok' or error description.",
    )
    vector_store: str = Field(
        ...,
        description="Qdrant status: 'ok' or error description.",
    )
    cache: str = Field(
        ...,
        description="Cache status: 'ok', 'degraded', or error description.",
    )
    details: dict = Field(
        default_factory=dict,
        description="Optional extra diagnostic info.",
    )


class IngestionResult(BaseModel):
    """Result of a document ingestion operation."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    collection: str
    chunks_stored: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
