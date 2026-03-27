"""Pipeline package — single entry point for the entire RAG system."""

from pipeline.rag_pipeline import RAGPipeline
from pipeline.models.pipeline_request import (
    IngestionResult,
    PipelineHealthStatus,
    PipelineQuery,
)
from pipeline.exceptions.pipeline_exceptions import (
    PipelineError,
    PipelineFallbackExhaustedError,
    PipelineIngestionError,
    PipelineInitError,
    PipelineValidationError,
)

__all__ = [
    "RAGPipeline",
    "PipelineQuery",
    "PipelineHealthStatus",
    "IngestionResult",
    "PipelineError",
    "PipelineFallbackExhaustedError",
    "PipelineIngestionError",
    "PipelineInitError",
    "PipelineValidationError",
]