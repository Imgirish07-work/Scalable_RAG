"""FastAPI dependencies."""

from fastapi import HTTPException, Request, status

from pipeline.rag_pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    """Return the lifespan-created RAGPipeline; 503 if not ready."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return pipeline
