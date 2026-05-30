"""FastAPI dependency-injection providers."""

from fastapi import HTTPException, Request, status

from pipeline.rag_pipeline import RAGPipeline


# Hardcoded while auth is removed; replaced by real identity extraction in Phase 8.
_DEV_USER_ID = "dev-user"


def get_pipeline(request: Request) -> RAGPipeline:
    """Return the lifespan-created RAGPipeline; 503 if it is not ready."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return pipeline


def get_current_user_id() -> str:
    """Return the calling user's id. Single source of truth across controllers
    so the Phase 8 auth swap is a one-line change."""
    return _DEV_USER_ID
