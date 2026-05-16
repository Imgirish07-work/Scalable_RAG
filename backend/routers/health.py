"""Health, readiness, and Prometheus metrics endpoints."""

from fastapi import APIRouter, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse, Response

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness — OK as long as the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness — 200 once pipeline.initialize() completes."""
    if not getattr(request.app.state, "ready", False):
        return JSONResponse(content={"status": "not_ready"}, status_code=503)
    return {"status": "ready"}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
