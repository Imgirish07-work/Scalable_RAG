"""Request-ID middleware + per-request access log."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.config import backend_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID, then access-log method/path/status/latency."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Request crashed | request_id=%s | %s %s | elapsed_ms=%.1f",
                request_id, request.method, request.url.path, elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id

        if backend_settings.log_requests:
            logger.info(
                "%s %s -> %d | request_id=%s | %.1f ms",
                request.method, request.url.path, response.status_code,
                request_id, elapsed_ms,
            )

        return response
