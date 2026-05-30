"""End-to-end smoke test for the object store.

Exercises every public path: bucket bootstrap, presigned PUT, real HTTP PUT
against MinIO, HEAD, streaming download, and idempotent delete. Designed to
run inside the backend container, against the dev MinIO service.

Usage:
    docker compose exec backend python -m backend.storage.smoke_test
"""

import asyncio
import sys

import httpx

from backend.storage.object_store import get_object_store
from utils.logger import get_logger

logger = get_logger(__name__)

_TEST_KEY = "_smoke/object.bin"
_TEST_MIME = "application/octet-stream"
_TEST_PAYLOAD = b"smoke-test-payload-" * 4096  # ~80 KB


async def _run() -> int:
    store = get_object_store()

    logger.info("Step 1/6 | ensure_bucket")
    await store.ensure_bucket()

    logger.info("Step 2/6 | generate presigned PUT")
    url = await store.generate_presigned_put_url(
        key=_TEST_KEY, content_type=_TEST_MIME, ttl_seconds=60,
    )

    logger.info("Step 3/6 | PUT against presigned URL")
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.put(url, content=_TEST_PAYLOAD, headers={"Content-Type": _TEST_MIME})
        if resp.status_code not in (200, 204):
            logger.error(
                "PUT failed | status=%d | body=%s", resp.status_code, resp.text[:200],
            )
            return 1
        logger.info("PUT succeeded | status=%d", resp.status_code)

    logger.info("Step 4/6 | head_object")
    meta = await store.head_object(_TEST_KEY)
    size = meta.get("ContentLength")
    if size != len(_TEST_PAYLOAD):
        logger.error("HEAD size mismatch | got=%s | want=%d", size, len(_TEST_PAYLOAD))
        return 1
    logger.info("HEAD ok | size=%d | etag=%s", size, meta.get("ETag"))

    logger.info("Step 5/6 | stream-download")
    received = 0
    async for chunk in store.get_object_stream(_TEST_KEY):
        received += len(chunk)
    if received != len(_TEST_PAYLOAD):
        logger.error(
            "Stream size mismatch | got=%d | want=%d", received, len(_TEST_PAYLOAD),
        )
        return 1
    logger.info("Stream-download ok | bytes=%d", received)

    logger.info("Step 6/6 | delete_object (twice for idempotency)")
    await store.delete_object(_TEST_KEY)
    await store.delete_object(_TEST_KEY)

    logger.info("Smoke test PASSED")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception:
        logger.exception("Smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
