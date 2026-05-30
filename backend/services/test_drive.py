"""End-to-end exercise of the document service via the live HTTP API.

Runs the full happy path: create session → direct upload to MinIO → finalize
→ poll until ready → verify in list + collections → soft-delete → verify 404.

Usage (backend must be running and healthy):
    docker compose exec backend python -m backend.services.test_drive
"""

import asyncio
import json
import sys
import time

import aioboto3
import httpx

from backend.settings import storage_settings
from utils.logger import get_logger

logger = get_logger(__name__)


_BACKEND_URL = "http://localhost:8000"
_TEST_COLLECTION = "test-drive"
_TEST_FILE_NAME = "test_drive.txt"
_TEST_MIME = "text/plain"
_TEST_PAYLOAD = (
    b"Scalable RAG test-drive document.\n"
    b"This file exists only to exercise the upload + ingest pipeline end-to-end.\n"
    b"It is small enough to chunk into a single piece for fast verification.\n"
)
_POLL_INTERVAL_S = 1.0
_POLL_TIMEOUT_S = 120.0


async def _upload_directly_to_minio(s3_key: str) -> None:
    """Simulate the browser-direct PUT, but from inside the docker network.

    Uses the internal `s3_endpoint` so the test works without DNS for the
    public hostname. The real-browser path (presigned URL → host endpoint)
    is exercised separately when this code is wired into a UI in step 3.
    """
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=storage_settings.s3_endpoint,
        region_name=storage_settings.s3_region,
        use_ssl=storage_settings.s3_use_ssl,
    ) as s3:
        await s3.put_object(
            Bucket=storage_settings.s3_bucket,
            Key=s3_key,
            Body=_TEST_PAYLOAD,
            ContentType=_TEST_MIME,
        )


async def _read_one_sse_event(client: httpx.AsyncClient, doc_id: str) -> dict:
    """Open the SSE stream and return the first non-keepalive event payload."""
    async with client.stream("GET", f"/v1/documents/{doc_id}/events") as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"SSE open failed | status={resp.status_code}")
        data_lines: list[str] = []
        async for line in resp.aiter_lines():
            if line.startswith(":"):
                continue  # keepalive comment
            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
                continue
            if line == "" and data_lines:
                payload = "\n".join(data_lines)
                return json.loads(payload)


async def _poll_until_terminal(client: httpx.AsyncClient, doc_id: str) -> dict:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    last_status = None
    while time.monotonic() < deadline:
        resp = await client.get(f"/v1/documents/{doc_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"GET failed | status={resp.status_code}")
        body = resp.json()
        if body["status"] != last_status:
            logger.info("Polling | doc_id=%s | status=%s", doc_id, body["status"])
            last_status = body["status"]
        if body["status"] in ("ready", "failed"):
            return body
        await asyncio.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(f"Doc {doc_id} never reached terminal state")


async def _run() -> int:
    async with httpx.AsyncClient(base_url=_BACKEND_URL, timeout=120.0) as client:
        logger.info("Step 1/8 | POST /v1/ingest — start upload session")
        create_resp = await client.post(
            "/v1/ingest",
            json={
                "file_name": _TEST_FILE_NAME,
                "mime_type": _TEST_MIME,
                "size_bytes": len(_TEST_PAYLOAD),
                "collection": _TEST_COLLECTION,
            },
        )
        if create_resp.status_code != 201:
            logger.error(
                "Create failed | status=%d | body=%s",
                create_resp.status_code, create_resp.text,
            )
            return 1
        session = create_resp.json()
        doc_id = session["doc_id"]
        s3_key = session["s3_key"]
        logger.info("Session created | doc_id=%s | key=%s", doc_id, s3_key)

        logger.info("Step 2/8 | PUT bytes to MinIO directly (simulates browser)")
        await _upload_directly_to_minio(s3_key)

        logger.info("Step 3/8 | POST /v1/documents/{id}/finalize")
        fin_resp = await client.post(f"/v1/documents/{doc_id}/finalize")
        if fin_resp.status_code != 202:
            logger.error(
                "Finalize failed | status=%d | body=%s",
                fin_resp.status_code, fin_resp.text,
            )
            return 1
        logger.info("Finalize acknowledged | ack=%s", fin_resp.json())

        logger.info("Step 4/8 | Poll GET /v1/documents/{id} until ready")
        final = await _poll_until_terminal(client, doc_id)
        if final["status"] != "ready":
            logger.error("Doc reached non-ready terminal | final=%s", final)
            return 1
        logger.info(
            "Document ready | doc_id=%s | chunks=%d", doc_id, final["chunks_count"],
        )

        logger.info("Step 5/8 | GET /v1/documents/{id}/events — late-subscriber snapshot")
        snapshot = await _read_one_sse_event(client, doc_id)
        if snapshot.get("phase") != "ready" or not snapshot.get("snapshot"):
            logger.error("SSE snapshot wrong | got=%s", snapshot)
            return 1
        logger.info("SSE snapshot ok | phase=%s | chunks=%s",
                    snapshot.get("phase"), snapshot.get("chunks_count"))

        logger.info("Step 6/8 | GET /v1/documents — verify in list")
        list_resp = await client.get(
            "/v1/documents", params={"collection": _TEST_COLLECTION},
        )
        if list_resp.status_code != 200:
            logger.error("List failed | status=%d", list_resp.status_code)
            return 1
        ids = [d["doc_id"] for d in list_resp.json()["documents"]]
        if doc_id not in ids:
            logger.error("Doc missing from list | doc_id=%s | got=%s", doc_id, ids)
            return 1
        logger.info("Doc present in list | count=%d", len(ids))

        logger.info("Step 7/8 | GET /v1/collections — verify collection appears")
        coll_resp = await client.get("/v1/collections")
        if coll_resp.status_code != 200:
            logger.error("Collections failed | status=%d", coll_resp.status_code)
            return 1
        names = [c["name"] for c in coll_resp.json()["collections"]]
        if _TEST_COLLECTION not in names:
            logger.error(
                "Collection missing | want=%s | got=%s", _TEST_COLLECTION, names,
            )
            return 1
        logger.info("Collection visible | name=%s", _TEST_COLLECTION)

        logger.info("Step 8/8 | DELETE /v1/documents/{id} + verify 404")
        del_resp = await client.delete(f"/v1/documents/{doc_id}")
        if del_resp.status_code not in (200, 204):
            logger.error(
                "Delete failed | status=%d | body=%s",
                del_resp.status_code, del_resp.text,
            )
            return 1
        get_resp = await client.get(f"/v1/documents/{doc_id}")
        if get_resp.status_code != 404:
            logger.error("Expected 404 after delete | status=%d", get_resp.status_code)
            return 1
        logger.info("Soft-delete verified | doc_id=%s", doc_id)

    logger.info("Test drive PASSED")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception:
        logger.exception("Test drive FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
