"""Document ingestion pipeline — download, hash, dedup, chunk, embed, upsert."""

import asyncio
import hashlib
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import magic

from backend.metrics import (
    ingest_chunks_total,
    ingest_jobs_failed_total,
    ingest_jobs_inflight,
    ingest_total,
)
from backend.repositories.database import session_scope
from backend.repositories.document_repository import DocumentRepository
from backend.services.event_bus import EventBus
from backend.settings import worker_settings
from backend.storage.object_store import ObjectStore
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 8.0
_RETRY_JITTER_RATIO = 0.25


class DataLevelIngestionError(Exception):
    """Bad input (MIME mismatch, corrupt file) — no retry; blob is purged."""


class IngestionService:
    """Runs the full ingest pipeline for one document."""

    def __init__(
        self,
        object_store: ObjectStore,
        pipeline: RAGPipeline,
        event_bus: EventBus,
    ) -> None:
        self._store = object_store
        self._pipeline = pipeline
        self._bus = event_bus

    async def run(
        self, doc_id: str, user_id: str, request_id: Optional[str] = None,
    ) -> None:
        publish = self._make_publisher(doc_id, request_id)
        progress = _ChunkProgressEmitter(
            publish=publish,
            every_n=worker_settings.progress_publish_every_n_chunks,
            min_interval_ms=worker_settings.progress_publish_min_interval_ms,
        )
        start = time.perf_counter()
        temp_path: Optional[Path] = None
        ingest_jobs_inflight.inc()

        try:
            async with self._with_repo() as repo:
                document = await repo.find_by_id(doc_id)
                if document is None:
                    raise RuntimeError(f"Document vanished: {doc_id}")
                s3_key = document.s3_key
                expected_mime = document.mime_type
                file_ext = Path(document.file_name).suffix
                logical_collection = document.collection
                await repo.update_status(doc_id, "processing")

            await publish("processing")

            metadata = await self._retry(self._store.head_object, s3_key)
            await publish(
                "downloading",
                size_bytes=int(metadata.get("ContentLength", 0)),
            )

            temp_path, content_hash = await self._download_and_hash(
                s3_key=s3_key, file_ext=file_ext, doc_id=doc_id,
            )
            await publish("hashed", content_hash=content_hash)

            self._verify_mime_from_disk(temp_path, expected=expected_mime)

            async with self._with_repo() as repo:
                existing = await repo.find_active_by_content_hash(
                    user_id, content_hash,
                )
            duplicate_id = (
                existing.id
                if existing is not None and existing.id != doc_id
                else None
            )
            if duplicate_id is not None:
                await self._handle_duplicate(
                    doc_id=doc_id,
                    s3_key=s3_key,
                    existing_id=duplicate_id,
                    publish=publish,
                )
                return

            async with self._with_repo() as repo:
                await repo.set_content_hash(doc_id, content_hash)

            await publish("chunking")
            ingestion = await self._retry(
                self._pipeline.ingest,
                str(temp_path), logical_collection, user_id, doc_id,
                on_batch_progress=progress.emit,
            )

            # guarded transition — skip success if sweeper already moved the row
            async with self._with_repo() as repo:
                committed = await repo.mark_ready_if_processing(
                    doc_id, chunks_count=ingestion.chunks_stored,
                )

            if not committed:
                logger.warning(
                    "Ingest finished but row was no longer 'processing' | doc_id=%s",
                    doc_id,
                )
                return

            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            ingest_total.labels(outcome="ready").inc()
            ingest_chunks_total.inc(ingestion.chunks_stored)
            await publish(
                "ready",
                chunks_count=ingestion.chunks_stored,
                elapsed_ms=elapsed_ms,
            )
            logger.info(
                "Ingestion complete | doc_id=%s | chunks=%d | elapsed_ms=%.1f",
                doc_id, ingestion.chunks_stored, elapsed_ms,
            )

        except Exception as exc:
            logger.exception("Ingestion failed | doc_id=%s", doc_id)
            try:
                if isinstance(exc, DataLevelIngestionError):
                    await self._rollback(doc_id, reason=str(exc))
                else:
                    await self._mark_failed_in_dlq(doc_id, exc)
            except Exception:
                logger.exception(
                    "Failure handler itself failed | doc_id=%s", doc_id,
                )
            ingest_total.labels(outcome="failed").inc()
            try:
                await publish(
                    "failed", reason=type(exc).__name__, message=str(exc),
                )
            except Exception:
                logger.exception(
                    "Failed event publish failed | doc_id=%s", doc_id,
                )

        finally:
            self._unlink_quietly(temp_path)
            ingest_jobs_inflight.dec()

    @asynccontextmanager
    async def _with_repo(self):
        async with session_scope() as session:
            yield DocumentRepository(session)

    async def _download_and_hash(
        self, s3_key: str, file_ext: str, doc_id: str,
    ) -> tuple[Path, str]:
        hasher = hashlib.sha256()
        fd, temp_path_str = tempfile.mkstemp(
            suffix=file_ext, prefix=f"{doc_id}_",
        )
        os.close(fd)
        temp_path = Path(temp_path_str)
        try:
            with temp_path.open("wb") as out:
                async for chunk in self._store.get_object_stream(s3_key):
                    hasher.update(chunk)
                    out.write(chunk)
            return temp_path, hasher.hexdigest()
        except Exception:
            self._unlink_quietly(temp_path)
            raise

    def _verify_mime_from_disk(self, path: Path, expected: str) -> None:
        sniffed = magic.from_file(str(path), mime=True)
        if sniffed != expected:
            raise DataLevelIngestionError(
                f"MIME mismatch — declared='{expected}' sniffed='{sniffed}'"
            )

    async def _handle_duplicate(
        self,
        doc_id: str,
        s3_key: str,
        existing_id: str,
        publish: Callable[..., Awaitable[None]],
    ) -> None:
        await self._cascade_delete(doc_id, s3_key)
        async with self._with_repo() as repo:
            await repo.hard_delete(doc_id)
        ingest_total.labels(outcome="duplicate").inc()
        await publish("duplicate", duplicate_of=existing_id)
        logger.info(
            "Duplicate suppressed | doc_id=%s | existing=%s",
            doc_id, existing_id,
        )

    async def _rollback(self, doc_id: str, reason: str) -> None:
        async with self._with_repo() as repo:
            document = await repo.find_by_id(doc_id)
        if document is None:
            return
        await self._cascade_delete(doc_id, document.s3_key)
        async with self._with_repo() as repo:
            await repo.hard_delete(doc_id)
        logger.info("Rollback complete | doc_id=%s | reason=%s", doc_id, reason)

    async def _mark_failed_in_dlq(self, doc_id: str, exc: Exception) -> None:
        try:
            await self._delete_qdrant_chunks(doc_id)
        except Exception:
            logger.exception(
                "Qdrant cleanup failed during DLQ mark | doc_id=%s", doc_id,
            )
        error_message = f"{type(exc).__name__}: {exc}"[:500]
        try:
            async with self._with_repo() as repo:
                await repo.mark_failed(doc_id, error_message=error_message)
        except Exception:
            logger.exception("Failed to mark DLQ status | doc_id=%s", doc_id)
        ingest_jobs_failed_total.labels(reason=type(exc).__name__).inc()

    async def _cascade_delete(self, doc_id: str, s3_key: str) -> None:
        try:
            await self._delete_qdrant_chunks(doc_id)
        except Exception:
            logger.exception("Qdrant cleanup failed | doc_id=%s", doc_id)
        try:
            await self._retry(self._store.delete_object, s3_key, attempts=2)
        except Exception:
            logger.exception(
                "MinIO cleanup failed | doc_id=%s | key=%s", doc_id, s3_key,
            )

    async def _delete_qdrant_chunks(self, doc_id: str) -> None:
        store = getattr(self._pipeline, "_store", None)
        if store is None:
            return
        await store.delete_by_doc_id(doc_id)

    def _make_publisher(
        self, doc_id: str, request_id: Optional[str],
    ) -> Callable[..., Awaitable[None]]:
        async def publish(phase: str, **payload) -> None:
            event = {
                "doc_id": doc_id,
                "phase": phase,
                "ts": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                **payload,
            }
            await self._bus.publish(doc_id, event)
        return publish

    async def _retry(
        self, fn, *args, attempts: int = _RETRY_ATTEMPTS, **kwargs,
    ):
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    raise
                base = min(
                    _RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
                    _RETRY_MAX_DELAY_S,
                )
                jitter = base * _RETRY_JITTER_RATIO * (2 * random.random() - 1)
                wait = max(0.0, base + jitter)
                logger.warning(
                    "Retrying | fn=%s | attempt=%d/%d | wait=%.2fs | error=%s",
                    getattr(fn, "__name__", repr(fn)),
                    attempt, attempts, wait, exc,
                )
                await asyncio.sleep(wait)
        raise last_exc  # unreachable; satisfies type checker

    def _unlink_quietly(self, path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            if path.exists():
                path.unlink()
        except Exception:
            logger.warning("Failed to delete temp file | path=%s", path)


class _ChunkProgressEmitter:
    """Throttled progress publisher; always emits first and last."""

    def __init__(
        self,
        publish: Callable[..., Awaitable[None]],
        every_n: int,
        min_interval_ms: int,
    ) -> None:
        self._publish = publish
        self._every_n = max(1, every_n)
        self._min_interval_s = max(0.0, min_interval_ms / 1000.0)
        self._last_processed = -1
        self._last_emit_t = 0.0

    async def emit(self, processed: int, total: int) -> None:
        now = time.monotonic()
        is_first = self._last_processed < 0
        is_last = total > 0 and processed >= total
        crossed_n = processed - self._last_processed >= self._every_n
        crossed_t = (now - self._last_emit_t) >= self._min_interval_s
        if not (is_first or is_last or crossed_n or crossed_t):
            return
        self._last_processed = processed
        self._last_emit_t = now
        await self._publish(
            "embedding",
            chunks_processed=processed,
            chunks_total=total,
        )
