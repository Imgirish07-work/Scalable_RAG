"""EventBus Protocol + InProcessEventBus (tests); RedisEventBus is production."""

import asyncio
from functools import lru_cache
from typing import AsyncIterator, Protocol, runtime_checkable

from backend.services.redis_event_bus import RedisEventBus
from backend.settings import worker_settings
from utils.logger import get_logger

logger = get_logger(__name__)


_TERMINAL_PHASES = {"ready", "failed", "duplicate"}
_SUBSCRIBER_QUEUE_SIZE = 128  # cap memory if an sse consumer falls behind


@runtime_checkable
class EventBus(Protocol):
    """Minimal pub/sub surface used by DocumentService and SSE handler."""

    async def publish(self, doc_id: str, event: dict) -> None: ...
    def subscribe(self, doc_id: str) -> AsyncIterator[dict]: ...
    async def close(self) -> None: ...


class InProcessEventBus:
    """Per-doc_id fan-out using `asyncio.Queue`. No persistence."""

    def __init__(self) -> None:
        self._topics: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, doc_id: str, event: dict) -> None:
        async with self._lock:
            subscribers = list(self._topics.get(doc_id, ()))
        if not subscribers:
            return
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "Event dropped (slow subscriber) | doc_id=%s | phase=%s",
                    doc_id, event.get("phase"),
                )

    async def subscribe(self, doc_id: str) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        async with self._lock:
            self._topics.setdefault(doc_id, set()).add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("phase") in _TERMINAL_PHASES:
                    return
        finally:
            async with self._lock:
                topic = self._topics.get(doc_id)
                if topic is None:
                    return
                topic.discard(queue)
                if not topic:
                    del self._topics[doc_id]

    async def close(self) -> None:
        return


@lru_cache
def get_event_bus() -> EventBus:
    """Process-wide singleton — Redis-backed for cross-process fan-out."""
    return RedisEventBus(redis_url=worker_settings.redis_url)
