"""Async in-process pub/sub for document ingestion progress events.

One producer (the DocumentService background task) publishes phase events
keyed by `doc_id`. Zero or more consumers (the SSE handler in step 3, the
test driver in step 2, future audit hooks) subscribe and receive events as
they arrive.

This is an in-process implementation — fine for single-worker dev. A drop-in
`RedisEventBus` will replace it when ingestion runs in separate worker
processes (Phase 3); call sites do not change because both implementations
satisfy the `EventBus` Protocol.
"""

import asyncio
from functools import lru_cache
from typing import AsyncIterator, Protocol, runtime_checkable

from utils.logger import get_logger

logger = get_logger(__name__)


# Subscribers exit their iteration loop naturally when one of these arrives.
_TERMINAL_PHASES = {"ready", "failed", "duplicate"}

# Bound per-subscriber backpressure; a slow consumer drops events rather than
# pinning unbounded memory on a misbehaving SSE client.
_SUBSCRIBER_QUEUE_SIZE = 128


@runtime_checkable
class EventBus(Protocol):
    """Minimal pub/sub surface used by DocumentService and SSE handler."""

    async def publish(self, doc_id: str, event: dict) -> None: ...
    def subscribe(self, doc_id: str) -> AsyncIterator[dict]: ...


class InProcessEventBus:
    """Per-doc_id fan-out using `asyncio.Queue`. No persistence."""

    def __init__(self) -> None:
        self._topics: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, doc_id: str, event: dict) -> None:
        async with self._lock:
            subscribers = list(self._topics.get(doc_id, ()))
        if not subscribers:
            # No listener — fire-and-forget. Late SSE subscribers can read the
            # terminal state from the DB instead.
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


@lru_cache
def get_event_bus() -> InProcessEventBus:
    """Process-wide singleton — state is fully encapsulated, no race risk."""
    return InProcessEventBus()
