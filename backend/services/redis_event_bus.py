"""Redis pub/sub EventBus — used for cross-process ingestion progress fan-out."""

import asyncio
import json
from contextlib import suppress
from typing import AsyncIterator, Optional

import redis.asyncio as redis

from backend.settings import worker_settings
from utils.logger import get_logger

logger = get_logger(__name__)


_TERMINAL_PHASES = {"ready", "failed", "duplicate"}


class RedisEventBus:
    """Pub/sub fan-out over Redis; satisfies the EventBus Protocol."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Optional[redis.Redis] = None
        self._connect_lock = asyncio.Lock()

    async def _client_or_connect(self) -> redis.Redis:
        if self._client is not None:
            return self._client
        async with self._connect_lock:
            if self._client is None:
                self._client = redis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info(
                    "Redis event bus connected | url=%s", self._redis_url,
                )
        return self._client

    async def publish(self, doc_id: str, event: dict) -> None:
        channel = worker_settings.events_channel(doc_id)
        try:
            client = await self._client_or_connect()
            await client.publish(channel, json.dumps(event, default=str))
        except Exception:
            logger.warning(
                "Event publish failed | doc_id=%s | phase=%s",
                doc_id, event.get("phase"),
                exc_info=True,
            )

    async def subscribe(self, doc_id: str) -> AsyncIterator[dict]:
        channel = worker_settings.events_channel(doc_id)
        try:
            client = await self._client_or_connect()
        except Exception:
            logger.exception(
                "Redis subscribe failed (connect) | doc_id=%s", doc_id,
            )
            return

        pubsub = client.pubsub()
        try:
            await pubsub.subscribe(channel)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                event = self._parse(doc_id, message.get("data"))
                if event is None:
                    continue
                yield event
                if event.get("phase") in _TERMINAL_PHASES:
                    return
        finally:
            with suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

    async def close(self) -> None:
        if self._client is None:
            return
        with suppress(Exception):
            await self._client.aclose()
        self._client = None
        logger.info("Redis event bus closed")

    @staticmethod
    def _parse(doc_id: str, data) -> Optional[dict]:
        try:
            return json.loads(data)
        except (TypeError, ValueError):
            logger.warning(
                "Malformed event dropped | doc_id=%s | data=%r", doc_id, data,
            )
            return None
