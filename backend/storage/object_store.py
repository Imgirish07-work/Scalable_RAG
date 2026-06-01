"""Async S3-compatible object store wrapper for MinIO (dev) and AWS S3; credentials come from standard AWS env vars via aioboto3."""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

import aioboto3
from botocore.exceptions import ClientError

from backend.settings import StorageSettings, get_storage_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# 1 MiB chunk — big enough for throughput, small enough that embed can start before a 50 MB download completes
_STREAM_CHUNK_BYTES = 1024 * 1024

_NOT_FOUND_CODES = {"404", "NoSuchBucket", "NoSuchKey"}

# returned by create_bucket when another process won the race; treat as success
_BUCKET_EXISTS_CODES = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}


class ObjectStore:
    """Async wrapper over the S3 API. Stateless beyond the aioboto3 session."""

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._session = aioboto3.Session()

    @asynccontextmanager
    async def _client(self) -> AsyncIterator:
        """Yield a configured aioboto3 S3 client for internal (backend-side) ops."""
        async with self._session.client(
            "s3",
            endpoint_url=self._settings.s3_endpoint,
            region_name=self._settings.s3_region,
            use_ssl=self._settings.s3_use_ssl,
        ) as s3:
            yield s3

    @asynccontextmanager
    async def _signing_client(self) -> AsyncIterator:
        """Client configured with the public endpoint, used only for presigned URLs so the browser hits the host-reachable address."""
        async with self._session.client(
            "s3",
            endpoint_url=self._settings.effective_public_endpoint,
            region_name=self._settings.s3_region,
            use_ssl=self._settings.s3_use_ssl,
        ) as s3:
            yield s3

    async def ensure_bucket(self) -> None:
        """Create the bucket if missing and apply CORS. Idempotent and safe to call repeatedly."""
        bucket = self._settings.s3_bucket
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=bucket)
                logger.info("Bucket already exists | bucket=%s", bucket)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in _NOT_FOUND_CODES:
                    logger.exception("head_bucket failed | bucket=%s | code=%s", bucket, code)
                    raise
                try:
                    await s3.create_bucket(Bucket=bucket)
                    logger.info("Bucket created | bucket=%s", bucket)
                except ClientError as create_exc:
                    create_code = create_exc.response.get("Error", {}).get("Code", "")
                    if create_code in _BUCKET_EXISTS_CODES:
                        logger.info("Bucket created by concurrent worker | bucket=%s", bucket)
                    else:
                        raise

            # minio returns NotImplemented for PutBucketCors; cors is handled globally via MINIO_API_CORS_ALLOW_ORIGIN
            try:
                await s3.put_bucket_cors(
                    Bucket=bucket,
                    CORSConfiguration={
                        "CORSRules": [
                            {
                                "AllowedMethods": ["PUT", "GET", "HEAD"],
                                "AllowedOrigins": self._settings.s3_cors_origin_list,
                                "AllowedHeaders": ["*"],
                                "ExposeHeaders": ["ETag"],
                                "MaxAgeSeconds": 3600,
                            }
                        ]
                    },
                )
                logger.info(
                    "Bucket CORS applied | bucket=%s | origins=%s",
                    bucket, self._settings.s3_cors_origin_list,
                )
            except ClientError as cors_exc:
                cors_code = cors_exc.response.get("Error", {}).get("Code", "")
                if cors_code == "NotImplemented":
                    logger.info(
                        "Bucket CORS skipped — backend does not implement PutBucketCors "
                        "(MinIO uses MINIO_API_CORS_ALLOW_ORIGIN globally) | bucket=%s",
                        bucket,
                    )
                else:
                    raise

    async def generate_presigned_put_url(
        self,
        key: str,
        content_type: str,
        ttl_seconds: int | None = None,
    ) -> str:
        """Sign a PUT URL bound to the given key + Content-Type; MinIO rejects mismatches so the backend never re-validates."""
        ttl = ttl_seconds or self._settings.presigned_url_ttl_seconds
        async with self._signing_client() as s3:
            url = await s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self._settings.s3_bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=ttl,
            )
        logger.info(
            "Presigned PUT issued | key=%s | mime=%s | ttl=%ds", key, content_type, ttl,
        )
        return url

    async def generate_presigned_get_url(
        self,
        key: str,
        ttl_seconds: int | None = None,
    ) -> str:
        """Sign a GET URL so the browser can fetch the object directly without round-tripping the backend."""
        ttl = ttl_seconds or self._settings.presigned_url_ttl_seconds
        async with self._signing_client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self._settings.s3_bucket,
                    "Key": key,
                },
                ExpiresIn=ttl,
            )
        logger.info("Presigned GET issued | key=%s | ttl=%ds", key, ttl)
        return url

    async def head_object(self, key: str) -> dict:
        """Return MinIO metadata for a key (size, etag, content-type)."""
        async with self._client() as s3:
            return await s3.head_object(Bucket=self._settings.s3_bucket, Key=key)

    async def get_object_stream(self, key: str) -> AsyncIterator[bytes]:
        """Stream-download an object as async bytes chunks; consumer must fully iterate or aclose() to release the connection."""
        async with self._client() as s3:
            response = await s3.get_object(Bucket=self._settings.s3_bucket, Key=key)
            body = response["Body"]
            async for chunk in body.iter_chunks(_STREAM_CHUNK_BYTES):
                yield chunk

    async def delete_object(self, key: str) -> None:
        """Idempotent delete — missing keys do not raise."""
        async with self._client() as s3:
            try:
                await s3.delete_object(Bucket=self._settings.s3_bucket, Key=key)
                logger.info("Object deleted | key=%s", key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in _NOT_FOUND_CODES:
                    logger.info("Object already gone | key=%s", key)
                    return
                logger.exception("delete_object failed | key=%s | code=%s", key, code)
                raise


@lru_cache
def get_object_store() -> ObjectStore:
    """Process-wide singleton; safe because ObjectStore holds no per-request state."""
    return ObjectStore(get_storage_settings())
