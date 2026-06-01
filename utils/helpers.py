"""General-purpose utilities: ID generation, hashing, text manipulation, batching, retry."""

import hashlib
import random
import time
import functools
from typing import Any, Dict, List

import uuid_utils as uuid

from utils.logger import logger


def generate_unique_id() -> str:
    """Return a time-ordered UUID v7 (sql-friendly, monotonically increasing)."""
    return str(uuid.uuid7())


def hash_text(text: str) -> str:
    """Return SHA-256 hex digest of the normalised (lowercased, stripped) input text."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters, appending '...' when exceeded."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def timer(func):
    """Decorator that logs the wall-clock execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"'{func.__name__}' executed in {end - start:.4f}s")
        return result
    return wrapper


def safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """Traverse a nested dictionary without raising KeyError; returns default on miss."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


def flatten_list(nested_list: List) -> List:
    """Recursively flatten a nested list into a single flat list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(list_to_chunk: List, chunk_size: int) -> List[List]:
    """Split a list into consecutive sublists of at most chunk_size elements."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    return [
        list_to_chunk[i:i + chunk_size]
        for i in range(0, len(list_to_chunk), chunk_size)
    ]


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator that retries with exponential backoff plus up to 30% jitter."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    exponential_wait = base_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 0.3 * exponential_wait)
                    wait_time = min(exponential_wait + jitter, max_delay)

                    logger.warning(
                        f"'{func.__name__}' failed | "
                        f"attempt={attempt}/{max_retries} | "
                        f"wait={wait_time:.2f}s | "
                        f"error={e}"
                    )

                    if attempt < max_retries:
                        time.sleep(wait_time)

            logger.error(
                f"'{func.__name__}' failed after {max_retries} attempts | "
                f"last_error={last_exception}"
            )
            raise last_exception

        return wrapper
    return decorator
