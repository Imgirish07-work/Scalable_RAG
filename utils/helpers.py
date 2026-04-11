"""
General-purpose utility functions used across the RAG pipeline.

Design:
    Collection of pure functions and decorators covering ID generation,
    hashing, text manipulation, batching, and retry logic. No shared state.
    Functions are intentionally small and independently testable.

Chain of Responsibility:
    Called by any module that needs these utilities (e.g., qdrant_store uses
    hash_text for deduplication; LLM clients use retry for resilience).
    No downstream module calls.

Dependencies:
    uuid_utils, utils.logger
"""

import hashlib
import random
import time
import functools
from typing import Any, Dict, List

import uuid_utils as uuid

from utils.logger import logger


def generate_unique_id() -> str:
    """Return a time-ordered UUID v7.

    UUID v7 is SQL-friendly — monotonically increasing values prevent
    B-tree index fragmentation and sort naturally by creation time.
    """
    return str(uuid.uuid7())


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of the normalised input text.

    Text is lowercased and stripped before hashing so minor casing or
    whitespace differences do not produce different cache keys.

    Args:
        text: Input string to hash.

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters for safe log output.

    Args:
        text: Text to truncate.
        max_length: Maximum allowed length before truncation.

    Returns:
        Original text if within limit, or the first max_length characters
        followed by '...'.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def timer(func):
    """Decorator that logs the wall-clock execution time of a function.

    Useful for tracking latency of LLM calls and I/O-bound operations.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"'{func.__name__}' executed in {end - start:.4f}s")
        return result
    return wrapper


def safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """Traverse a nested dictionary without raising KeyError.

    Args:
        data: The dictionary to traverse.
        *keys: Sequence of keys representing the path to the target value.
        default: Value returned when any key is missing or data is not a dict.

    Returns:
        The nested value if all keys exist, otherwise default.
    """
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


def flatten_list(nested_list: List) -> List:
    """Recursively flatten a nested list into a single flat list.

    Args:
        nested_list: A list that may contain other lists at any depth.

    Returns:
        A new flat list with all elements in depth-first order.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(list_to_chunk: List, chunk_size: int) -> List[List]:
    """Split a list into consecutive sublists of at most chunk_size elements.

    Args:
        list_to_chunk: The list to split.
        chunk_size: Maximum number of elements per sublist.

    Returns:
        List of sublists; the last sublist may be smaller than chunk_size.

    Raises:
        ValueError: If chunk_size is not greater than 0.
    """
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
    """Decorator that retries a function with exponential backoff and jitter.

    Wait formula per attempt:
        wait = min(base_delay * 2^(attempt-1), max_delay) + uniform_jitter

    Jitter is up to 30% of the exponential component, preventing thundering-
    herd problems when multiple workers retry simultaneously.

    Args:
        max_retries: Maximum number of retry attempts (must be >= 1).
        base_delay: Starting delay in seconds before the first retry.
        max_delay: Upper cap on the computed delay in seconds.
        exceptions: Tuple of exception types that trigger a retry.

    Raises:
        ValueError: If max_retries is less than 1.
    """
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
