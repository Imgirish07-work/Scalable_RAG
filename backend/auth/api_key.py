"""API-key generation and SHA-256 hashing.

SHA-256 (not bcrypt) because keys have ~190 bits of entropy — slow KDFs add
no security but break indexed lookup.
"""

import hashlib
import secrets

from backend.config import backend_settings


def generate_key() -> tuple[str, str, str]:
    """Return (full_key, key_prefix, key_hash). Full key is shown only once."""
    random_part = secrets.token_urlsafe(32)
    full_key = f"{backend_settings.api_key_prefix}{random_part}"
    key_prefix = full_key[: len(backend_settings.api_key_prefix) + 8]
    return full_key, key_prefix, hash_key(full_key)


def hash_key(key: str) -> str:
    """SHA-256 hex digest of a key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
