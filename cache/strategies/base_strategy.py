"""
Abstract base class for cache key generation strategies.

Two implementations:
    ExactStrategy   — SHA-256 hash of normalized inputs (sync, CPU only)
    SemanticStrategy — BGE embedding + Qdrant vector similarity (async, I/O)

The strategy is responsible for:
    1. Generating a deterministic key from the query + params
    2. Finding similar existing entries (semantic only)
    3. Returning enough metadata for the cache manager to decide hit/miss

Design notes:
    - make_key() is sync (Rule 2) — pure CPU, no I/O
    - find_similar() is async (Rule 1) — vector DB lookup
    - The exact strategy's find_similar() just checks key existence
    - The semantic strategy's find_similar() does ANN search in Qdrant
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class SimilarityMatch:
    """Result of a similarity search in the cache.

    Attributes:
        cache_key: The key of the matched entry in the backend.
        similarity_score: Cosine similarity (1.0 = exact, 0.0 = unrelated).
                          For exact strategy, this is always 1.0 on hit.
        tier: Confidence tier string (direct/high/partial/miss).
    """
    
    cache_key: str
    similarity_score: float
    tier: str

class BaseCacheStrategy(ABC):
    """Interface for cache key generation and similarity matching."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging (e.g. 'exact', 'semantic')."""

    @abstractmethod
    def make_key(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> str:
        """Generate a deterministic cache key from inputs.

        This is always sync (CPU only — Rule 2).
        The key must be deterministic: same inputs = same key.

        Args:
            query: The user's prompt (raw, before normalization).
            model_name: LLM model identifier (e.g. 'gemini-2.5-flash').
            temperature: Generation temperature (0.0-2.0).
            system_prompt_hash: Pre-computed SHA-256 of the system prompt.
                                Empty string if no system prompt.

        Returns:
            Deterministic string key (e.g. SHA-256 hex digest).
        """

    @abstractmethod
    async def find_similar(
        self,
        query: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> Optional[SimilarityMatch]:
        """Search for a similar cached entry.

        For ExactStrategy: checks if make_key() exists in the backend.
        For SemanticStrategy: embeds the query, searches Qdrant, applies
            tiered threshold logic.

        Args:
            query: The user's prompt.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of the system prompt.

        Returns:
            SimilarityMatch if a match is found above threshold, None otherwise.
        """

    @abstractmethod
    async def index_entry(
        self,
        query: str,
        cache_key: str,
        model_name: str,
        temperature: float,
        system_prompt_hash: str = "",
    ) -> None:
        """Index a new entry for future similarity lookups.

        For ExactStrategy: no-op (exact keys don't need a separate index).
        For SemanticStrategy: embeds the query and upserts into Qdrant.

        Args:
            query: The user's prompt that produced this cached response.
            cache_key: The key this entry is stored under in the backend.
            model_name: LLM model identifier.
            temperature: Generation temperature.
            system_prompt_hash: Pre-computed SHA-256 of the system prompt.
        """
    